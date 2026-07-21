from __future__ import annotations

import numpy as np

from owl.core.actions import Action
from owl.core.config import SimulationConfig
from owl.core.constants import DEFAULT_READOUT_DTYPE
from owl.core.state import WorldState, field_shape
from owl.raqic.aggregation import (
    aggregate_raqic_patches_to_global,
    aggregate_raqic_records_to_patches,
    update_raqic_global_intention,
    update_raqic_patch_intention,
)
from owl.raqic.dense_feature_extraction import (
    build_dense_feature_batch_gpu,
    build_dense_feature_batch_numpy,
)
from owl.raqic.engine import OWLRAQICDecisionBatch
from owl.raqic.precision import raqic_numpy_dtype
from owl.raqic.state import ensure_raqic_fields, reset_raqic_audit_flags
from owl.raqic.topdown import dispatch_raqic_intention_to_cells
from owl_raqic.gpu.actualization_extensions import (
    ActualizationExtensionConfig,
    aggregate_action_phase_context,
)
from owl_raqic.gpu.backend import detect_cupy
from owl_raqic.gpu.decision_engine import RAQICDenseDecisionEngine, RAQICDenseExecutionConfig


def _rq_value(rq: object, name: str, default: object) -> object:
    return getattr(rq, name, default)


def _actualization_config_from_cfg(cfg: SimulationConfig) -> ActualizationExtensionConfig:
    rq = cfg.raqic
    return ActualizationExtensionConfig(
        variant=str(getattr(rq, "actualization_variant", "stable_baseline")),
        utility_coupling=float(getattr(rq, "utility_coupling", 0.0)),
        utility_projection_epsilon=float(getattr(rq, "utility_projection_epsilon", 1e-8)),
        utility_bound_floor=float(getattr(rq, "utility_bound_floor", 1.0)),
        phase_resonance_coupling=float(getattr(rq, "phase_resonance_coupling", 0.0)),
        interference_mixer_strength=float(getattr(rq, "interference_mixer_strength", 0.0)),
        interference_trotter_steps=int(getattr(rq, "interference_trotter_steps", 1)),
        shadow_only=bool(getattr(rq, "experimental_shadow_only", False)),
    )


def _phase_child_weights_numpy(state: WorldState, cfg: SimulationConfig) -> np.ndarray:
    h, w = field_shape(state)
    size = int(cfg.world.patch_size)
    py, px = h // size, w // size
    eps = float(cfg.actions.epsilon)
    alive = ((state.health > 0.0) & (~state.obstacle)).astype(np.float64)
    boundary = np.clip(state.boundary, 0.0, 1.0)
    coherence = np.clip(getattr(state, "noetic_C", state.integration), 0.0, 1.0)
    resource = np.clip(state.resource / max(float(cfg.resources.max_resource), eps), 0.0, 1.0)
    yy, xx = np.indices((size, size), dtype=np.float64)
    distance = np.sqrt(
        (yy.reshape(-1) - (size - 1) / 2.0) ** 2 + (xx.reshape(-1) - (size - 1) / 2.0) ** 2
    ) / max(float(size), 1.0)

    def patch_cells(value: np.ndarray) -> np.ndarray:
        return value.reshape(py, size, px, size).transpose(0, 2, 1, 3).reshape(py, px, size * size)

    alive_patch = patch_cells(alive)
    raw = (
        patch_cells(boundary)
        * patch_cells(coherence)
        * patch_cells(resource)
        * np.exp(-distance)[None, None, :]
        * alive_patch
    )
    raw_sum = np.sum(raw, axis=-1, keepdims=True)
    alive_count = np.sum(alive_patch, axis=-1, keepdims=True)
    uniform = np.divide(
        alive_patch, alive_count, out=np.zeros_like(alive_patch), where=alive_count > 0.0
    )
    normalized = np.divide(raw, raw_sum, out=np.zeros_like(raw), where=raw_sum > 1e-12)
    patch_weights = np.where(raw_sum > 1e-12, normalized, uniform)
    return patch_weights.reshape(py, px, size, size).transpose(0, 2, 1, 3).reshape(h, w)


def _prepare_action_phase_context_numpy(state: WorldState, cfg: SimulationConfig) -> None:
    if getattr(state, "raqic_parent_action_phase", None) is None:
        return
    arrays = aggregate_action_phase_context(
        np.asarray(state.raqic_probabilities, dtype=np.float64),
        np.asarray(state.raqic_phase, dtype=np.float64),
        _phase_child_weights_numpy(state, cfg),
        patch_confidence=np.asarray(state.raqic_patch_confidence, dtype=np.float64),
        patch_size=int(cfg.world.patch_size),
        patch_weight=float(cfg.raqic.phase_resonance_patch_weight),
        global_weight=float(cfg.raqic.phase_resonance_global_weight),
        support_epsilon=float(cfg.raqic.phase_resonance_support_epsilon),
        rest_index=int(Action.REST),
        xp=np,
        dtype=np.float64,
    )
    names = (
        "raqic_patch_action_phase",
        "raqic_patch_action_coherence",
        "raqic_global_action_phase",
        "raqic_global_action_coherence",
        "raqic_parent_action_phase",
        "raqic_parent_action_coherence",
    )
    for name, value in zip(names, arrays, strict=True):
        target = getattr(state, name, None)
        if isinstance(target, np.ndarray):
            target[...] = np.asarray(value, dtype=target.dtype)


def _dense_reference_config_from_cfg(
    cfg: SimulationConfig,
) -> RAQICDenseExecutionConfig:
    """Build the NumPy dense reference with the production RAQIC settings."""
    rq = cfg.raqic
    return RAQICDenseExecutionConfig(
        seed=int(cfg.world.seed),
        beta_intention=float(rq.beta_intention),
        temperature=float(rq.action_temperature),
        epsilon_adelic=float(rq.epsilon_adelic),
        prime_weights=dict(rq.prime_weights),
        modulus_power=int(getattr(rq, "modulus_power", 8)),
        precision=str(rq.full_gpu_precision),
        backend="numpy",
        strict_gpu=False,
        audit_limit=int(rq.gpu_audit_limit),
        tolerance=float(rq.gpu_probability_tolerance),
        phase_mode=str(rq.full_gpu_phase_mode),
        compute_phase=True,
        host_diagnostics=False,
        actualization=_actualization_config_from_cfg(cfg),
    )


class OWLRAQICGPUEngine:
    """Additive dense GPU/CPU-dense RAQIC engine for OWL.

    In strict gpu_batch mode, CuPy is required. If fallback is explicitly enabled,
    dense NumPy is used and the result metadata/audit counts record fallback.
    """

    def prepare_cross_scale_context(self, state: WorldState, cfg: SimulationConfig) -> np.ndarray:
        ensure_raqic_fields(state, cfg)
        aggregate_raqic_records_to_patches(state, cfg)
        aggregate_raqic_patches_to_global(state, cfg)
        update_raqic_patch_intention(state, cfg)
        update_raqic_global_intention(state, cfg)
        parent = dispatch_raqic_intention_to_cells(state, cfg)
        _prepare_action_phase_context_numpy(state, cfg)
        return parent

    def _execution_config(self, cfg: SimulationConfig, backend: str) -> RAQICDenseExecutionConfig:
        """Build a dense RAQIC execution config aligned with the full GPU stage.

         repair: stage parity is scientifically meaningful only when
        the CPU dense-reference adapter and CuPy device stage use the same seed,
        temperature, p-adic weights, precision, phase mode, and tolerance.
        """
        rq = cfg.raqic
        return RAQICDenseExecutionConfig(
            seed=int(cfg.world.seed),
            beta_intention=float(getattr(rq, "beta_intention", 1.0)),
            temperature=float(getattr(rq, "temperature", getattr(rq, "action_temperature", 1.0))),
            epsilon_adelic=float(getattr(rq, "epsilon_adelic", 1.0)),
            prime_weights=dict(getattr(rq, "prime_weights", {})),
            modulus_power=int(getattr(rq, "modulus_power", 8)),
            precision=str(
                getattr(rq, "full_gpu_precision", getattr(rq, "gpu_precision", "audit64"))
            ),
            backend=str(backend),
            strict_gpu=bool(getattr(rq, "strict_gpu", True) and str(backend) == "cupy"),
            audit_limit=int(getattr(rq, "gpu_audit_limit", 8)),
            tolerance=float(getattr(rq, "gpu_probability_tolerance", 1e-8)),
            phase_mode=str(
                getattr(rq, "full_gpu_phase_mode", getattr(rq, "phase_mode", "scalar_reference"))
            ),
            compute_phase=True,
            host_diagnostics=False,
            actualization=_actualization_config_from_cfg(cfg),
        )

    def decide_cells(
        self,
        state: WorldState,
        cfg: SimulationConfig,
        authority: np.ndarray,
        rng: np.random.Generator,
        utilities: np.ndarray | None = None,
    ) -> OWLRAQICDecisionBatch:
        ensure_raqic_fields(state, cfg)
        reset_raqic_audit_flags(state)
        h, w = field_shape(state)
        actions = len(Action)
        decision_dtype = raqic_numpy_dtype(cfg)
        probs = np.zeros((h, w, actions), dtype=decision_dtype)
        probs[..., int(Action.REST)] = 1.0
        readout = np.full((h, w), int(Action.REST), dtype=DEFAULT_READOUT_DTYPE)
        scores = np.zeros((h, w, actions), dtype=decision_dtype)
        phases = np.zeros((h, w, actions), dtype=decision_dtype)
        rec_readout = np.zeros((h, w), dtype=np.int32)
        confidence = np.zeros((h, w), dtype=decision_dtype)
        trace_error = np.zeros((h, w), dtype=decision_dtype)
        min_eig = np.zeros((h, w), dtype=decision_dtype)
        backend_code = np.zeros((h, w), dtype=np.int32)

        backend = "numpy"
        fallback_count = 0
        errors: list[str] = []
        info = detect_cupy()
        if not info.available:
            if cfg.raqic.strict_gpu and not cfg.raqic.fallback_on_backend_error:
                raise RuntimeError(
                    f"gpu_batch requested but CuPy/CUDA is unavailable: {info.error}"
                )
            backend = "numpy"
            fallback_count = 1
            errors.append(f"GPU unavailable; dense NumPy fallback used: {info.error}")

        batch = (
            build_dense_feature_batch_gpu(
                state,
                cfg,
                authority,
                np.asarray(state.raqic_parent_intention, dtype=np.float64),
                utilities=utilities,
                parent_action_phase=getattr(state, "raqic_parent_action_phase", None),
                parent_action_coherence=getattr(state, "raqic_parent_action_coherence", None),
            )
            if backend == "cupy"
            else build_dense_feature_batch_numpy(
                state,
                cfg,
                authority,
                np.asarray(state.raqic_parent_intention, dtype=np.float64),
                utilities=utilities,
                parent_action_phase=getattr(state, "raqic_parent_action_phase", None),
                parent_action_coherence=getattr(state, "raqic_parent_action_coherence", None),
            )
        )
        engine = RAQICDenseDecisionEngine(self._execution_config(cfg, backend))
        result = engine.decide_batch(batch).to_numpy()
        yx = np.asarray(batch.to_numpy().yx, dtype=np.int32)

        if yx.shape[0]:
            yy = yx[:, 0]
            xx = yx[:, 1]
            probs[yy, xx, :] = np.asarray(result.probabilities, dtype=decision_dtype)
            readout[yy, xx] = np.asarray(result.readout, dtype=DEFAULT_READOUT_DTYPE)
            scores[yy, xx, :] = np.asarray(result.scores, dtype=decision_dtype)
            phases[yy, xx, :] = np.asarray(result.phases, dtype=decision_dtype)
            rec_readout[yy, xx] = np.argmax(probs[yy, xx, :], axis=1).astype(np.int32)
            confidence[yy, xx] = np.asarray(result.confidence, dtype=decision_dtype)
            trace_error[yy, xx] = np.asarray(result.trace_error, dtype=decision_dtype)
            min_eig[yy, xx] = np.asarray(result.min_eigenvalue, dtype=decision_dtype)
            backend_code[yy, xx] = np.asarray(result.backend_code, dtype=np.int32)
            diagnostic_map = {
                "raqic_pre_mixer_probabilities": result.pre_mixer_probabilities,
                "raqic_utility_innovation": result.utility_innovation,
                "raqic_phase_alignment": result.phase_alignment,
                "raqic_resonant_parent_intention": result.resonant_parent_intention,
                "raqic_interference_delta_l1": result.interference_delta_l1,
                "raqic_policy_kl": result.policy_kl,
                "raqic_utility_projection_fraction": result.utility_projection_fraction,
                "raqic_utility_score_cosine": result.utility_score_cosine,
                "raqic_utility_orthogonality_residual": result.utility_orthogonality_residual,
                "raqic_utility_innovation_norm": result.utility_innovation_norm,
                "raqic_interference_norm_error": result.interference_norm_error,
                "raqic_interference_illegal_mass": result.interference_illegal_mass,
                "raqic_shadow_probabilities": result.shadow_probabilities,
                "raqic_shadow_readout": result.shadow_readout,
            }
            for field_name, row_values in diagnostic_map.items():
                target = getattr(state, field_name, None)
                if isinstance(target, np.ndarray) and row_values is not None:
                    target[yy, xx, ...] = np.asarray(row_values, dtype=target.dtype)

        dead = (state.health <= 0.0) | state.obstacle
        if np.any(dead):
            probs[dead, :] = 0.0
            probs[dead, int(Action.REST)] = 1.0
            readout[dead] = int(Action.REST)

        sums = np.sum(probs, axis=-1, keepdims=True, dtype=decision_dtype)
        probs = np.divide(probs, sums, out=np.zeros_like(probs), where=sums > cfg.actions.epsilon)
        bad = sums[..., 0] <= cfg.actions.epsilon
        if np.any(bad):
            probs[bad, :] = 0.0
            probs[bad, int(Action.REST)] = 1.0
            readout[bad] = int(Action.REST)

        eligible = int(np.sum((state.health > 0.0) & (~state.obstacle)))
        processed_value = batch.metadata.get("processed_cells", yx.shape[0])
        if processed_value is None:
            processed_value = yx.shape[0]
        processed = int(processed_value)
        if cfg.raqic.gpu_all_cells_required and processed != eligible:
            raise RuntimeError(
                f"gpu all-cell condition failed: processed={processed}, eligible={eligible}"
            )

        audit = {
            "mode": cfg.raqic.mode,
            "backend": backend,
            "gpu_info": info.to_dict(),
            "processed_cells": processed,
            "eligible_cells": eligible,
            "all_cells_satisfied": processed == eligible,
            "fallback_count": fallback_count,
            "backend_errors": errors[:10],
            "dense_result_metadata": result.metadata,
        }
        return OWLRAQICDecisionBatch(
            probabilities=probs,
            readout=readout.astype(DEFAULT_READOUT_DTYPE),
            records={
                "action": readout,
                "readout": rec_readout,
                "confidence": confidence,
                "trace_error": trace_error,
                "min_eigenvalue": min_eig,
                "backend_code": backend_code,
            },
            scores=scores,
            phases=phases,
            audit=audit,
        )
