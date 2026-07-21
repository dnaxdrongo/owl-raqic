from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from owl.core.actions import Action
from owl.core.config import SimulationConfig
from owl.core.constants import DEFAULT_READOUT_DTYPE
from owl.core.state import WorldState, field_shape
from owl.raqic.action_adapter import map_raqic_probs_to_owl_probs
from owl.raqic.aggregation import (
    aggregate_raqic_patches_to_global,
    aggregate_raqic_records_to_patches,
    update_raqic_global_intention,
    update_raqic_patch_intention,
)
from owl.raqic.config import build_raqic_action_set, convert_owl_cfg_to_raqic_cfg
from owl.raqic.feature_extraction import build_feature_packets
from owl.raqic.precision import raqic_numpy_dtype
from owl.raqic.state import (
    RAQIC_FLAG_BACKEND_ERROR,
    RAQIC_FLAG_FALLBACK_USED,
    ensure_raqic_fields,
    reset_raqic_audit_flags,
)
from owl.raqic.topdown import dispatch_raqic_intention_to_cells
from owl_raqic.algorithms.raqic_driver import RAQICDecisionEngine


@dataclass
class OWLRAQICDecisionBatch:
    probabilities: np.ndarray
    readout: np.ndarray
    records: dict[str, np.ndarray]
    scores: np.ndarray
    phases: np.ndarray
    audit: dict[str, Any]


class OWLRAQICEngine:
    def __init__(self, cfg: SimulationConfig):
        self.action_set = build_raqic_action_set()
        self.raqic_config = convert_owl_cfg_to_raqic_cfg(cfg)
        self.decision_engine = RAQICDecisionEngine(self.raqic_config, self.action_set)

    def prepare_cross_scale_context(self, state: WorldState, cfg: SimulationConfig) -> np.ndarray:
        ensure_raqic_fields(state, cfg)
        aggregate_raqic_records_to_patches(state, cfg)
        aggregate_raqic_patches_to_global(state, cfg)
        update_raqic_patch_intention(state, cfg)
        update_raqic_global_intention(state, cfg)
        return dispatch_raqic_intention_to_cells(state, cfg)

    def decide_cells(
        self,
        state: WorldState,
        cfg: SimulationConfig,
        authority: np.ndarray,
        rng: np.random.Generator,
        utilities: np.ndarray | None = None,
    ) -> OWLRAQICDecisionBatch:
        if rng is None:
            raise ValueError("rng must be an explicit np.random.Generator")
        ensure_raqic_fields(state, cfg)
        assert state.raqic_audit_flags is not None
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
        fallback_count = 0
        errors = []
        packets = build_feature_packets(
            state, cfg, authority, np.asarray(state.raqic_parent_intention, dtype=decision_dtype)
        )
        for packet in packets:
            y = int(packet.metadata["y"])
            x = int(packet.metadata["x"])
            try:
                result = self.decision_engine.decide(packet, sample=True)
                p = map_raqic_probs_to_owl_probs(result.action_probabilities, self.action_set)
                probs[y, x, :] = p
                sampled = (
                    int(result.sampled_action)
                    if result.sampled_action is not None
                    else int(np.argmax(p))
                )
                readout[y, x] = sampled if 0 <= sampled < actions else int(Action.REST)
                sc = np.asarray(
                    result.measurement_record.get("scores", np.zeros(actions)), dtype=decision_dtype
                )
                ph = np.asarray(
                    result.measurement_record.get("phases", np.zeros(actions)), dtype=decision_dtype
                )
                scores[y, x, : min(actions, sc.size)] = sc[:actions]
                phases[y, x, : min(actions, ph.size)] = ph[:actions]
                traces = np.asarray(
                    result.measurement_record.get("recursive_traces", [1.0]), dtype=float
                )
                trace_error[y, x] = (
                    np.asarray(np.max(np.abs(traces - 1.0)), dtype=decision_dtype)
                    if traces.size
                    else 0.0
                )
                fd = result.recovery_checks.get("final_density", {})
                min_eig[y, x] = np.asarray(
                    fd.get("min_eigenvalue", 0.0) if isinstance(fd, dict) else 0.0,
                    dtype=decision_dtype,
                )
                rec_readout[y, x] = int(np.argmax(p))
                confidence[y, x] = np.asarray(np.max(p), dtype=decision_dtype)
                backend_code[y, x] = 1
            except Exception as exc:
                if not cfg.raqic.fallback_on_backend_error:
                    raise
                fallback_count += 1
                errors.append(f"{type(exc).__name__}: {exc}")
                state.raqic_audit_flags[y, x, RAQIC_FLAG_FALLBACK_USED] = 1
                state.raqic_audit_flags[y, x, RAQIC_FLAG_BACKEND_ERROR] = 1
                probs[y, x, :] = 0
                probs[y, x, int(Action.REST)] = 1
                readout[y, x] = int(Action.REST)
        dead = (state.health <= 0.0) | state.obstacle
        if np.any(dead):
            probs[dead, :] = 0
            probs[dead, int(Action.REST)] = 1
            readout[dead] = int(Action.REST)
        sums = np.sum(probs, axis=-1, keepdims=True, dtype=decision_dtype)
        probs = np.divide(probs, sums, out=np.zeros_like(probs), where=sums > cfg.actions.epsilon)
        bad = sums[..., 0] <= cfg.actions.epsilon
        if np.any(bad):
            probs[bad, :] = 0
            probs[bad, int(Action.REST)] = 1
            readout[bad] = int(Action.REST)
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
            audit={
                "packets": len(packets),
                "fallback_count": fallback_count,
                "backend_errors": errors[:10],
                "mode": cfg.raqic.mode,
                "decision_policy": cfg.raqic.decision_policy,
            },
        )


def apply_raqic_decisions(
    state: WorldState,
    cfg: SimulationConfig,
    authority: np.ndarray,
    rng: np.random.Generator,
    utilities: np.ndarray | None = None,
) -> OWLRAQICDecisionBatch:
    from owl.raqic.state import ensure_raqic_fields

    ensure_raqic_fields(state, cfg)
    assert state.raqic_audit_flags is not None
    assert state.raqic_probabilities is not None
    assert state.raqic_readout is not None
    assert state.raqic_record_action is not None
    assert state.raqic_record_readout is not None
    assert state.raqic_record_confidence is not None
    assert state.raqic_score is not None
    assert state.raqic_phase is not None
    assert state.raqic_trace_error is not None
    assert state.raqic_min_eigenvalue is not None
    assert state.raqic_backend_code is not None
    engine: Any
    mode = str(getattr(cfg.raqic, "mode", ""))

    # CPU/GPU parity must compare the same dense RAQIC decision family. Hybrid
    # audit modes therefore use the dense reference rather than the scalar
    # packet engine.
    dense_reference_modes = {
        "gpu_batch",
        "gpu_hybrid_audit",
        "gpu_full",
        "gpu_full_hybrid_audit",
        "gpu_full_production",
    }
    experimental_variant = (
        str(getattr(cfg.raqic, "actualization_variant", "stable_baseline")) != "stable_baseline"
    )
    diagnostic_reference = bool(getattr(cfg.raqic, "record_actualization_diagnostics", False))
    if (
        mode in dense_reference_modes
        or mode.startswith("gpu_full")
        or experimental_variant
        or diagnostic_reference
    ):
        from owl.raqic.gpu_engine import OWLRAQICGPUEngine

        engine = OWLRAQICGPUEngine()
    else:
        engine = OWLRAQICEngine(cfg)
    engine.prepare_cross_scale_context(state, cfg)
    result = cast(
        OWLRAQICDecisionBatch,
        engine.decide_cells(state, cfg, authority, rng, utilities=utilities),
    )
    # Preserve the configured precision for authoritative RAQIC recursion.
    # The physical possibility tensor intentionally remains in the established
    # OWL physical dtype and receives an explicit projection/cast.
    state.possibility[...] = np.asarray(result.probabilities, dtype=state.possibility.dtype)
    state.readout[...] = np.asarray(result.readout, dtype=state.readout.dtype)
    state.raqic_probabilities[...] = np.asarray(
        result.probabilities, dtype=state.raqic_probabilities.dtype
    )
    state.raqic_readout[...] = np.asarray(result.readout, dtype=state.raqic_readout.dtype)
    state.raqic_record_action[...] = np.asarray(
        result.records["action"], dtype=state.raqic_record_action.dtype
    )
    state.raqic_record_readout[...] = np.asarray(
        result.records["readout"], dtype=state.raqic_record_readout.dtype
    )
    state.raqic_record_confidence[...] = np.asarray(
        result.records["confidence"], dtype=state.raqic_record_confidence.dtype
    )
    state.raqic_score[...] = np.asarray(result.scores, dtype=state.raqic_score.dtype)
    state.raqic_phase[...] = np.asarray(result.phases, dtype=state.raqic_phase.dtype)
    state.raqic_trace_error[...] = np.asarray(
        result.records["trace_error"], dtype=state.raqic_trace_error.dtype
    )
    state.raqic_min_eigenvalue[...] = np.asarray(
        result.records["min_eigenvalue"], dtype=state.raqic_min_eigenvalue.dtype
    )
    state.raqic_backend_code[...] = np.asarray(
        result.records["backend_code"], dtype=state.raqic_backend_code.dtype
    )

    # The independent ``cpu_audit`` engine uses its own diagnostic contract.
    # arrays. When the stable baseline is explicitly recorded, populate the
    # extension evidence from the authoritative baseline outputs rather than
    # leaving the allocation-time REST sentinels in living cells. This does
    # not alter action probabilities or readout and gives the scalar CPU
    # reference the same semantic evidence as the dense NumPy/CuPy paths.
    if str(
        getattr(cfg.raqic, "actualization_variant", "stable_baseline")
    ) == "stable_baseline" and bool(getattr(cfg.raqic, "record_actualization_diagnostics", False)):
        pre_mixer = getattr(state, "raqic_pre_mixer_probabilities", None)
        if isinstance(pre_mixer, np.ndarray):
            pre_mixer[...] = np.asarray(result.probabilities, dtype=pre_mixer.dtype)
        resonant_parent = getattr(state, "raqic_resonant_parent_intention", None)
        if isinstance(resonant_parent, np.ndarray):
            parent = np.clip(np.asarray(state.raqic_parent_intention, dtype=np.float64), 0.0, None)
            sums = np.asarray(
                np.sum(parent, axis=-1, keepdims=True, dtype=np.float64),
                dtype=np.float64,
            )
            normalized = np.divide(
                parent,
                sums,
                out=np.zeros_like(parent),
                where=sums > float(cfg.actions.epsilon),
            )
            bad = sums[..., 0] <= float(cfg.actions.epsilon)
            if np.any(bad):
                normalized[bad, :] = 0.0
                normalized[bad, int(Action.REST)] = 1.0
            resonant_parent[...] = normalized.astype(resonant_parent.dtype, copy=False)
    return result
