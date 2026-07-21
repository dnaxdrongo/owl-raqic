from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from owl_raqic.gpu.actualization_extensions import ActualizationExtensionConfig
from owl_raqic.gpu.backend import detect_cupy, require_cupy
from owl_raqic.gpu.dense_types import RAQICDenseBatch, RAQICDenseResult
from owl_raqic.gpu.instrument_checks import analytic_probability_checks, explicit_instrument_audit
from owl_raqic.gpu.math_gpu import decide_dense, to_numpy
from owl_raqic.gpu.profiler import GPUProfile
from owl_raqic.math.action_graph import action_graph_hash


@dataclass(frozen=True)
class RAQICDenseExecutionConfig:
    seed: int = 1234
    beta_intention: float = 1.0
    temperature: float = 1.0
    epsilon_adelic: float = 1.0
    prime_weights: dict[int, float] | None = None
    modulus_power: int = 8
    precision: str = "audit64"
    backend: str = "numpy"
    strict_gpu: bool = True
    audit_limit: int = 8
    tolerance: float = 1e-10
    phase_mode: str = "scalar_reference"
    compute_phase: bool = True
    host_diagnostics: bool = True
    actualization: ActualizationExtensionConfig | None = None


class RAQICDenseDecisionEngine:
    """Backend-neutral dense RAQIC engine.

    On GPU, production mode keeps probabilities and phases on device.  Explicit
    Kraus/instrument audits copy only the configured sample, never the full grid.
    """

    def __init__(self, config: RAQICDenseExecutionConfig) -> None:
        self.config = config

    def _xp(self) -> Any:
        return require_cupy() if self.config.backend == "cupy" else np

    def _to_backend(self, batch: RAQICDenseBatch, xp: Any) -> Any:
        if self.config.backend != "cupy":
            return batch.to_numpy()
        return RAQICDenseBatch(
            ow_id=xp.asarray(batch.ow_id),
            yx=xp.asarray(batch.yx),
            features=xp.asarray(batch.features),
            feature_bins=xp.asarray(batch.feature_bins),
            adelic_codes=xp.asarray(batch.adelic_codes),
            authority_mask=xp.asarray(batch.authority_mask),
            parent_intention=xp.asarray(batch.parent_intention),
            alive_mask=xp.asarray(batch.alive_mask),
            scale_id=xp.asarray(batch.scale_id),
            tick=batch.tick,
            feature_names=batch.feature_names,
            action_names=batch.action_names,
            active_primes=batch.active_primes,
            action_utilities=(
                None if batch.action_utilities is None else xp.asarray(batch.action_utilities)
            ),
            parent_action_phase=(
                None if batch.parent_action_phase is None else xp.asarray(batch.parent_action_phase)
            ),
            parent_action_coherence=(
                None
                if batch.parent_action_coherence is None
                else xp.asarray(batch.parent_action_coherence)
            ),
            interference_amplitude_output=(
                None
                if batch.interference_amplitude_output is None
                else xp.asarray(batch.interference_amplitude_output)
            ),
            interference_left_scratch=(
                None
                if batch.interference_left_scratch is None
                else xp.asarray(batch.interference_left_scratch)
            ),
            interference_right_scratch=(
                None
                if batch.interference_right_scratch is None
                else xp.asarray(batch.interference_right_scratch)
            ),
            metadata=dict(batch.metadata),
        )

    def decide_batch(self, batch: RAQICDenseBatch) -> RAQICDenseResult:
        profile = GPUProfile()
        xp = self._xp()
        backend_code = 20 if self.config.backend == "cupy" else 10
        if self.config.backend == "cupy":
            info = detect_cupy()
            if not info.available and self.config.strict_gpu:
                raise RuntimeError(f"strict GPU requested but CuPy is unavailable: {info.error}")

        with profile.stage("to_backend"):
            batch_backend = self._to_backend(batch, xp)

        with profile.stage("dense_decision"):
            scores, phases, probs, readout, confidence, extension_evidence = decide_dense(
                batch_backend,
                seed=self.config.seed,
                beta_intention=self.config.beta_intention,
                temperature=self.config.temperature,
                epsilon_adelic=self.config.epsilon_adelic,
                prime_weights=self.config.prime_weights
                or {p: 1.0 / max(1, len(batch.active_primes)) for p in batch.active_primes},
                modulus_power=self.config.modulus_power,
                precision=self.config.precision,
                xp=xp,
                phase_mode=self.config.phase_mode,
                compute_phase=self.config.compute_phase,
                actualization_config=self.config.actualization,
                return_extension_evidence=True,
            )

        n = int(probs.shape[0])
        checks: dict[str, Any] = {"deferred": not self.config.host_diagnostics}
        if self.config.host_diagnostics:
            with profile.stage("sampled_host_diagnostics"):
                limit = (
                    n
                    if self.config.backend == "numpy"
                    else min(n, max(0, int(self.config.audit_limit)))
                )
                # Probability checks need only rows in the explicit audit sample
                # on GPU. The device metric slab separately checks all row sums.
                if self.config.backend == "numpy":
                    probs_np = np.asarray(probs)
                    phases_np = np.asarray(phases)
                    mask_np = np.asarray(batch.authority_mask, dtype=bool)
                else:
                    probs_np = to_numpy(probs[:limit])
                    phases_np = to_numpy(phases[:limit])
                    mask_np = to_numpy(batch_backend.authority_mask[:limit]).astype(bool)
                checks = analytic_probability_checks(probs_np, mask_np, tol=self.config.tolerance)
                if limit > 0 and self.config.audit_limit > 0:
                    checks["explicit_instrument"] = explicit_instrument_audit(
                        probs_np, phases_np, limit=limit, tol=self.config.tolerance
                    )
                actualization_checks: dict[str, Any] = {
                    "finite_scores": bool(np.all(np.isfinite(to_numpy(scores[:limit])))),
                    "finite_phases": bool(np.all(np.isfinite(phases_np))),
                    "finite_probabilities": bool(np.all(np.isfinite(probs_np))),
                    "max_interference_norm_error": 0.0,
                    "max_interference_illegal_mass": 0.0,
                    "passed": True,
                }
                norm_error = extension_evidence.get("interference_norm_error")
                illegal_mass = extension_evidence.get("interference_illegal_mass")
                if norm_error is not None and limit:
                    actualization_checks["max_interference_norm_error"] = float(
                        np.max(np.abs(to_numpy(norm_error[:limit])))
                    )
                if illegal_mass is not None and limit:
                    actualization_checks["max_interference_illegal_mass"] = float(
                        np.max(np.abs(to_numpy(illegal_mass[:limit])))
                    )
                actualization_checks["passed"] = bool(
                    actualization_checks["finite_scores"]
                    and actualization_checks["finite_phases"]
                    and actualization_checks["finite_probabilities"]
                    and actualization_checks["max_interference_norm_error"] <= self.config.tolerance
                    and actualization_checks["max_interference_illegal_mass"]
                    <= self.config.tolerance
                )
                checks["actualization"] = actualization_checks
                checks["passed"] = bool(
                    checks.get("passed", False) and actualization_checks["passed"]
                )
                checks["audited_rows"] = int(limit)

        dtype = xp.float64 if self.config.precision not in ("balanced32", "fast32") else xp.float32
        return RAQICDenseResult(
            probabilities=probs,
            readout=readout,
            scores=scores,
            phases=phases,
            confidence=confidence,
            trace_error=xp.zeros((n,), dtype=dtype),
            min_eigenvalue=xp.zeros((n,), dtype=dtype),
            backend_code=xp.full((n,), backend_code, dtype=xp.int32),
            audit_flags=xp.zeros((n, 8), dtype=xp.int32),
            pre_mixer_probabilities=extension_evidence["pre_mixer_probabilities"],
            utility_innovation=extension_evidence["utility_innovation"],
            phase_alignment=extension_evidence["phase_alignment"],
            resonant_parent_intention=extension_evidence["resonant_parent_intention"],
            interference_delta_l1=extension_evidence["interference_delta_l1"],
            policy_kl=extension_evidence["policy_kl"],
            utility_projection_fraction=extension_evidence["utility_projection_fraction"],
            utility_score_cosine=extension_evidence["utility_score_cosine"],
            utility_orthogonality_residual=extension_evidence["utility_orthogonality_residual"],
            utility_innovation_norm=extension_evidence["utility_innovation_norm"],
            interference_norm_error=extension_evidence["interference_norm_error"],
            interference_illegal_mass=extension_evidence["interference_illegal_mass"],
            shadow_probabilities=extension_evidence["shadow_probabilities"],
            shadow_readout=extension_evidence["shadow_readout"],
            metadata={
                "backend": self.config.backend,
                "precision": self.config.precision,
                "phase_mode": self.config.phase_mode,
                "compute_phase": self.config.compute_phase,
                "checks": checks,
                "profile": profile.to_dict(),
                "processed_cells": n,
                "actualization_variant": extension_evidence["variant"],
                "actualization_extension_enabled": extension_evidence["extension_enabled"],
                "actualization_shadow_only": extension_evidence["shadow_only"],
                "action_graph_hash": (
                    action_graph_hash(tuple(batch.action_names))
                    if bool(extension_evidence["extension_enabled"])
                    else None
                ),
            },
        )
