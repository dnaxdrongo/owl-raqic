from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np


def _to_numpy(value: Any) -> Any:
    if value is None:
        return None
    try:
        import cupy as cp

        if isinstance(value, cp.ndarray):
            return cp.asnumpy(value)
    except Exception:
        pass
    return np.asarray(value)


@dataclass(frozen=True)
class RAQICDenseBatch:
    """Dense all-cell RAQIC input for a single tick.

    N = eligible units, F = feature count, A = action count,
    P = active finite-prime count.
    """

    ow_id: Any
    yx: Any
    features: Any
    feature_bins: Any
    adelic_codes: Any
    authority_mask: Any
    parent_intention: Any
    alive_mask: Any
    scale_id: Any
    tick: int
    feature_names: tuple[str, ...]
    action_names: tuple[str, ...]
    active_primes: tuple[int, ...]
    action_utilities: Any | None = None
    parent_action_phase: Any | None = None
    parent_action_coherence: Any | None = None
    interference_amplitude_output: Any | None = None
    interference_left_scratch: Any | None = None
    interference_right_scratch: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(getattr(self.features, "shape", (0,))[0])

    @property
    def n_actions(self) -> int:
        return int(getattr(self.authority_mask, "shape", (0, 0))[1])

    def to_numpy(self) -> RAQICDenseBatch:
        return replace(
            self,
            ow_id=_to_numpy(self.ow_id).astype(np.int64, copy=False),
            yx=_to_numpy(self.yx).astype(np.int32, copy=False),
            features=_to_numpy(self.features),
            feature_bins=_to_numpy(self.feature_bins).astype(np.int32, copy=False),
            adelic_codes=_to_numpy(self.adelic_codes).astype(np.int32, copy=False),
            authority_mask=_to_numpy(self.authority_mask).astype(bool, copy=False),
            parent_intention=_to_numpy(self.parent_intention),
            alive_mask=_to_numpy(self.alive_mask).astype(bool, copy=False),
            scale_id=_to_numpy(self.scale_id).astype(np.int32, copy=False),
            action_utilities=_to_numpy(self.action_utilities),
            parent_action_phase=_to_numpy(self.parent_action_phase),
            parent_action_coherence=_to_numpy(self.parent_action_coherence),
            interference_amplitude_output=None,
            interference_left_scratch=None,
            interference_right_scratch=None,
        )


@dataclass(frozen=True)
class RAQICDenseResult:
    probabilities: Any
    readout: Any
    scores: Any
    phases: Any
    confidence: Any
    trace_error: Any
    min_eigenvalue: Any
    backend_code: Any
    audit_flags: Any
    pre_mixer_probabilities: Any | None = None
    utility_innovation: Any | None = None
    phase_alignment: Any | None = None
    resonant_parent_intention: Any | None = None
    interference_delta_l1: Any | None = None
    policy_kl: Any | None = None
    utility_projection_fraction: Any | None = None
    utility_score_cosine: Any | None = None
    utility_orthogonality_residual: Any | None = None
    utility_innovation_norm: Any | None = None
    interference_norm_error: Any | None = None
    interference_illegal_mass: Any | None = None
    shadow_probabilities: Any | None = None
    shadow_readout: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_numpy(self) -> RAQICDenseResult:
        return replace(
            self,
            probabilities=_to_numpy(self.probabilities),
            readout=_to_numpy(self.readout).astype(np.int32, copy=False),
            scores=_to_numpy(self.scores),
            phases=_to_numpy(self.phases),
            confidence=_to_numpy(self.confidence),
            trace_error=_to_numpy(self.trace_error),
            min_eigenvalue=_to_numpy(self.min_eigenvalue),
            backend_code=_to_numpy(self.backend_code).astype(np.int32, copy=False),
            audit_flags=_to_numpy(self.audit_flags).astype(np.int32, copy=False),
            pre_mixer_probabilities=_to_numpy(self.pre_mixer_probabilities),
            utility_innovation=_to_numpy(self.utility_innovation),
            phase_alignment=_to_numpy(self.phase_alignment),
            resonant_parent_intention=_to_numpy(self.resonant_parent_intention),
            interference_delta_l1=_to_numpy(self.interference_delta_l1),
            policy_kl=_to_numpy(self.policy_kl),
            utility_projection_fraction=_to_numpy(self.utility_projection_fraction),
            utility_score_cosine=_to_numpy(self.utility_score_cosine),
            utility_orthogonality_residual=_to_numpy(self.utility_orthogonality_residual),
            utility_innovation_norm=_to_numpy(self.utility_innovation_norm),
            interference_norm_error=_to_numpy(self.interference_norm_error),
            interference_illegal_mass=_to_numpy(self.interference_illegal_mass),
            shadow_probabilities=_to_numpy(self.shadow_probabilities),
            shadow_readout=(
                None
                if self.shadow_readout is None
                else _to_numpy(self.shadow_readout).astype(np.int32, copy=False)
            ),
        )
