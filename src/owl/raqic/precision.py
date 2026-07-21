"""Shared numerical precision policy for RAQIC scientific state.

The physical OWL simulation remains in its established float32 contract.  RAQIC
keeps a smaller set of authoritative recursive and audit fields at the declared
decision precision so the NumPy scientific reference and CuPy implementation do
not silently execute different recursions.
"""

from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np

PrecisionMode = Literal["audit64", "mixed", "balanced32", "fast32"]

_VALID_PRECISION_MODES: frozenset[str] = frozenset({"audit64", "mixed", "balanced32", "fast32"})

# These fields are promoted by the device precision policy and must therefore
# be allocated at the same precision by the NumPy scientific reference.
RAQIC_AUDIT_REAL_FIELDS: frozenset[str] = frozenset(
    {
        "raqic_probabilities",
        "raqic_score",
        "raqic_phase",
        "raqic_record_confidence",
        "raqic_trace_error",
        "raqic_min_eigenvalue",
        "raqic_compare_l1",
        "raqic_compare_kl",
        "raqic_debug_density_diag",
        "raqic_legacy_shadow_possibility",
        "raqic_patch_action_phase",
        "raqic_patch_action_coherence",
        "raqic_global_action_phase",
        "raqic_global_action_coherence",
        "raqic_parent_action_phase",
        "raqic_parent_action_coherence",
        "raqic_pre_mixer_probabilities",
        "raqic_utility_innovation",
        "raqic_phase_alignment",
        "raqic_resonant_parent_intention",
        "raqic_interference_delta_l1",
        "raqic_policy_kl",
        "raqic_utility_projection_fraction",
        "raqic_utility_score_cosine",
        "raqic_utility_orthogonality_residual",
        "raqic_utility_innovation_norm",
        "raqic_interference_norm_error",
        "raqic_interference_illegal_mass",
        "raqic_shadow_probabilities",
    }
)

# The fields below cross tick boundaries and directly influence the next RAQIC
# decision. They are called out separately for tests and certificate evidence.
RAQIC_RECURSIVE_REAL_FIELDS: frozenset[str] = frozenset(
    {
        "raqic_probabilities",
        "raqic_score",
        "raqic_phase",
        "raqic_patch_action_phase",
        "raqic_patch_action_coherence",
        "raqic_global_action_phase",
        "raqic_global_action_coherence",
        "raqic_parent_action_phase",
        "raqic_parent_action_coherence",
    }
)


def precision_mode(config_or_raqic: Any) -> PrecisionMode:
    """Return the validated RAQIC precision mode from a config or RAQIC object."""
    raqic = getattr(config_or_raqic, "raqic", config_or_raqic)
    value = str(
        getattr(
            raqic,
            "full_gpu_precision",
            getattr(raqic, "gpu_precision", "audit64"),
        )
    )
    if value not in _VALID_PRECISION_MODES:
        raise ValueError(f"unsupported RAQIC precision mode: {value!r}")
    return cast(PrecisionMode, value)


def uses_float64_audit(config_or_raqic: Any) -> bool:
    """Return whether RAQIC audit/recursive fields use float64."""
    return precision_mode(config_or_raqic) in {"audit64", "mixed"}


def raqic_numpy_dtype(config_or_raqic: Any) -> np.dtype[Any]:
    """Return the NumPy dtype for RAQIC audit and recursive real fields."""
    return np.dtype(np.float64 if uses_float64_audit(config_or_raqic) else np.float32)


def raqic_backend_real_dtype(config_or_raqic: Any, xp: Any) -> Any:
    """Return the backend real dtype for RAQIC audit and recursive fields."""
    return xp.float64 if uses_float64_audit(config_or_raqic) else xp.float32


def raqic_backend_complex_dtype(config_or_raqic: Any, xp: Any) -> Any:
    """Return the backend complex dtype paired with the RAQIC real dtype."""
    return xp.complex128 if uses_float64_audit(config_or_raqic) else xp.complex64
