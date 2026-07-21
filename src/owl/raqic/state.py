from __future__ import annotations

from typing import Any

import numpy as np

from owl.core.actions import Action
from owl.core.config import SimulationConfig
from owl.core.constants import DEFAULT_FLOAT_DTYPE, DEFAULT_INT_DTYPE, DEFAULT_READOUT_DTYPE
from owl.core.state import WorldState, field_shape
from owl.engine.aggregation import _validate_patch_size, _validate_patchable_shape
from owl.raqic.precision import raqic_numpy_dtype

RAQIC_FLAG_FALLBACK_USED = 0
RAQIC_FLAG_BACKEND_ERROR = 1
RAQIC_FLAG_QISKIT_CHECKED = 2
RAQIC_FLAG_RECOVERY_FAILED = 3
RAQIC_FLAG_COUNT = 8


def _rest_probs(
    shape: tuple[int, int], actions: int, *, dtype: Any = DEFAULT_FLOAT_DTYPE
) -> np.ndarray:
    out = np.zeros((*shape, actions), dtype=dtype)
    out[..., int(Action.REST)] = 1.0
    return out


def ensure_raqic_fields(state: WorldState, cfg: SimulationConfig) -> None:
    if not getattr(cfg.raqic, "enabled", False):
        return
    h, w = field_shape(state)
    actions = len(Action)
    patch = _validate_patch_size(cfg.world.patch_size)
    ph, pw = _validate_patchable_shape((h, w), patch)
    audit_dtype = raqic_numpy_dtype(cfg)
    specs = {
        "raqic_probabilities": _rest_probs((h, w), actions, dtype=audit_dtype),
        "raqic_readout": np.full((h, w), int(Action.REST), dtype=DEFAULT_READOUT_DTYPE),
        "raqic_record_action": np.full((h, w), int(Action.REST), dtype=DEFAULT_READOUT_DTYPE),
        "raqic_record_readout": np.zeros((h, w), dtype=DEFAULT_INT_DTYPE),
        "raqic_record_confidence": np.zeros((h, w), dtype=audit_dtype),
        "raqic_score": np.zeros((h, w, actions), dtype=audit_dtype),
        "raqic_phase": np.zeros((h, w, actions), dtype=audit_dtype),
        "raqic_parent_intention": _rest_probs((h, w), actions),
        "raqic_audit_flags": np.zeros((h, w, RAQIC_FLAG_COUNT), dtype=DEFAULT_INT_DTYPE),
        "raqic_trace_error": np.zeros((h, w), dtype=audit_dtype),
        "raqic_min_eigenvalue": np.zeros((h, w), dtype=audit_dtype),
        "raqic_backend_code": np.zeros((h, w), dtype=DEFAULT_INT_DTYPE),
        "raqic_legacy_shadow_possibility": _rest_probs((h, w), actions, dtype=audit_dtype),
        "raqic_legacy_shadow_readout": np.full(
            (h, w), int(Action.REST), dtype=DEFAULT_READOUT_DTYPE
        ),
        "raqic_compare_l1": np.zeros((h, w), dtype=audit_dtype),
        "raqic_compare_kl": np.zeros((h, w), dtype=audit_dtype),
        "raqic_debug_density_diag": _rest_probs((h, w), actions, dtype=audit_dtype),
        "raqic_patch_intention": _rest_probs((ph, pw), actions),
        "raqic_patch_record_aggregate": _rest_probs((ph, pw), actions),
        "raqic_patch_confidence": np.zeros((ph, pw), dtype=DEFAULT_FLOAT_DTYPE),
        "raqic_global_intention": np.eye(actions, dtype=DEFAULT_FLOAT_DTYPE)[int(Action.REST)],
        "raqic_global_record_aggregate": np.eye(actions, dtype=DEFAULT_FLOAT_DTYPE)[
            int(Action.REST)
        ],
    }
    extension_enabled = (
        str(getattr(cfg.raqic, "actualization_variant", "stable_baseline")) != "stable_baseline"
        or bool(getattr(cfg.raqic, "experimental_shadow_only", False))
        or bool(getattr(cfg.raqic, "record_actualization_diagnostics", False))
    )
    needs_phase_context = (
        float(getattr(cfg.raqic, "phase_resonance_coupling", 0.0)) != 0.0
        or float(getattr(cfg.raqic, "interference_mixer_strength", 0.0)) != 0.0
        or bool(getattr(cfg.raqic, "record_actualization_diagnostics", False))
    )
    if needs_phase_context:
        specs.update(
            {
                "raqic_patch_action_phase": np.zeros((ph, pw, actions), dtype=audit_dtype),
                "raqic_patch_action_coherence": np.zeros((ph, pw, actions), dtype=audit_dtype),
                "raqic_global_action_phase": np.zeros((actions,), dtype=audit_dtype),
                "raqic_global_action_coherence": np.zeros((actions,), dtype=audit_dtype),
                "raqic_parent_action_phase": np.zeros((h, w, actions), dtype=audit_dtype),
                "raqic_parent_action_coherence": np.zeros((h, w, actions), dtype=audit_dtype),
            }
        )
    if extension_enabled:
        specs.update(
            {
                "raqic_pre_mixer_probabilities": _rest_probs((h, w), actions, dtype=audit_dtype),
                "raqic_utility_innovation": np.zeros((h, w, actions), dtype=audit_dtype),
                "raqic_phase_alignment": np.zeros((h, w, actions), dtype=audit_dtype),
                "raqic_resonant_parent_intention": _rest_probs((h, w), actions, dtype=audit_dtype),
                "raqic_interference_delta_l1": np.zeros((h, w), dtype=audit_dtype),
                "raqic_policy_kl": np.zeros((h, w), dtype=audit_dtype),
                "raqic_utility_projection_fraction": np.zeros((h, w), dtype=audit_dtype),
                "raqic_utility_score_cosine": np.zeros((h, w), dtype=audit_dtype),
                "raqic_utility_orthogonality_residual": np.zeros((h, w), dtype=audit_dtype),
                "raqic_utility_innovation_norm": np.zeros((h, w), dtype=audit_dtype),
                "raqic_interference_norm_error": np.zeros((h, w), dtype=audit_dtype),
                "raqic_interference_illegal_mass": np.zeros((h, w), dtype=audit_dtype),
                "raqic_shadow_probabilities": _rest_probs((h, w), actions, dtype=audit_dtype),
                "raqic_shadow_readout": np.full(
                    (h, w), int(Action.REST), dtype=DEFAULT_READOUT_DTYPE
                ),
            }
        )
    for name, init in specs.items():
        arr = getattr(state, name, None)
        if not isinstance(arr, np.ndarray) or arr.shape != init.shape:
            setattr(state, name, np.array(init, copy=True))
        elif arr.dtype != init.dtype:
            # Preserve checkpoint/live values when only the declared precision
            # changes. Reinitializing would silently erase authoritative RAQIC
            # recursion at the exact point where audit64 is intended to protect it.
            setattr(state, name, arr.astype(init.dtype, copy=True))
    dead = (state.health <= 0.0) | state.obstacle
    if np.any(dead):
        for name in (
            "raqic_probabilities",
            "raqic_parent_intention",
            "raqic_debug_density_diag",
            "raqic_pre_mixer_probabilities",
            "raqic_resonant_parent_intention",
            "raqic_shadow_probabilities",
        ):
            arr = getattr(state, name, None)
            if isinstance(arr, np.ndarray) and arr.ndim == 3:
                arr[dead, :] = 0.0
                arr[dead, int(Action.REST)] = 1.0
        for name in (
            "raqic_readout",
            "raqic_record_action",
            "raqic_legacy_shadow_readout",
            "raqic_shadow_readout",
        ):
            arr = getattr(state, name, None)
            if isinstance(arr, np.ndarray):
                arr[dead] = int(Action.REST)


def reset_raqic_audit_flags(state: WorldState) -> None:
    flags = getattr(state, "raqic_audit_flags", None)
    if isinstance(flags, np.ndarray):
        flags.fill(0)


def quiesce_dead_raqic_fields(state: WorldState) -> None:
    """Force dead/obstacle RAQIC records to REST after physical consequences."""
    probs = getattr(state, "raqic_probabilities", None)
    if not isinstance(probs, np.ndarray):
        return
    dead = (state.health <= 0.0) | state.obstacle
    if not np.any(dead):
        return
    for name in (
        "raqic_probabilities",
        "raqic_parent_intention",
        "raqic_debug_density_diag",
        "raqic_pre_mixer_probabilities",
        "raqic_resonant_parent_intention",
        "raqic_shadow_probabilities",
    ):
        arr = getattr(state, name, None)
        if isinstance(arr, np.ndarray) and arr.ndim == 3:
            arr[dead, :] = 0.0
            arr[dead, int(Action.REST)] = 1.0
    for name in (
        "raqic_readout",
        "raqic_record_action",
        "raqic_legacy_shadow_readout",
        "raqic_shadow_readout",
    ):
        arr = getattr(state, name, None)
        if isinstance(arr, np.ndarray):
            arr[dead] = int(Action.REST)
    for name in (
        "raqic_utility_innovation",
        "raqic_phase_alignment",
        "raqic_parent_action_phase",
        "raqic_parent_action_coherence",
    ):
        arr = getattr(state, name, None)
        if isinstance(arr, np.ndarray) and arr.ndim == 3:
            arr[dead, :] = 0.0
    for name in (
        "raqic_interference_delta_l1",
        "raqic_policy_kl",
        "raqic_utility_projection_fraction",
        "raqic_utility_score_cosine",
        "raqic_utility_orthogonality_residual",
        "raqic_utility_innovation_norm",
        "raqic_interference_norm_error",
        "raqic_interference_illegal_mass",
    ):
        arr = getattr(state, name, None)
        if isinstance(arr, np.ndarray):
            arr[dead] = 0.0
