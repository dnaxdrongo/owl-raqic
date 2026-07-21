"""Allocate optional state arrays used by advanced simulation features.

The advanced build adds optional dense arrays while preserving backwards
compatibility with  snapshots/tests/manual constructors. Call
``ensure_advanced_fields`` before using advanced arrays.
"""

from __future__ import annotations

from typing import cast

import numpy as np

from owl.core.actions import Action
from owl.core.config import SimulationConfig
from owl.core.constants import DEFAULT_FLOAT_DTYPE, DEFAULT_INT_DTYPE
from owl.core.state import WorldState, action_shape, channel_shape, field_shape

_MOORE_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def moore_directions() -> tuple[tuple[int, int], ...]:
    """Return the canonical eight-neighbor direction order."""
    return _MOORE_DIRECTIONS


def allocate_new_ow_id(state: WorldState) -> int:
    """Return a globally fresh OW identity and advance ``state.next_ow_id``.

    IDs are never derived from spatial coordinates after initialization. This
    preserves identity through movement and prevents a child born into a vacated
    cell from reusing a former occupant's spatial id.
    """
    alive_ids = np.asarray(state.occupancy, dtype=np.int64)
    max_seen = int(np.max(alive_ids)) if alive_ids.size else 0
    current = int(getattr(state, "next_ow_id", 1))
    new_id = max(current, max_seen + 1, 1)
    state.next_ow_id = int(new_id + 1)
    return int(new_id)


def _zeros(shape: tuple[int, ...]) -> np.ndarray:
    return np.zeros(shape, dtype=DEFAULT_FLOAT_DTYPE)


def _ones(shape: tuple[int, ...]) -> np.ndarray:
    return np.ones(shape, dtype=DEFAULT_FLOAT_DTYPE)


def ensure_advanced_fields(state: WorldState, cfg: SimulationConfig) -> WorldState:
    """Allocate missing optional advanced simulation arrays in place.

    The function is idempotent. It allocates arrays with the same grid/action/
    channel shapes as the canonical  state, and it repairs shape drift
    after loading alternate snapshots.
    """
    h, w = field_shape(state)
    ah, aw, actions = action_shape(state)
    ch, cw, channels = channel_shape(state)
    if (ah, aw) != (h, w) or (ch, cw) != (h, w):
        raise ValueError("advanced field allocation requires consistent cell/action/channel shapes")

    cell_shape = (h, w)
    action_shape_ = (h, w, actions)
    channel_shape_ = (h, w, channels)
    directions = len(_MOORE_DIRECTIONS)
    genome_len = int(getattr(cfg.reproduction, "genome_length", 8))

    for name in (
        "digestion",
        "waste",
        "age_stress",
        "last_intake",
        "prediction_error",
        "starvation_debt",
        "movement_loop_score",
        "development_stage",
        "symbiosis",
        "pre_resource",
        "pre_health",
        "pre_food",
        "pre_starvation_debt",
        "last_decision_urgency",
        "last_homeostatic_error",
        "noetic_B",
        "noetic_M",
        "noetic_P",
        "noetic_C",
        "noetic_K",
        "noetic_Theta",
        "noetic_N",
    ):
        arr = getattr(state, name, None)
        if not isinstance(arr, np.ndarray) or arr.shape != cell_shape:
            setattr(state, name, _zeros(cell_shape))

    if (
        not isinstance(state.last_death_mask, np.ndarray)
        or state.last_death_mask.shape != cell_shape
    ):
        state.last_death_mask = np.zeros(cell_shape, dtype=np.bool_)

    if (
        not isinstance(state.last_movement_action, np.ndarray)
        or state.last_movement_action.shape != cell_shape
    ):
        state.last_movement_action = np.full(cell_shape, int(Action.REST), dtype=DEFAULT_INT_DTYPE)

    for name in (
        "last_utilities",
        "last_logits",
        "last_action_probabilities",
        "action_cooldown",
        "pre_authority",
        "pre_utilities",
        "pre_parent_bias",
        "last_survival_value",
    ):
        arr = getattr(state, name, None)
        if not isinstance(arr, np.ndarray) or arr.shape != action_shape_:
            setattr(state, name, _zeros(action_shape_))

    if not isinstance(
        state.last_macro_probabilities, np.ndarray
    ) or state.last_macro_probabilities.shape != (h, w, actions + 1):
        state.last_macro_probabilities = _zeros((h, w, actions + 1))

    if (
        not isinstance(state.last_chosen_macro, np.ndarray)
        or state.last_chosen_macro.shape != cell_shape
    ):
        state.last_chosen_macro = np.full(cell_shape, int(Action.REST), dtype=DEFAULT_INT_DTYPE)

    if (
        not isinstance(state.signal_source_id, np.ndarray)
        or state.signal_source_id.shape != channel_shape_
    ):
        state.signal_source_id = np.full(channel_shape_, -1, dtype=DEFAULT_INT_DTYPE)

    for name in ("deception_memory", "source_confidence"):
        arr = getattr(state, name, None)
        if not isinstance(arr, np.ndarray) or arr.shape != channel_shape_:
            setattr(state, name, _zeros(channel_shape_))

    if not isinstance(state.neighbor_trust, np.ndarray) or state.neighbor_trust.shape != (
        h,
        w,
        directions,
        channels,
    ):
        state.neighbor_trust = _ones((h, w, directions, channels))

    for name in ("phase_frequency", "phase_lag", "parent_weight"):
        arr = getattr(state, name, None)
        if not isinstance(arr, np.ndarray) or arr.shape != cell_shape:
            setattr(state, name, _zeros(cell_shape))
    assert state.phase_frequency is not None
    assert state.phase_lag is not None
    assert state.parent_weight is not None
    if np.all(state.phase_frequency == 0):
        state.phase_frequency[...] = np.float32(cfg.phase.base_omega)
    if np.all(state.parent_weight == 0):
        state.parent_weight[...] = np.float32(cfg.phase.parent_coupling)
    if hasattr(cfg, "hierarchy"):
        state.phase_lag[...] = np.clip(state.phase_lag, -np.pi, np.pi)

    if not isinstance(state.same_scale_weight, np.ndarray) or state.same_scale_weight.shape != (
        h,
        w,
        directions,
    ):
        state.same_scale_weight = np.full(
            (h, w, directions),
            cfg.phase.same_scale_coupling / directions,
            dtype=DEFAULT_FLOAT_DTYPE,
        )

    if not isinstance(state.genome, np.ndarray) or state.genome.shape != (
        h,
        w,
        genome_len,
    ):
        state.genome = np.zeros((h, w, genome_len), dtype=DEFAULT_FLOAT_DTYPE)

    assert state.digestion is not None
    assert state.waste is not None
    assert state.age_stress is not None
    assert state.starvation_debt is not None
    assert state.movement_loop_score is not None
    assert state.action_cooldown is not None
    assert state.deception_memory is not None
    assert state.source_confidence is not None
    assert state.neighbor_trust is not None
    assert state.same_scale_weight is not None
    assert state.parent_weight is not None
    assert state.phase_frequency is not None
    assert state.phase_lag is not None
    assert state.genome is not None
    assert state.development_stage is not None
    assert state.symbiosis is not None

    # Patch-level advanced arrays.
    ph, pw = state.patches.integration.shape
    for name in (
        "centroid_y",
        "centroid_x",
        "velocity_y",
        "velocity_x",
        "prediction_error",
        "alive_density",
        "food_mean",
        "starvation_debt_mean",
        "reproduction_fraction",
        "movement_fraction",
        "feed_fraction",
        "death_pressure",
        "patch_crisis",
        "patch_carrying_pressure",
        "noetic_B",
        "noetic_M",
        "noetic_P",
        "noetic_C",
        "noetic_K",
        "noetic_Theta",
        "noetic_N",
    ):
        arr = getattr(state.patches, name, None)
        if not isinstance(arr, np.ndarray) or arr.shape != (ph, pw):
            setattr(state.patches, name, _zeros((ph, pw)))

    np.clip(state.digestion, 0.0, 1.0, out=state.digestion)
    np.clip(state.waste, 0.0, 1.0, out=state.waste)
    np.clip(state.age_stress, 0.0, 1.0, out=state.age_stress)
    np.clip(state.starvation_debt, 0.0, 1.0, out=state.starvation_debt)
    np.clip(state.movement_loop_score, 0.0, 1.0, out=state.movement_loop_score)
    np.clip(state.action_cooldown, 0.0, 1.0, out=state.action_cooldown)
    if state.last_action_probabilities is not None:
        sums = state.last_action_probabilities.sum(axis=-1, keepdims=True)
        np.divide(
            state.last_action_probabilities,
            np.maximum(sums, 1e-8),
            out=state.last_action_probabilities,
        )
    np.clip(state.deception_memory, 0.0, 1.0, out=state.deception_memory)
    np.clip(state.source_confidence, 0.0, 1.0, out=state.source_confidence)
    np.clip(state.neighbor_trust, 0.0, 1.0, out=state.neighbor_trust)
    np.clip(state.same_scale_weight, 0.0, 1.0, out=state.same_scale_weight)
    np.clip(state.parent_weight, 0.0, 1.0, out=state.parent_weight)
    np.clip(state.genome, 0.0, 1.0, out=state.genome)
    np.clip(state.development_stage, 0.0, 1.0, out=state.development_stage)
    np.clip(state.symbiosis, 0.0, 1.0, out=state.symbiosis)
    for name in (
        "pre_resource",
        "pre_health",
        "pre_food",
        "pre_starvation_debt",
        "last_decision_urgency",
        "last_homeostatic_error",
        "noetic_B",
        "noetic_M",
        "noetic_P",
        "noetic_C",
        "noetic_K",
        "noetic_Theta",
        "noetic_N",
    ):
        arr = getattr(state, name, None)
        if isinstance(arr, np.ndarray):
            np.clip(arr, 0.0, 1.0, out=arr)
    if isinstance(state.last_survival_value, np.ndarray):
        np.clip(state.last_survival_value, 0.0, 1.0, out=state.last_survival_value)
    if isinstance(state.last_macro_probabilities, np.ndarray):
        np.clip(state.last_macro_probabilities, 0.0, 1.0, out=state.last_macro_probabilities)
    live_ids = state.occupancy[state.occupancy >= 0]
    max_id = int(np.max(live_ids)) if live_ids.size else 0
    start_id = int(getattr(getattr(cfg, "identity", object()), "start_id", 1))
    if int(getattr(state, "next_ow_id", 1)) <= max_id:
        state.next_ow_id = max(max_id + 1, start_id)
    return state


def ensure_action_transition_fields(state: WorldState, cfg: SimulationConfig) -> WorldState:
    """Allocate action-transition arrays only when the explicit contract is enabled."""
    transition = cfg.action_transitions
    if not bool(transition.enabled):
        return state
    h, w = field_shape(state)
    cell = (h, w)
    families = 2
    directions = len(_MOORE_DIRECTIONS)
    for name in (
        "active_sense_food_memory",
        "active_sense_toxin_memory",
        "active_sense_alive_memory",
    ):
        value = getattr(state, name, None)
        if not isinstance(value, np.ndarray) or value.shape != cell:
            setattr(state, name, np.zeros(cell, dtype=DEFAULT_FLOAT_DTYPE))
    for name in (
        "active_sense_ttl",
        "active_sense_new_cell_count",
        "active_sense_new_target_count",
        "flee_compiled_action",
        "pursue_compiled_action",
        "compiled_execution_action",
    ):
        value = getattr(state, name, None)
        if not isinstance(value, np.ndarray) or value.shape != cell:
            fill = -1 if "compiled_action" in name else 0
            setattr(state, name, np.full(cell, fill, dtype=DEFAULT_INT_DTYPE))
    for name in (
        "action_target_y",
        "action_target_x",
        "action_target_ow_id",
        "action_target_kind",
        "action_target_source",
    ):
        value = getattr(state, name, None)
        if not isinstance(value, np.ndarray) or value.shape != (h, w, families):
            setattr(
                state,
                name,
                np.full((h, w, families), -1, dtype=DEFAULT_INT_DTYPE),
            )
    for name in ("action_target_distance", "action_target_confidence"):
        value = getattr(state, name, None)
        if not isinstance(value, np.ndarray) or value.shape != (h, w, families):
            setattr(state, name, np.zeros((h, w, families), dtype=DEFAULT_FLOAT_DTYPE))
    for name in ("action_direction_y", "action_direction_x"):
        value = getattr(state, name, None)
        shape = (h, w, families, directions)
        if not isinstance(value, np.ndarray) or value.shape != shape:
            setattr(state, name, np.full(shape, -1, dtype=DEFAULT_INT_DTYPE))
    value = state.action_direction_executable
    if not isinstance(value, np.ndarray) or value.shape != (h, w, families, directions):
        state.action_direction_executable = np.zeros(
            (h, w, families, directions), dtype=np.bool_
        )
    for name in (
        "action_direction_score",
        "action_direction_distance_delta",
        "action_direction_hazard",
        "action_direction_opportunity",
    ):
        value = getattr(state, name, None)
        shape = (h, w, families, directions)
        if not isinstance(value, np.ndarray) or value.shape != shape:
            setattr(state, name, np.zeros(shape, dtype=DEFAULT_FLOAT_DTYPE))
    assert state.compiled_execution_action is not None
    state.compiled_execution_action[...] = state.readout
    return state


def move_advanced_cell_fields(
    state: WorldState, source: tuple[int, int], target: tuple[int, int]
) -> None:
    """Move optional advanced per-cell arrays together with cell identity.

    The base movement routine moves canonical baseline fields first. This helper moves
    optional advanced diagnostics and resets the source to a neutral empty-cell
    state so trajectory/decision diagnostics follow the OW rather than remaining
    stuck to a grid coordinate.
    """
    sy, sx = map(int, source)
    ty, tx = map(int, target)
    shape = field_shape(state)
    if not (
        0 <= sy < shape[0] and 0 <= sx < shape[1] and 0 <= ty < shape[0] and 0 <= tx < shape[1]
    ):
        raise ValueError(
            f"advanced movement positions out of bounds: {source}->{target} for {shape}"
        )

    for name in (
        "digestion",
        "waste",
        "age_stress",
        "last_intake",
        "prediction_error",
        "starvation_debt",
        "movement_loop_score",
        "development_stage",
        "symbiosis",
        "phase_frequency",
        "phase_lag",
        "parent_weight",
        "pre_resource",
        "pre_health",
        "pre_food",
        "pre_starvation_debt",
        "last_decision_urgency",
        "last_homeostatic_error",
        "noetic_B",
        "noetic_M",
        "noetic_P",
        "noetic_C",
        "noetic_K",
        "noetic_Theta",
        "noetic_N",
    ):
        arr = getattr(state, name, None)
        if isinstance(arr, np.ndarray) and arr.shape == shape:
            arr[ty, tx] = arr[sy, sx]
            arr[sy, sx] = 0

    for name in (
        "last_utilities",
        "last_logits",
        "last_action_probabilities",
        "action_cooldown",
        "pre_authority",
        "pre_utilities",
        "pre_parent_bias",
        "last_survival_value",
        "last_macro_probabilities",
        "deception_memory",
        "source_confidence",
    ):
        arr = getattr(state, name, None)
        if isinstance(arr, np.ndarray) and arr.shape[:2] == shape:
            arr[ty, tx, ...] = arr[sy, sx, ...]
            arr[sy, sx, ...] = 0

    if isinstance(state.neighbor_trust, np.ndarray) and state.neighbor_trust.shape[:2] == shape:
        state.neighbor_trust[ty, tx, ...] = state.neighbor_trust[sy, sx, ...]
        state.neighbor_trust[sy, sx, ...] = 1.0
    if (
        isinstance(state.same_scale_weight, np.ndarray)
        and state.same_scale_weight.shape[:2] == shape
    ):
        state.same_scale_weight[ty, tx, ...] = state.same_scale_weight[sy, sx, ...]
        state.same_scale_weight[sy, sx, ...] = 0
    if isinstance(state.genome, np.ndarray) and state.genome.shape[:2] == shape:
        state.genome[ty, tx, :] = state.genome[sy, sx, :]
        state.genome[sy, sx, :] = 0

    if (
        isinstance(state.last_movement_action, np.ndarray)
        and state.last_movement_action.shape == shape
    ):
        # The movement module sets the target to the just-executed move after
        # this helper returns; here we only clear the source.
        state.last_movement_action[sy, sx] = int(Action.REST)
    if isinstance(state.last_chosen_macro, np.ndarray) and state.last_chosen_macro.shape == shape:
        state.last_chosen_macro[ty, tx] = state.last_chosen_macro[sy, sx]
        state.last_chosen_macro[sy, sx] = int(Action.REST)


def living_mask(state: WorldState) -> np.ndarray:
    """Return living, non-obstacle mask."""
    return (state.health > 0.0) & (~state.obstacle)


def action_entropy(possibility: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Return normalized entropy over the last axis."""
    p = np.clip(np.asarray(possibility, dtype=np.float32), 0.0, 1.0)
    denom = np.maximum(p.sum(axis=-1, keepdims=True), epsilon)
    p = p / denom
    k = max(p.shape[-1], 2)
    entropy = -np.sum(np.where(p > 0, p * np.log(np.maximum(p, epsilon)), 0.0), axis=-1) / np.log(k)
    return cast(np.ndarray, np.clip(entropy, 0.0, 1.0).astype(np.float32))


def cooldown_decay(
    state: WorldState, readout: np.ndarray, decay: float = 0.92, impulse: float = 0.20
) -> None:
    """Update action cooldown traces in place."""
    if state.action_cooldown is None:
        return
    state.action_cooldown *= np.float32(np.clip(decay, 0.0, 1.0))
    h, w = readout.shape
    yy, xx = np.indices((h, w))
    idx = np.clip(readout.astype(np.int64), 0, state.action_cooldown.shape[-1] - 1)
    state.action_cooldown[yy, xx, idx] += np.float32(impulse)
    np.clip(state.action_cooldown, 0.0, 1.0, out=state.action_cooldown)


def rest_probability_tensor(shape: tuple[int, int, int]) -> np.ndarray:
    """Return one-hot REST probability cube."""
    out = np.zeros(shape, dtype=np.float32)
    out[..., int(Action.REST)] = 1.0
    return out
