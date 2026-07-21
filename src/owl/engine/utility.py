"""Action utility and motivation functions.

This module implements the motivational layer of Observer-Window Life. It does
not actualize actions and it does not mutate :class:`WorldState`; it only returns
cell-level drive fields and action utility tensors. The possibility/actualization
layer consumes these utilities later.
"""

from __future__ import annotations

from typing import cast

import numpy as np

from owl.core.actions import DIAGONAL_MOVES, MOVE_DELTAS, REVERSE_MOVE_ACTION, Action, SignalChannel
from owl.core.config import SimulationConfig
from owl.core.constants import DEFAULT_FLOAT_DTYPE
from owl.core.state import WorldState, action_shape, field_shape
from owl.engine.sensing import (
    compute_crowding,
    compute_local_food_pressure,
    compute_local_toxin_pressure,
    compute_novelty,
)
from owl.kernels.numpy_kernels import gradient_wrap, neighbor_mean_wrap


def _channel_or_zero(
    state: WorldState, channel: SignalChannel, cfg: SimulationConfig
) -> np.ndarray:
    """Return a received-signal channel as a cell field, or zeros if absent."""
    shape = field_shape(state)
    idx = int(channel)
    if state.signal_reception.ndim != 3 or state.signal_reception.shape[:2] != shape:
        raise ValueError(
            "state.signal_reception must have shape "
            f"(height, width, channels), got {state.signal_reception.shape}"
        )
    if idx >= min(cfg.communication.num_channels, state.signal_reception.shape[-1]):
        return np.zeros(shape, dtype=DEFAULT_FLOAT_DTYPE)
    return np.clip(state.signal_reception[..., idx], 0.0, 1.0).astype(
        DEFAULT_FLOAT_DTYPE, copy=False
    )


def _validate_parent_bias(state: WorldState, parent_bias: np.ndarray) -> np.ndarray:
    """Return parent bias as a float32 action tensor with exact cell-action shape."""
    expected = action_shape(state)
    bias = np.asarray(parent_bias, dtype=np.float32)
    if bias.shape != expected:
        raise ValueError(f"parent_bias must have shape {expected}, got {bias.shape}")
    if not np.all(np.isfinite(bias)):
        raise ValueError("parent_bias must contain only finite values")
    return bias.astype(DEFAULT_FLOAT_DTYPE, copy=False)


def _alive_mask(state: WorldState) -> np.ndarray:
    """Return living, non-obstacle mask as a float32 cell field."""
    return ((state.health > 0.0) & (~state.obstacle)).astype(DEFAULT_FLOAT_DTYPE)


def compute_internal_drives(state: WorldState, cfg: SimulationConfig) -> dict[str, np.ndarray]:
    """Compute cell-level internal drives.

    Parameters
    ----------
    state:
        Runtime dense state. This function does not mutate state.
    cfg:
        Simulation coefficients.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary of ``(height, width)`` float32 fields in ``[0, 1]``. Keys
        include ``hunger``, ``pain``, ``boundary_stress``, ``crowding``,
        ``food_pressure``, ``toxin_pressure``, ``novelty``, and
        ``social_need``.

    Notes
    -----
    These drives are operational control signals, not human emotions. They are
    the physical/communication inputs used by the utility layer.
    """
    shape = field_shape(state)
    for name in ("resource", "health", "boundary", "memory", "integration"):
        if getattr(state, name).shape != shape:
            raise ValueError(
                f"state.{name} must have cell shape {shape}, got {getattr(state, name).shape}"
            )

    alive = _alive_mask(state)

    hunger = 1.0 - np.clip(
        state.resource / max(float(cfg.resources.max_resource), cfg.actions.epsilon), 0.0, 1.0
    )
    pain = 1.0 - np.clip(state.health, 0.0, 1.0)
    boundary_stress = 1.0 - np.clip(state.boundary, 0.0, 1.0)
    crowding = compute_crowding(state)
    food_pressure = compute_local_food_pressure(state, cfg)
    toxin_pressure = compute_local_toxin_pressure(state, cfg)
    if bool(getattr(cfg.action_transitions, "enabled", False)):
        ttl = getattr(state, "active_sense_ttl", None)
        active_food = getattr(state, "active_sense_food_memory", None)
        active_toxin = getattr(state, "active_sense_toxin_memory", None)
        if isinstance(ttl, np.ndarray) and isinstance(active_food, np.ndarray):
            remembered = ttl > 0
            food_pressure = np.where(remembered, active_food, food_pressure)
        if isinstance(ttl, np.ndarray) and isinstance(active_toxin, np.ndarray):
            remembered = ttl > 0
            toxin_pressure = np.where(remembered, active_toxin, toxin_pressure)
    novelty = compute_novelty(state, cfg)

    # Social need is a bounded proxy for low same-scale coherence: a cooperative
    # cell surrounded by living neighbors but with low integration wants
    # coordination more than an isolated cell does.
    local_integration = neighbor_mean_wrap(np.clip(state.integration, 0.0, 1.0))
    social_need = (
        np.clip(state.cooperation, 0.0, 1.0)
        * crowding
        * (1.0 - np.clip(local_integration, 0.0, 1.0))
    )

    drives = {
        "hunger": hunger,
        "pain": pain,
        "boundary_stress": boundary_stress,
        "crowding": crowding,
        "food_pressure": food_pressure,
        "toxin_pressure": toxin_pressure,
        "novelty": novelty,
        "social_need": social_need,
    }

    for key, value in list(drives.items()):
        arr = np.asarray(value, dtype=DEFAULT_FLOAT_DTYPE)
        if arr.shape != shape:
            raise ValueError(f"drive {key!r} must have shape {shape}, got {arr.shape}")
        arr = np.clip(arr, 0.0, 1.0)
        arr[alive <= 0.0] = 0.0
        drives[key] = arr.astype(DEFAULT_FLOAT_DTYPE, copy=False)

    return drives


def _base_compute_utilities(
    state: WorldState, parent_bias: np.ndarray, cfg: SimulationConfig
) -> np.ndarray:
    """Compute action utility scores for every cell.

    Parameters
    ----------
    state:
        Runtime dense state. This function does not mutate state.
    parent_bias:
        Cell-level action-bias tensor with shape
        ``(height, width, len(Action))``. The tensor is validated here so shape
        errors surface before actualization; the main top-down logit effect is
        applied in :func:`owl.engine.actualization.compute_action_logits`.
    cfg:
        Simulation coefficients.

    Returns
    -------
    np.ndarray
        Float32 utility tensor with shape ``(height, width, len(Action))``.
        Utilities are finite but not clipped to ``[0, 1]`` because negative
        utility is meaningful before softmax actualization.
    """
    _validate_parent_bias(state, parent_bias)
    h, w = field_shape(state)
    utilities = np.zeros((h, w, len(Action)), dtype=DEFAULT_FLOAT_DTYPE)

    drives = compute_internal_drives(state, cfg)
    hunger = drives["hunger"]
    pain = drives["pain"]
    boundary_stress = drives["boundary_stress"]
    crowding = drives["crowding"]
    food_pressure = drives["food_pressure"]
    toxin_pressure = drives["toxin_pressure"]
    novelty = drives["novelty"]
    social_need = drives["social_need"]

    food_signal = _channel_or_zero(state, SignalChannel.FOOD, cfg)
    danger_signal = _channel_or_zero(state, SignalChannel.DANGER, cfg)
    threat_signal = _channel_or_zero(state, SignalChannel.THREAT, cfg)
    coord_signal = _channel_or_zero(state, SignalChannel.COORDINATION, cfg)
    distress_signal = _channel_or_zero(state, SignalChannel.DISTRESS, cfg)
    repro_signal = _channel_or_zero(state, SignalChannel.REPRODUCTION, cfg)
    territory_signal = _channel_or_zero(state, SignalChannel.TERRITORY, cfg)
    integration_signal = _channel_or_zero(state, SignalChannel.INTEGRATION, cfg)

    resource = np.clip(state.resource, 0.0, cfg.resources.max_resource)
    resource_norm = np.clip(
        resource / max(float(cfg.resources.max_resource), cfg.actions.epsilon), 0.0, 1.0
    )
    health = np.clip(state.health, 0.0, 1.0)
    boundary = np.clip(state.boundary, 0.0, 1.0)
    integration = np.clip(state.integration, 0.0, 1.0)
    memory = np.clip(state.memory, 0.0, 1.0)
    alive = _alive_mask(state)

    # Rest is useful for damaged/stressed cells and mildly useful under high
    # toxin pressure when movement/repair may not yet be selected.
    utilities[..., int(Action.REST)] = (
        0.15 * pain + 0.10 * boundary_stress + 0.05 * toxin_pressure - 0.02 * hunger
    )

    # Sense is the safe information-gathering action. It is attractive under
    # novelty, signal conflict, or uncertainty but does not directly repair.
    signal_spread = np.mean(np.clip(state.signal_reception, 0.0, 1.0), axis=-1)
    utilities[..., int(Action.SENSE)] = 0.45 * novelty + 0.20 * signal_spread + 0.10 * memory - 0.02

    # Environmental feeding converts food into internal resource in a later pass.
    utilities[..., int(Action.FEED)] = (
        1.50 * hunger * food_pressure * np.clip(state.grazing, 0.0, 1.0)
        + 0.35 * food_signal
        - 0.10 * toxin_pressure
        - 0.05
    )
    emergency = resource_norm <= float(cfg.resources.emergency_feed_threshold)
    utilities[..., int(Action.FEED)] += np.float32(
        cfg.resources.emergency_feed_boost
    ) * hunger * food_pressure * emergency.astype(np.float32) + np.float32(0.50) * hunger * np.clip(
        state.food, 0.0, 1.0
    )

    # Inhibition is a defensive/competitive action. It becomes useful under
    # threat, territory conflict, and aggression.
    utilities[..., int(Action.INHIBIT)] = (
        0.75 * threat_signal * np.clip(state.aggression, 0.0, 1.0)
        + 0.35 * danger_signal
        + 0.25 * territory_signal
        - 0.08
    )

    # Integration is a constructive action: coordination pressure, integration
    # signal, memory, and social need all make it more attractive.
    utilities[..., int(Action.INTEGRATE)] = (
        0.90 * coord_signal
        + 0.45 * integration_signal
        + 0.35 * social_need
        + 0.30 * memory
        + 0.20 * integration
        - 0.10
    )

    # Repair is the action for boundary and health damage.
    utilities[..., int(Action.REPAIR)] = (
        1.10 * boundary_stress + 0.80 * pain + 0.25 * distress_signal - 0.15 * hunger - 0.05
    )

    utilities[..., int(Action.REPRODUCE)] = (
        np.clip(state.reproduction_rate, 0.0, 1.0)
        * resource_norm
        * health
        * boundary
        * np.maximum(integration, 0.05)
        + 0.20 * repro_signal
        - 0.40
    )

    prey_pressure = neighbor_mean_wrap(
        ((state.health > 0.0) & (~state.obstacle)).astype(np.float32)
    )
    utilities[..., int(Action.INGEST)] = (
        1.20 * np.clip(state.predation, 0.0, 1.0) * prey_pressure * hunger
        + 0.25 * threat_signal * np.clip(state.aggression, 0.0, 1.0)
        + 0.15 * distress_signal * np.clip(state.predation, 0.0, 1.0)
        - 0.35
    )

    utilities[..., int(Action.FLEE)] = (
        0.95 * (danger_signal + toxin_pressure + threat_signal) * np.clip(state.mobility, 0.0, 1.0)
        + 0.25 * pain
        - 0.05
    )
    utilities[..., int(Action.PURSUE)] = (
        0.70 * np.clip(state.predation, 0.0, 1.0) * prey_pressure
        + 0.30 * threat_signal * np.clip(state.aggression, 0.0, 1.0)
        - 0.10
    )

    # Advanced topology actions are extension hooks with conservative
    # utility so authority/actualization can still suppress them cleanly.
    utilities[..., int(Action.EXPEL)] = -0.35 + 0.15 * boundary_stress
    utilities[..., int(Action.SPLIT)] = -0.45 + 0.20 * boundary_stress + 0.10 * (1.0 - integration)
    utilities[..., int(Action.MERGE)] = -0.40 + 0.20 * coord_signal + 0.10 * crowding

    utilities = add_movement_utilities(utilities, state, drives, cfg)
    utilities = add_communication_utilities(utilities, state, drives, cfg)

    utilities *= alive[..., None]
    utilities[state.obstacle, :] = 0.0

    if not np.all(np.isfinite(utilities)):
        raise ValueError("computed utilities contain non-finite values")
    return utilities.astype(DEFAULT_FLOAT_DTYPE, copy=False)


def add_movement_utilities(
    utilities: np.ndarray,
    state: WorldState,
    drives: dict[str, np.ndarray],
    cfg: SimulationConfig,
) -> np.ndarray:
    """Add survival-weighted direction-specific movement utilities.

    Movement combines gradient-like attraction/avoidance with direct neighbor
    lookahead. Diagonal movement is controlled by config. Optional advanced
    inertia rewards repeating the last movement direction and penalizes immediate
    reversal, reducing two-cell bouncing loops.
    """
    expected = action_shape(state)
    out = np.asarray(utilities, dtype=np.float32).copy()
    if out.shape != expected:
        raise ValueError(f"utilities must have shape {expected}, got {out.shape}")

    hunger = np.asarray(drives["hunger"], dtype=np.float32)
    novelty = np.asarray(drives["novelty"], dtype=np.float32)
    toxin_pressure = np.asarray(drives["toxin_pressure"], dtype=np.float32)
    food_pressure = np.asarray(drives["food_pressure"], dtype=np.float32)
    if hunger.shape != field_shape(state):
        raise ValueError("drive fields must match cell shape")

    if state.signal_reception.shape[-1] > int(SignalChannel.FOOD):
        food_signal_field = np.clip(state.signal_reception[..., int(SignalChannel.FOOD)], 0.0, 1.0)
    else:
        food_signal_field = np.zeros(field_shape(state), dtype=np.float32)
    if state.signal_reception.shape[-1] > int(SignalChannel.DANGER):
        danger_signal_field = np.clip(
            state.signal_reception[..., int(SignalChannel.DANGER)], 0.0, 1.0
        )
    else:
        danger_signal_field = np.zeros(field_shape(state), dtype=np.float32)
    if state.signal_reception.shape[-1] > int(SignalChannel.THREAT):
        threat_signal_field = np.clip(
            state.signal_reception[..., int(SignalChannel.THREAT)], 0.0, 1.0
        )
    else:
        threat_signal_field = np.zeros(field_shape(state), dtype=np.float32)
    if state.signal_reception.shape[-1] > int(SignalChannel.COORDINATION):
        coord_signal_field = np.clip(
            state.signal_reception[..., int(SignalChannel.COORDINATION)], 0.0, 1.0
        )
    else:
        coord_signal_field = np.zeros(field_shape(state), dtype=np.float32)

    food_drive = np.clip(food_pressure + 0.75 * food_signal_field, 0.0, 1.0)
    danger_drive = np.clip(toxin_pressure + danger_signal_field + threat_signal_field, 0.0, 1.0)
    social_drive = coord_signal_field

    food_gy, food_gx = gradient_wrap(food_drive)
    danger_gy, danger_gx = gradient_wrap(danger_drive)
    social_gy, social_gx = gradient_wrap(social_drive)

    mobility = np.clip(state.mobility, 0.0, 1.0)
    curiosity = np.clip(state.curiosity, 0.0, 1.0)
    occupied_or_obstacle = ((state.occupancy >= 0) | state.obstacle | (state.health > 0.0)).astype(
        np.float32
    )
    last = getattr(state, "last_movement_action", None)

    for action, (dy, dx) in MOVE_DELTAS.items():
        if (not cfg.actions.diagonal_movement_enabled) and action in DIAGONAL_MOVES:
            out[..., int(action)] = -1.0
            continue

        toward_food = dy * food_gy + dx * food_gx
        away_danger = -(dy * danger_gy + dx * danger_gx)
        toward_social = dy * social_gy + dx * social_gx

        # Neighbor lookahead: roll by -dy/-dx so field[y,x] reads the candidate
        # target's state for the action direction.
        neighbor_food = np.roll(np.roll(np.clip(state.food, 0.0, 1.0), -dy, axis=0), -dx, axis=1)
        neighbor_toxin = np.roll(np.roll(np.clip(state.toxin, 0.0, 1.0), -dy, axis=0), -dx, axis=1)
        neighbor_blocked = np.roll(np.roll(occupied_or_obstacle, -dy, axis=0), -dx, axis=1)

        directional_value = (
            np.float32(cfg.actions.movement_food_weight) * hunger * neighbor_food
            - np.float32(cfg.actions.movement_toxin_weight) * neighbor_toxin
            - np.float32(cfg.actions.movement_crowding_weight) * neighbor_blocked
        )
        inertia = 0.0
        if isinstance(last, np.ndarray) and last.shape == field_shape(state):
            inertia = np.float32(cfg.actions.movement_persistence_bonus) * (
                last == int(action)
            ).astype(np.float32) - np.float32(cfg.actions.movement_reverse_penalty) * (
                last == int(REVERSE_MOVE_ACTION[action])
            ).astype(np.float32)

        out[..., int(action)] = (
            0.95 * mobility * hunger * toward_food
            + 1.10 * mobility * away_danger
            + 0.25 * mobility * curiosity * novelty
            + 0.20 * mobility * np.clip(state.cooperation, 0.0, 1.0) * toward_social
            + mobility * directional_value
            + inertia
            - 0.05
        )

    return out.astype(DEFAULT_FLOAT_DTYPE, copy=False)


def add_communication_utilities(
    utilities: np.ndarray,
    state: WorldState,
    drives: dict[str, np.ndarray],
    cfg: SimulationConfig,
) -> np.ndarray:
    """Add utility terms for the universal ``COMMUNICATE`` action.

    Communication is available to all observer windows. This function only
    scores whether communication is useful; emission details are handled by the
    communication module and resource cost is applied there.
    """
    expected = action_shape(state)
    out = np.asarray(utilities, dtype=np.float32).copy()
    if out.shape != expected:
        raise ValueError(f"utilities must have shape {expected}, got {out.shape}")

    if not cfg.communication.enabled:
        out[..., int(Action.COMMUNICATE)] = -1.0
        return out.astype(DEFAULT_FLOAT_DTYPE, copy=False)

    food_signal = _channel_or_zero(state, SignalChannel.FOOD, cfg)
    danger_signal = _channel_or_zero(state, SignalChannel.DANGER, cfg)
    threat_signal = _channel_or_zero(state, SignalChannel.THREAT, cfg)
    coord_signal = _channel_or_zero(state, SignalChannel.COORDINATION, cfg)
    distress_signal = _channel_or_zero(state, SignalChannel.DISTRESS, cfg)
    repro_signal = _channel_or_zero(state, SignalChannel.REPRODUCTION, cfg)

    hunger = drives["hunger"]
    pain = drives["pain"]
    social_need = drives["social_need"]
    boundary_stress = drives["boundary_stress"]

    emit_capacity = (
        np.clip(state.emit_strength, 0.0, 1.0)
        * np.clip(state.emit_efficiency, 0.0, 1.0)
        * np.clip(state.resource, 0.0, cfg.resources.max_resource)
    )
    channel_interest = np.mean(np.clip(state.channel_emission_bias, 0.0, 1.0), axis=-1)

    out[..., int(Action.COMMUNICATE)] = (
        0.35 * food_signal * hunger
        + 0.70 * danger_signal
        + 0.35 * threat_signal
        + 0.55 * coord_signal
        + 0.45 * distress_signal
        + 0.20 * repro_signal
        + 0.45 * social_need
        + 0.25 * pain
        + 0.15 * boundary_stress
        + 0.25 * np.clip(state.cooperation, 0.0, 1.0)
        + 0.30 * emit_capacity
        + 0.15 * channel_interest
        - float(cfg.communication.base_emit_cost)
    )

    return out.astype(DEFAULT_FLOAT_DTYPE, copy=False)


# --- Advanced build overrides ------------------------------------------------
_mvp_compute_utilities = _base_compute_utilities


def compute_pragmatic_value(
    state: WorldState, drives: dict[str, np.ndarray], cfg: SimulationConfig
) -> np.ndarray:
    """Return baseline utility baseline used as pragmatic value."""
    return _mvp_compute_utilities(state, np.zeros(action_shape(state), dtype=np.float32), cfg)


def compute_epistemic_value(
    state: WorldState, drives: dict[str, np.ndarray], cfg: SimulationConfig
) -> np.ndarray:
    """Return action-value bonus for information gathering and coordination."""
    from owl.core.advanced import action_entropy

    h, w, actions = action_shape(state)
    out = np.zeros((h, w, actions), dtype=np.float32)
    entropy = action_entropy(state.possibility, cfg.actions.epsilon)
    novelty = drives.get("novelty", np.zeros((h, w), dtype=np.float32))
    low_memory = 1.0 - np.clip(state.memory, 0.0, 1.0)
    epistemic = np.clip(0.45 * novelty + 0.35 * entropy + 0.20 * low_memory, 0.0, 1.0)
    out[..., int(Action.SENSE)] = epistemic
    for action in MOVE_DELTAS:
        out[..., int(action)] = 0.25 * epistemic * np.clip(state.curiosity, 0.0, 1.0)
    out[..., int(Action.COMMUNICATE)] = 0.25 * epistemic * np.clip(state.cooperation, 0.0, 1.0)
    out[..., int(Action.INTEGRATE)] = 0.20 * epistemic
    return out


def compute_action_risk(
    state: WorldState, drives: dict[str, np.ndarray], cfg: SimulationConfig
) -> np.ndarray:
    """Return bounded action risk tensor."""
    h, w, actions = action_shape(state)
    out = np.zeros((h, w, actions), dtype=np.float32)
    toxin = drives.get("toxin_pressure", np.zeros((h, w), dtype=np.float32))
    pain = drives.get("pain", np.zeros((h, w), dtype=np.float32))
    boundary = drives.get("boundary_stress", np.zeros((h, w), dtype=np.float32))
    base = np.clip(0.45 * toxin + 0.30 * pain + 0.25 * boundary, 0.0, 1.0)
    for action in MOVE_DELTAS:
        out[..., int(action)] = 0.4 * base
    out[..., int(Action.INGEST)] = 0.6 * base + 0.2 * (1.0 - np.clip(state.predation, 0.0, 1.0))
    out[..., int(Action.REPRODUCE)] = 0.5 * (
        1.0 - np.clip(state.resource / cfg.resources.max_resource, 0.0, 1.0)
    )
    out[..., int(Action.FEED)] = 0.25 * toxin
    return np.clip(out, 0.0, 1.0)


def compute_action_effort(state: WorldState, cfg: SimulationConfig) -> np.ndarray:
    """Return bounded effort/action-cost tensor."""
    h, w, actions = action_shape(state)
    out = np.zeros((h, w, actions), dtype=np.float32)
    resource_low = 1.0 - np.clip(
        state.resource / max(float(cfg.resources.max_resource), cfg.actions.epsilon), 0.0, 1.0
    )
    for action in MOVE_DELTAS:
        out[..., int(action)] = (
            resource_low
            * cfg.resources.movement_cost
            / max(cfg.resources.max_resource, cfg.actions.epsilon)
        )
    out[..., int(Action.COMMUNICATE)] = (
        resource_low
        * cfg.communication.base_emit_cost
        / max(cfg.resources.max_resource, cfg.actions.epsilon)
    )
    out[..., int(Action.REPAIR)] = 0.15 * resource_low
    out[..., int(Action.REPRODUCE)] = 0.35 * resource_low
    out[..., int(Action.INGEST)] = 0.20 * resource_low
    return np.clip(out, 0.0, 1.0)


def _advanced_compute_utilities(
    state: WorldState, parent_bias: np.ndarray, cfg: SimulationConfig
) -> np.ndarray:
    """Compute utilities; advanced mode decomposes pragmatic/epistemic/risk/effort."""
    if not getattr(cfg.possibility, "advanced_enabled", False):
        out = _mvp_compute_utilities(state, parent_bias, cfg)
        from owl.core.advanced import ensure_advanced_fields

        ensure_advanced_fields(state, cfg)
        assert state.last_utilities is not None
        state.last_utilities[...] = out
        return out

    from owl.core.advanced import ensure_advanced_fields

    ensure_advanced_fields(state, cfg)
    assert state.last_utilities is not None
    _validate_parent_bias(state, parent_bias)
    drives = compute_internal_drives(state, cfg)
    pragmatic = compute_pragmatic_value(state, drives, cfg)
    epistemic = compute_epistemic_value(state, drives, cfg)
    risk = compute_action_risk(state, drives, cfg)
    effort = compute_action_effort(state, cfg)
    cooldown = state.action_cooldown if isinstance(state.action_cooldown, np.ndarray) else 0.0
    utilities = (
        pragmatic
        + np.float32(cfg.possibility.epistemic_weight) * epistemic
        - np.float32(cfg.possibility.risk_weight) * risk
        - np.float32(cfg.possibility.effort_weight) * effort
        - np.float32(cfg.possibility.cooldown_weight) * cooldown
    )
    utilities *= ((state.health > 0.0) & (~state.obstacle))[..., None]
    utilities[state.obstacle, :] = 0.0
    if not np.all(np.isfinite(utilities)):
        raise ValueError("advanced utilities contain non-finite values")
    state.last_utilities[...] = utilities.astype(np.float32, copy=False)
    return state.last_utilities.astype(np.float32, copy=True)


# --- Decision-homeostasis utility override -----------------------------------
def _smoothstep(x: np.ndarray, edge0: float, edge1: float) -> np.ndarray:
    """Bounded smoothstep for viability gates."""
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-8), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def _local_alive_density(state: WorldState) -> np.ndarray:
    alive = ((state.health > 0.0) & (~state.obstacle)).astype(np.float32)
    return cast(np.ndarray, np.clip(neighbor_mean_wrap(alive), 0.0, 1.0).astype(np.float32))


def _local_food_mean(state: WorldState) -> np.ndarray:
    return cast(
        np.ndarray,
        np.clip(
            neighbor_mean_wrap(np.clip(state.food, 0.0, 1.0).astype(np.float32)),
            0.0,
            1.0,
        ).astype(np.float32),
    )


def _upsampled_patch_attr(
    state: WorldState, cfg: SimulationConfig, name: str, default: float = 0.0
) -> np.ndarray:
    from owl.engine.aggregation import upsample_patch_field

    shape = field_shape(state)
    arr = getattr(state.patches, name, None)
    if isinstance(arr, np.ndarray):
        up = upsample_patch_field(np.clip(arr, 0.0, 1.0), cfg.world.patch_size)
        if up.shape == shape:
            return up.astype(np.float32, copy=False)
    return np.full(shape, default, dtype=np.float32)


def reproduction_viability_field(state: WorldState, cfg: SimulationConfig) -> np.ndarray:
    """Return bounded reproduction viability; no class quotas are imposed."""
    resource = np.clip(
        state.resource / max(float(cfg.resources.max_resource), float(cfg.actions.epsilon)),
        0.0,
        1.0,
    )
    resource_ok = _smoothstep(resource, 0.45, 0.85)
    health_ok = _smoothstep(np.clip(state.health, 0.0, 1.0), 0.60, 0.95)
    boundary_ok = _smoothstep(np.clip(state.boundary, 0.0, 1.0), 0.60, 0.95)
    local_density = _local_alive_density(state)
    local_food = _local_food_mean(state)
    patch_crisis = _upsampled_patch_attr(state, cfg, "patch_crisis", default=0.0)
    carrying = _upsampled_patch_attr(state, cfg, "patch_carrying_pressure", default=0.0)
    capacity = np.clip(
        1.0 - 0.65 * local_density - 0.75 * carrying - 0.75 * patch_crisis + 0.40 * local_food,
        0.0,
        1.0,
    )
    return cast(
        np.ndarray,
        np.clip(resource_ok * health_ok * boundary_ok * capacity, 0.0, 1.0).astype(np.float32),
    )


def compute_directional_survival_advantage(state: WorldState, cfg: SimulationConfig) -> np.ndarray:
    """Return direction survival advantages in action axis coordinates."""
    h, w = field_shape(state)
    out = np.zeros((h, w, len(Action)), dtype=np.float32)
    resource = np.clip(
        state.resource / max(float(cfg.resources.max_resource), float(cfg.actions.epsilon)),
        0.0,
        1.0,
    )
    hunger = np.clip(1.0 - resource, 0.0, 1.0)
    occupied = ((state.occupancy >= 0) | state.obstacle | (state.health > 0.0)).astype(np.float32)
    food = np.clip(state.food, 0.0, 1.0)
    toxin = np.clip(state.toxin, 0.0, 1.0)
    crowd = _local_alive_density(state)
    for action, (dy, dx) in MOVE_DELTAS.items():
        if action in DIAGONAL_MOVES and not cfg.actions.diagonal_movement_enabled:
            continue
        nfood = np.roll(np.roll(food, -dy, axis=0), -dx, axis=1)
        ntoxin = np.roll(np.roll(toxin, -dy, axis=0), -dx, axis=1)
        nblocked = np.roll(np.roll(occupied, -dy, axis=0), -dx, axis=1)
        advantage = 0.80 * hunger * nfood + 0.25 * (1.0 - crowd) - 0.70 * ntoxin - 0.45 * nblocked
        out[..., int(action)] = np.clip(advantage, 0.0, 1.0)
    return out


def compute_survival_value(
    state: WorldState, cfg: SimulationConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return action survival values, urgency, and homeostatic error.

    This is deliberately general: feeding, repair, movement, rest, sensing, and
    ingestion can each become optimal when their survival consequences are best
    and authority later allows them. Reproduction is penalized under urgency.
    """
    from owl.core.advanced import ensure_advanced_fields

    ensure_advanced_fields(state, cfg)
    h, w, actions = action_shape(state)
    out = np.zeros((h, w, actions), dtype=np.float32)
    res = np.clip(
        state.resource / max(float(cfg.resources.max_resource), float(cfg.actions.epsilon)),
        0.0,
        1.0,
    )
    health = np.clip(state.health, 0.0, 1.0)
    boundary = np.clip(state.boundary, 0.0, 1.0)
    starvation_source = (
        state.starvation_debt
        if isinstance(state.starvation_debt, np.ndarray)
        else np.asarray(1.0 - res)
    )
    starv: np.ndarray = np.asarray(np.clip(starvation_source, 0.0, 1.0), dtype=np.float32)
    food_local = np.clip(state.food, 0.0, 1.0)
    food_near = _local_food_mean(state)
    toxin = np.clip(state.toxin, 0.0, 1.0)
    patch_crisis = _upsampled_patch_attr(state, cfg, "patch_crisis", default=0.0)
    carrying = _upsampled_patch_attr(state, cfg, "patch_carrying_pressure", default=0.0)

    homeostatic_error = np.clip(
        0.35 * (1.0 - res) + 0.30 * starv + 0.20 * (1.0 - health) + 0.15 * (1.0 - boundary),
        0.0,
        1.0,
    ).astype(np.float32)
    urgency = np.clip(
        0.55 * homeostatic_error + 0.25 * toxin + 0.20 * patch_crisis, 0.0, 1.0
    ).astype(np.float32)

    out[..., int(Action.FEED)] += urgency * (0.95 * food_local + 0.45 * food_near + 0.10)
    out[..., int(Action.REPAIR)] += urgency * (
        0.85 * (1.0 - boundary) + 0.65 * (1.0 - health) + 0.10
    )
    out[..., int(Action.REST)] += urgency * 0.25 * (1.0 - food_local) * (1.0 - toxin)
    out[..., int(Action.SENSE)] += urgency * (0.30 * (1.0 - food_local) + 0.20 * patch_crisis)
    out[..., int(Action.FLEE)] += urgency * 0.70 * toxin * np.clip(state.mobility, 0.0, 1.0)
    out += compute_directional_survival_advantage(state, cfg) * urgency[..., None]
    prey_pressure = neighbor_mean_wrap(
        ((state.health > 0.0) & (~state.obstacle)).astype(np.float32)
    )
    out[..., int(Action.INGEST)] += (
        urgency * np.clip(state.predation, 0.0, 1.0) * prey_pressure * 0.50
    )
    rep_viability = reproduction_viability_field(state, cfg)
    out[..., int(Action.REPRODUCE)] += (
        0.25 * rep_viability * (1.0 - urgency)
        - 1.25 * np.maximum(urgency, patch_crisis)
        - 0.75 * carrying
    )
    out[..., int(Action.INTEGRATE)] += 0.15 * (1.0 - urgency) * np.clip(state.cooperation, 0.0, 1.0)
    out[..., int(Action.COMMUNICATE)] += 0.20 * patch_crisis + 0.10 * urgency * np.clip(
        state.cooperation, 0.0, 1.0
    )
    out = np.clip(out, 0.0, 1.0).astype(np.float32)
    return out, urgency, homeostatic_error


def compute_niche_payoff(state: WorldState, cfg: SimulationConfig) -> np.ndarray:
    """Organic niche payoff for genomes; diagnostic only, not a class quota."""
    from owl.core.advanced import action_entropy, ensure_advanced_fields

    ensure_advanced_fields(state, cfg)
    assert state.genome is not None
    payoff = np.zeros_like(state.genome, dtype=np.float32)
    if payoff.shape[-1] == 0:
        return payoff
    food = np.clip(state.food, 0.0, 1.0)
    scarcity = np.clip(1.0 - _local_food_mean(state), 0.0, 1.0)
    gy, gx = gradient_wrap(food)
    gradient = np.clip(np.sqrt(gy * gy + gx * gx), 0.0, 1.0)
    fragmentation = _upsampled_patch_attr(state, cfg, "prediction_error", default=0.0)
    entropy = action_entropy(state.possibility, cfg.actions.epsilon)
    starv = np.clip(
        state.starvation_debt if isinstance(state.starvation_debt, np.ndarray) else 0.0, 0.0, 1.0
    )
    rep_viability = reproduction_viability_field(state, cfg)
    channels = [
        food,
        scarcity * gradient,
        scarcity * (1.0 - starv),
        fragmentation * np.mean(np.clip(state.channel_trust_local, 0.0, 1.0), axis=-1),
        entropy * (np.clip(state.resource, 0.0, cfg.resources.max_resource) > 0.45),
        rep_viability,
        np.clip(1.0 - state.toxin, 0.0, 1.0) * np.clip(state.boundary, 0.0, 1.0),
        np.clip(state.curiosity, 0.0, 1.0) * entropy,
    ]
    for idx, ch in enumerate(channels[: payoff.shape[-1]]):
        payoff[..., idx] = np.clip(ch, 0.0, 1.0)
    return payoff


def compute_utilities(
    state: WorldState, parent_bias: np.ndarray, cfg: SimulationConfig
) -> np.ndarray:
    """Compute utilities with optional survival-optimal decision homeostasis."""
    from owl.core.advanced import ensure_advanced_fields

    ensure_advanced_fields(state, cfg)
    assert state.last_utilities is not None
    assert state.last_survival_value is not None
    assert state.last_decision_urgency is not None
    assert state.last_homeostatic_error is not None
    if not getattr(cfg.possibility, "advanced_enabled", False):
        out = _mvp_compute_utilities(state, parent_bias, cfg)
        state.last_utilities[...] = out
        return out.astype(np.float32, copy=False)

    _validate_parent_bias(state, parent_bias)
    drives = compute_internal_drives(state, cfg)
    pragmatic = compute_pragmatic_value(state, drives, cfg)
    epistemic = compute_epistemic_value(state, drives, cfg)
    risk = compute_action_risk(state, drives, cfg)
    effort = compute_action_effort(state, cfg)
    cooldown = state.action_cooldown if isinstance(state.action_cooldown, np.ndarray) else 0.0

    if getattr(cfg.decision_homeostasis, "enabled", False):
        survival, urgency, homeostatic_error = compute_survival_value(state, cfg)
        safe_epistemic = np.where(
            urgency[..., None] > float(cfg.decision_homeostasis.urgent_threshold),
            np.float32(cfg.decision_homeostasis.epistemic_weight_emergency),
            np.float32(cfg.decision_homeostasis.epistemic_weight_safe),
        )
        noetic_bonus = np.clip(
            getattr(state, "noetic_N", np.zeros(field_shape(state), dtype=np.float32)), 0.0, 1.0
        )
        utilities = (
            pragmatic
            + np.float32(cfg.decision_homeostasis.survival_weight) * survival
            + safe_epistemic * epistemic
            - np.float32(cfg.possibility.risk_weight) * risk
            - np.float32(cfg.possibility.effort_weight) * effort
            - np.float32(cfg.possibility.cooldown_weight) * cooldown
            + np.float32(cfg.decision_homeostasis.noetic_bias_scale)
            * noetic_bonus[..., None]
            * parent_bias
        )
        state.last_survival_value[...] = survival
        state.last_decision_urgency[...] = urgency
        state.last_homeostatic_error[...] = homeostatic_error
        # Crisis strongly suppresses reproduction but does not erase it under safe conditions.
        rep_viability = reproduction_viability_field(state, cfg)
        state.last_survival_value[..., int(Action.REPRODUCE)] = np.clip(rep_viability, 0.0, 1.0)
    else:
        utilities = (
            pragmatic
            + np.float32(cfg.possibility.epistemic_weight) * epistemic
            - np.float32(cfg.possibility.risk_weight) * risk
            - np.float32(cfg.possibility.effort_weight) * effort
            - np.float32(cfg.possibility.cooldown_weight) * cooldown
        )
        state.last_survival_value.fill(0.0)
        state.last_decision_urgency.fill(0.0)
        state.last_homeostatic_error.fill(0.0)

    # Niche payoffs mildly modulate matching actions/traits without hard-coded class survival.
    if getattr(cfg.reproduction, "advanced_enabled", False) and isinstance(
        state.genome, np.ndarray
    ):
        niche = compute_niche_payoff(state, cfg)
        # Motility, grazing, cooperation, curiosity, reproduction axes.
        if niche.shape[-1] > 0:
            for action in MOVE_DELTAS:
                utilities[..., int(action)] += 0.05 * niche[..., min(1, niche.shape[-1] - 1)]
        if niche.shape[-1] > 0:
            utilities[..., int(Action.FEED)] += 0.08 * niche[..., 0]
        if niche.shape[-1] > 3:
            utilities[..., int(Action.COMMUNICATE)] += 0.06 * niche[..., 3]
        if niche.shape[-1] > 5:
            utilities[..., int(Action.REPRODUCE)] += 0.06 * niche[..., 5]
        if niche.shape[-1] > 7:
            utilities[..., int(Action.SENSE)] += 0.05 * niche[..., 7]

    alive = (state.health > 0.0) & (~state.obstacle)
    utilities *= alive[..., None]
    utilities[state.obstacle, :] = 0.0
    if not np.all(np.isfinite(utilities)):
        raise ValueError("decision-homeostasis utilities contain non-finite values")
    state.last_utilities[...] = utilities.astype(np.float32, copy=False)
    return state.last_utilities.astype(np.float32, copy=True)
