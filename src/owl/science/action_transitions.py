"""Shared NumPy/CuPy contracts for SENSE, FLEE, and PURSUE transitions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

from owl.core.actions import MOVE_DELTAS, Action

ACTION_TRANSITION_CONTRACT_VERSION = "owl.action-transitions.v1"
FLEE_FAMILY = 0
PURSUE_FAMILY = 1
DIRECTION_ACTIONS: tuple[Action, ...] = tuple(MOVE_DELTAS)
DIRECTION_DELTAS: tuple[tuple[int, int], ...] = tuple(
    MOVE_DELTAS[action] for action in DIRECTION_ACTIONS
)


class ActionTargetKind(IntEnum):
    NONE = 0
    PERCEIVED_HAZARD_CELL = 1
    PERCEIVED_LIVING_OW = 2


class ActionTargetSource(IntEnum):
    NONE = 0
    SENSED_CURRENT = 1
    ACTIVE_SENSE_MEMORY = 2


@dataclass(frozen=True)
class ActionTransitionContext:
    """Agent-visible target and eight-direction compilation evidence."""

    target_y: Any
    target_x: Any
    target_ow_id: Any
    target_kind: Any
    target_source: Any
    target_distance: Any
    target_confidence: Any
    direction_y: Any
    direction_x: Any
    direction_executable: Any
    direction_score: Any
    direction_distance_delta: Any
    direction_hazard: Any
    direction_opportunity: Any
    flee_compiled_action: Any
    pursue_compiled_action: Any
    flee_executable: Any
    pursue_executable: Any


def _resolved_coordinates(
    y: Any,
    x: Any,
    *,
    dy: int,
    dx: int,
    height: int,
    width: int,
    boundary_mode: str,
    xp: Any,
) -> tuple[Any, Any, Any]:
    target_y = y + int(dy)
    target_x = x + int(dx)
    if str(boundary_mode) == "toroidal":
        return target_y % height, target_x % width, xp.ones_like(y, dtype=bool)
    valid = (
        (target_y >= 0)
        & (target_y < height)
        & (target_x >= 0)
        & (target_x < width)
    )
    return (
        xp.clip(target_y, 0, height - 1),
        xp.clip(target_x, 0, width - 1),
        valid,
    )


def summarize_radius_fields(
    *,
    health: Any,
    obstacle: Any,
    food: Any,
    toxin: Any,
    radius: int,
    threat_threshold: float,
    boundary_mode: str,
    xp: Any,
) -> tuple[Any, ...]:
    """Return backend-native bounded-radius summaries without offset loops."""
    h, w = map(int, health.shape)
    y, x = xp.indices((h, w), dtype=xp.int32)
    axis = xp.arange(-int(radius), int(radius) + 1, dtype=xp.int32)
    offset_y = xp.repeat(axis, axis.size)
    offset_x = xp.tile(axis, axis.size)
    raw_y = y[..., None] + offset_y
    raw_x = x[..., None] + offset_x
    if str(boundary_mode) == "toroidal":
        local_y = raw_y % h
        local_x = raw_x % w
        in_bounds = xp.ones_like(local_y, dtype=bool)
    else:
        in_bounds = (
            (raw_y >= 0) & (raw_y < h) & (raw_x >= 0) & (raw_x < w)
        )
        local_y = xp.clip(raw_y, 0, h - 1)
        local_x = xp.clip(raw_x, 0, w - 1)
    visible = in_bounds & (~obstacle[local_y, local_x])
    count = xp.sum(visible, axis=-1, dtype=xp.int32)
    living = visible & (health[local_y, local_x] > 0.0)
    target = living | (
        visible & (toxin[local_y, local_x] >= float(threat_threshold))
    )
    denominator = xp.maximum(count, 1).astype(health.dtype)
    return (
        xp.sum(xp.where(visible, food[local_y, local_x], 0.0), axis=-1)
        / denominator,
        xp.sum(xp.where(visible, toxin[local_y, local_x], 0.0), axis=-1)
        / denominator,
        xp.sum(living.astype(health.dtype), axis=-1) / denominator,
        count,
        xp.sum(target, axis=-1, dtype=xp.int32),
    )


def _chebyshev_distance(
    source_y: Any,
    source_x: Any,
    target_y: Any,
    target_x: Any,
    *,
    height: int,
    width: int,
    boundary_mode: str,
    xp: Any,
) -> Any:
    dy = xp.abs(source_y - target_y)
    dx = xp.abs(source_x - target_x)
    if str(boundary_mode) == "toroidal":
        dy = xp.minimum(dy, height - dy)
        dx = xp.minimum(dx, width - dx)
    return xp.maximum(dy, dx)


def _select_slot(mask: Any, primary: Any, distance: Any, *, maximize: bool, xp: Any) -> Any:
    """Lexicographic primary/distance/slot choice with stable first-index ties."""
    if maximize:
        optimum = xp.max(xp.where(mask, primary, -xp.inf), axis=-1, keepdims=True)
        tied = mask & (primary == optimum)
    else:
        optimum = xp.min(xp.where(mask, primary, xp.inf), axis=-1, keepdims=True)
        tied = mask & (primary == optimum)
    nearest = xp.min(xp.where(tied, distance, xp.inf), axis=-1, keepdims=True)
    chosen = tied & (distance == nearest)
    return xp.argmax(chosen, axis=-1).astype(xp.int32)


def resolve_action_transition_context(
    *,
    health: Any,
    resource: Any,
    obstacle: Any,
    occupancy: Any,
    food: Any,
    toxin: Any,
    predation: Any,
    aggression: Any,
    mobility: Any,
    cfg: Any,
    xp: Any,
) -> ActionTransitionContext:
    """Resolve targets and directions using only the declared bounded sensor.

    The v1 sensor exposes local physical occupancy, toxin, food, and visible
    threat phenotype inside ``target_sense_radius``. It never queries future
    state or counterfactual outcomes. Oracle evaluation remains a separate
    recorder perspective.
    """
    transition = cfg.action_transitions
    if not bool(transition.enabled):
        raise ValueError("action-transition resolver requires the explicit v1 contract")
    h, w = map(int, health.shape)
    y, x = xp.indices((h, w), dtype=xp.int32)
    radius = int(transition.target_sense_radius)
    axis = xp.arange(-radius, radius + 1, dtype=xp.int32)
    offset_y = xp.repeat(axis, axis.size)
    offset_x = xp.tile(axis, axis.size)
    nonself = (offset_y != 0) | (offset_x != 0)
    offset_y = offset_y[nonself]
    offset_x = offset_x[nonself]
    raw_y = y[..., None] + offset_y
    raw_x = x[..., None] + offset_x
    if str(cfg.world.boundary_mode) == "toroidal":
        local_y = raw_y % h
        local_x = raw_x % w
        valid = xp.ones_like(local_y, dtype=bool)
    else:
        valid = (
            (raw_y >= 0) & (raw_y < h) & (raw_x >= 0) & (raw_x < w)
        )
        local_y = xp.clip(raw_y, 0, h - 1)
        local_x = xp.clip(raw_x, 0, w - 1)
    local_alive = valid & (health[local_y, local_x] > 0.0) & (~obstacle[local_y, local_x])
    local_occupancy = occupancy[local_y, local_x]
    local_toxin = xp.clip(toxin[local_y, local_x], 0.0, 1.0)
    local_trait_threat = xp.clip(
        predation[local_y, local_x] + aggression[local_y, local_x], 0.0, 1.0
    )
    perceived_threat = xp.maximum(local_toxin, xp.where(local_alive, local_trait_threat, 0.0))
    local_distance = _chebyshev_distance(
        y[..., None],
        x[..., None],
        local_y,
        local_x,
        height=h,
        width=w,
        boundary_mode=str(cfg.world.boundary_mode),
        xp=xp,
    ).astype(health.dtype)
    alive = (health > 0.0) & (~obstacle)
    flee_targets = valid & (perceived_threat >= float(transition.perceived_threat_threshold))
    pursue_capable = xp.clip(predation + aggression, 0.0, 1.0) >= float(
        transition.pursuit_trait_threshold
    )
    pursue_targets = local_alive & pursue_capable[..., None]
    flee_slot = _select_slot(
        flee_targets, perceived_threat, local_distance, maximize=True, xp=xp
    )
    pursue_slot = _select_slot(
        pursue_targets, local_distance, local_distance, maximize=False, xp=xp
    )
    has_flee_target = alive & xp.any(flee_targets, axis=-1)
    has_pursue_target = alive & xp.any(pursue_targets, axis=-1)
    family_slots = xp.stack((flee_slot, pursue_slot), axis=-1)
    gather = family_slots[..., None]
    target_y = xp.take_along_axis(
        xp.broadcast_to(local_y[..., None, :], (*local_y.shape[:2], 2, local_y.shape[-1])),
        gather,
        axis=-1,
    )[..., 0]
    target_x = xp.take_along_axis(
        xp.broadcast_to(local_x[..., None, :], (*local_x.shape[:2], 2, local_x.shape[-1])),
        gather,
        axis=-1,
    )[..., 0]
    target_distance = xp.take_along_axis(
        xp.broadcast_to(
            local_distance[..., None, :],
            (*local_distance.shape[:2], 2, local_distance.shape[-1]),
        ),
        gather,
        axis=-1,
    )[..., 0]
    flee_confidence = xp.take_along_axis(perceived_threat, flee_slot[..., None], axis=-1)[..., 0]
    pursue_confidence = xp.take_along_axis(
        xp.where(pursue_targets, 1.0 / xp.maximum(local_distance, 1.0), 0.0),
        pursue_slot[..., None],
        axis=-1,
    )[..., 0]
    target_confidence = xp.stack((flee_confidence, pursue_confidence), axis=-1).astype(
        health.dtype
    )
    target_ow_id = xp.stack(
        (
            xp.full((h, w), -1, dtype=xp.int64),
            xp.take_along_axis(local_occupancy, pursue_slot[..., None], axis=-1)[..., 0],
        ),
        axis=-1,
    )
    exists = xp.stack((has_flee_target, has_pursue_target), axis=-1)
    target_y = xp.where(exists, target_y, -1).astype(xp.int32)
    target_x = xp.where(exists, target_x, -1).astype(xp.int32)
    target_ow_id = xp.where(exists, target_ow_id, -1).astype(xp.int64)
    target_kind = xp.stack(
        (
            xp.where(has_flee_target, int(ActionTargetKind.PERCEIVED_HAZARD_CELL), 0),
            xp.where(has_pursue_target, int(ActionTargetKind.PERCEIVED_LIVING_OW), 0),
        ),
        axis=-1,
    ).astype(xp.int16)
    target_source = xp.where(
        exists, int(ActionTargetSource.SENSED_CURRENT), int(ActionTargetSource.NONE)
    ).astype(xp.int16)
    target_distance = xp.where(exists, target_distance, 0).astype(health.dtype)
    target_confidence = xp.where(exists, target_confidence, 0).astype(health.dtype)

    direction_deltas = xp.asarray(DIRECTION_DELTAS, dtype=xp.int32)
    raw_y = y[..., None] + direction_deltas[:, 0]
    raw_x = x[..., None] + direction_deltas[:, 1]
    if str(cfg.world.boundary_mode) == "toroidal":
        candidate_y = raw_y % h
        candidate_x = raw_x % w
        in_bounds = xp.ones_like(candidate_y, dtype=bool)
    else:
        in_bounds = (
            (raw_y >= 0) & (raw_y < h) & (raw_x >= 0) & (raw_x < w)
        )
        candidate_y = xp.clip(raw_y, 0, h - 1)
        candidate_x = xp.clip(raw_x, 0, w - 1)
    destination_free = (
        in_bounds
        & (~obstacle[candidate_y, candidate_x])
        & (health[candidate_y, candidate_x] <= 0.0)
        & (occupancy[candidate_y, candidate_x] < 0)
    )
    enough_resource = resource >= float(cfg.resources.movement_cost)
    mobile = mobility > 0.0
    base_executable = (
        alive[..., None]
        & enough_resource[..., None]
        & mobile[..., None]
        & destination_free
    )
    direction_executable = xp.stack(
        (
            base_executable & has_flee_target[..., None],
            base_executable & has_pursue_target[..., None],
        ),
        axis=2,
    )
    before = target_distance[..., None]
    after = _chebyshev_distance(
        candidate_y[..., None, :],
        candidate_x[..., None, :],
        target_y[..., :, None],
        target_x[..., :, None],
        height=h,
        width=w,
        boundary_mode=str(cfg.world.boundary_mode),
        xp=xp,
    ).astype(health.dtype)
    distance_delta = xp.stack(
        (after[..., FLEE_FAMILY, :] - before[..., FLEE_FAMILY, :],
         before[..., PURSUE_FAMILY, :] - after[..., PURSUE_FAMILY, :]),
        axis=2,
    )
    destination_hazard = xp.clip(toxin[candidate_y, candidate_x], 0.0, 1.0)
    destination_opportunity = xp.clip(food[candidate_y, candidate_x], 0.0, 1.0)
    contact = (after[..., PURSUE_FAMILY, :] <= 1).astype(health.dtype)
    hazard = xp.broadcast_to(destination_hazard[..., None, :], distance_delta.shape)
    opportunity = xp.stack((destination_opportunity, contact), axis=2)
    normalized_cost = float(cfg.resources.movement_cost) / max(
        float(cfg.resources.max_resource), float(cfg.actions.epsilon)
    )

    def score(weights: Any, family: int) -> Any:
        return (
            float(weights.distance) * distance_delta[..., family, :]
            - float(weights.hazard) * hazard[..., family, :]
            - float(weights.cost) * normalized_cost
            + float(weights.opportunity) * opportunity[..., family, :]
        )

    direction_score = xp.stack(
        (
            score(transition.flee_score_weights, FLEE_FAMILY),
            score(transition.pursue_score_weights, PURSUE_FAMILY),
        ),
        axis=2,
    ).astype(health.dtype)
    direction_score = xp.where(direction_executable, direction_score, -xp.inf)
    best_direction = xp.argmax(direction_score, axis=-1).astype(xp.int32)
    has_direction = xp.any(direction_executable, axis=-1)
    action_lut = xp.asarray(tuple(int(action) for action in DIRECTION_ACTIONS), dtype=xp.int16)
    compiled = action_lut[best_direction]
    compiled = xp.where(has_direction, compiled, -1).astype(xp.int16)
    return ActionTransitionContext(
        target_y=target_y,
        target_x=target_x,
        target_ow_id=target_ow_id,
        target_kind=target_kind,
        target_source=target_source,
        target_distance=target_distance,
        target_confidence=target_confidence,
        direction_y=xp.broadcast_to(candidate_y[..., None, :], direction_executable.shape),
        direction_x=xp.broadcast_to(candidate_x[..., None, :], direction_executable.shape),
        direction_executable=direction_executable,
        direction_score=direction_score,
        direction_distance_delta=distance_delta,
        direction_hazard=hazard,
        direction_opportunity=opportunity,
        flee_compiled_action=compiled[..., FLEE_FAMILY],
        pursue_compiled_action=compiled[..., PURSUE_FAMILY],
        flee_executable=has_direction[..., FLEE_FAMILY],
        pursue_executable=has_direction[..., PURSUE_FAMILY],
    )


def compile_selected_execution_action(
    readout: Any, flee_compiled: Any, pursue_compiled: Any, *, xp: Any
) -> Any:
    """Preserve selected identity while returning the physical primitive."""
    compiled = readout.copy()
    compiled = xp.where(readout == int(Action.FLEE), flee_compiled, compiled)
    compiled = xp.where(readout == int(Action.PURSUE), pursue_compiled, compiled)
    return compiled.astype(readout.dtype, copy=False)
