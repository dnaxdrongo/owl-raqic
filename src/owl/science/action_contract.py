"""Backend-neutral deterministic action consequence contracts.

These kernels define simultaneous target-owner semantics so NumPy, CuPy,
CUDA-graph, and distributed paths share one scientific transition law.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from owl.core.actions import DIAGONAL_MOVES, MOVE_DELTAS, Action
from owl.record.cadc_schema import ReasonCode, TargetKind
from owl_raqic.random_contract import RNGStream, uniform_u64


@dataclass(frozen=True)
class MovementPlan:
    mover_y: Any
    mover_x: Any
    target_y: Any
    target_x: Any
    accepted: Any
    blocked: Any
    collision: Any
    priority: Any


@dataclass(frozen=True)
class CandidateTargetContext:
    """Pure pre-choice target and executability planes for the fixed action axis.

    These arrays are recorder evidence only.  They do not resolve simultaneous
    conflicts and never invoke a random stream.  Selected movement and
    reproduction results remain authoritative in ``movement_plan`` and
    ``reproduction_plan``.
    """

    target_kind: Any
    proposed_y: Any
    proposed_x: Any
    resolved_y: Any
    resolved_x: Any
    target_ow_id: Any
    destination_occupancy: Any
    destination_obstacle: Any
    destination_food: Any
    destination_toxin: Any
    opportunity_count: Any
    executable: Any
    reason_code: Any
    target_source: Any
    target_distance: Any
    target_confidence: Any
    compiled_action: Any


def candidate_target_context(
    health: Any,
    resource: Any,
    obstacle: Any,
    occupancy: Any,
    food: Any,
    toxin: Any,
    parent_id: Any,
    policy_legal: Any,
    *,
    boundary_mode: str,
    diagonal_movement_enabled: bool,
    xp: Any,
    action_transition_context: Any | None = None,
    action_transition_config: Any | None = None,
    movement_cost: float = 0.0,
) -> CandidateTargetContext:
    """Derive all 22 candidate targets without mutation or RNG consumption."""
    h, w = map(int, health.shape)
    actions = len(Action)
    expected = (h, w, actions)
    if tuple(policy_legal.shape) != expected:
        raise ValueError(f"policy_legal must have shape {expected}, got {policy_legal.shape}")

    y, x = xp.indices((h, w), dtype=xp.int32)
    source_y = xp.broadcast_to(y[..., None], expected)
    source_x = xp.broadcast_to(x[..., None], expected)
    proposed_y = source_y.copy()
    proposed_x = source_x.copy()
    resolved_y = source_y.copy()
    resolved_x = source_x.copy()
    target_kind = xp.full(expected, int(TargetKind.SELF), dtype=xp.int8)
    target_source = xp.zeros(expected, dtype=xp.int16)
    target_distance = xp.zeros(expected, dtype=health.dtype)
    target_confidence = xp.zeros(expected, dtype=health.dtype)
    compiled_action = xp.broadcast_to(
        xp.arange(actions, dtype=xp.int16)[None, None, :], expected
    ).copy()
    in_bounds = xp.ones(expected, dtype=bool)

    for action, (dy, dx) in MOVE_DELTAS.items():
        index = int(action)
        py = y + int(dy)
        px = x + int(dx)
        proposed_y[..., index] = py
        proposed_x[..., index] = px
        if str(boundary_mode) == "toroidal":
            resolved_y[..., index] = py % h
            resolved_x[..., index] = px % w
        else:
            valid = (py >= 0) & (py < h) & (px >= 0) & (px < w)
            in_bounds[..., index] = valid
            resolved_y[..., index] = xp.clip(py, 0, h - 1)
            resolved_x[..., index] = xp.clip(px, 0, w - 1)
        target_kind[..., index] = int(TargetKind.CELL)

    target_kind[..., int(Action.COMMUNICATE)] = int(TargetKind.LOCAL_BROADCAST)
    target_kind[..., int(Action.INHIBIT)] = int(TargetKind.OW_SET)
    target_kind[..., int(Action.REPRODUCE)] = int(TargetKind.EMPTY_NEIGHBOR_SET)
    target_kind[..., int(Action.INGEST)] = int(TargetKind.OW_SET)
    target_kind[..., int(Action.EXPEL)] = int(TargetKind.OW_SET)
    target_kind[..., int(Action.SPLIT)] = int(TargetKind.EMPTY_NEIGHBOR_SET)
    target_kind[..., int(Action.MERGE)] = int(TargetKind.OW_SET)
    target_kind[..., int(Action.FLEE)] = int(TargetKind.SEMANTIC_DIRECTION_SET)
    target_kind[..., int(Action.PURSUE)] = int(TargetKind.SEMANTIC_DIRECTION_SET)

    flat_target = (resolved_y.astype(xp.int64) * w + resolved_x.astype(xp.int64)).reshape(
        -1
    )
    destination_occupancy = occupancy.reshape(-1)[flat_target].reshape(expected)
    destination_obstacle = obstacle.reshape(-1)[flat_target].reshape(expected)
    destination_food = food.reshape(-1)[flat_target].reshape(expected)
    destination_toxin = toxin.reshape(-1)[flat_target].reshape(expected)
    target_ow_id = xp.where(destination_occupancy >= 0, destination_occupancy, -1).astype(
        xp.int64
    )
    set_valued_actions = (
        Action.COMMUNICATE,
        Action.INHIBIT,
        Action.REPRODUCE,
        Action.INGEST,
        Action.EXPEL,
        Action.SPLIT,
        Action.MERGE,
        Action.FLEE,
        Action.PURSUE,
    )
    for action in set_valued_actions:
        index = int(action)
        proposed_y[..., index] = -1
        proposed_x[..., index] = -1
        resolved_y[..., index] = -1
        resolved_x[..., index] = -1
        target_ow_id[..., index] = -1
        destination_occupancy[..., index] = -1
        destination_obstacle[..., index] = False
        destination_food[..., index] = 0
        destination_toxin[..., index] = 0

    alive = (health > 0.0) & (~obstacle)
    neighbor_living = xp.zeros((h, w), dtype=xp.int16)
    empty_neighbors = xp.zeros((h, w), dtype=xp.int16)
    for dy, dx in ((-1, 0), (1, 0), (0, 1), (0, -1), (-1, 1), (-1, -1), (1, 1), (1, -1)):
        cy = y + int(dy)
        cx = x + int(dx)
        if str(boundary_mode) == "toroidal":
            cy %= h
            cx %= w
            valid = xp.ones((h, w), dtype=bool)
        else:
            valid = (cy >= 0) & (cy < h) & (cx >= 0) & (cx < w)
            cy = xp.clip(cy, 0, h - 1)
            cx = xp.clip(cx, 0, w - 1)
        neighbor_living += (valid & (health[cy, cx] > 0.0) & (~obstacle[cy, cx])).astype(
            xp.int16
        )
        empty_neighbors += (
            valid
            & (~obstacle[cy, cx])
            & (health[cy, cx] <= 0.0)
            & (occupancy[cy, cx] < 0)
        ).astype(xp.int16)

    opportunity_count = xp.zeros(expected, dtype=xp.int16)
    for action in (Action.INHIBIT, Action.INGEST, Action.EXPEL, Action.MERGE, Action.PURSUE):
        opportunity_count[..., int(action)] = neighbor_living
    for action in (Action.REPRODUCE, Action.SPLIT):
        opportunity_count[..., int(action)] = empty_neighbors

    legal = policy_legal.astype(bool, copy=False)
    executable = legal & alive[..., None]
    reason = xp.where(
        alive[..., None],
        xp.where(legal, int(ReasonCode.NONE), int(ReasonCode.POLICY_ILLEGAL)),
        int(ReasonCode.NOT_ALIVE),
    ).astype(xp.int16)

    occupied = destination_obstacle | (destination_occupancy >= 0) | (
        health.reshape(-1)[flat_target].reshape(expected) > 0.0
    )
    for action in MOVE_DELTAS:
        index = int(action)
        allowed = (
            executable[..., index]
            & in_bounds[..., index]
            & (~destination_obstacle[..., index])
        )
        allowed &= ~occupied[..., index]
        executable[..., index] = allowed
        reason[..., index] = xp.where(
            ~alive,
            int(ReasonCode.NOT_ALIVE),
            xp.where(
                ~legal[..., index],
                int(ReasonCode.POLICY_ILLEGAL),
                xp.where(
                    ~in_bounds[..., index],
                    int(ReasonCode.BOUNDARY_BLOCKED),
                    xp.where(
                        destination_obstacle[..., index],
                        int(ReasonCode.OBSTACLE),
                        xp.where(occupied[..., index], int(ReasonCode.OCCUPIED), 0),
                    ),
                ),
            ),
        ).astype(xp.int16)
        if action in DIAGONAL_MOVES and not diagonal_movement_enabled:
            executable[..., index] = False
            reason[..., index] = xp.where(
                alive, int(ReasonCode.POLICY_ILLEGAL), int(ReasonCode.NOT_ALIVE)
            ).astype(xp.int16)

    for action in (Action.REPRODUCE, Action.SPLIT):
        index = int(action)
        has_target = empty_neighbors > 0
        executable[..., index] &= has_target
        reason[..., index] = xp.where(
            ~alive,
            int(ReasonCode.NOT_ALIVE),
            xp.where(
                ~legal[..., index],
                int(ReasonCode.POLICY_ILLEGAL),
                xp.where(has_target, 0, int(ReasonCode.NO_TARGET)),
            ),
        ).astype(xp.int16)

    ingest = int(Action.INGEST)
    executable[..., ingest] &= neighbor_living > 0
    reason[..., ingest] = xp.where(
        ~alive,
        int(ReasonCode.NOT_ALIVE),
        xp.where(
            ~legal[..., ingest],
            int(ReasonCode.POLICY_ILLEGAL),
            xp.where(neighbor_living > 0, 0, int(ReasonCode.NO_TARGET)),
        ),
    ).astype(xp.int16)

    split = int(Action.SPLIT)
    executable[..., split] &= resource > 0.05
    transitions_enabled = bool(
        action_transition_config is not None and action_transition_config.enabled
    )
    if not transitions_enabled:
        # Baseline actions may be scored without applying authoritative effects.
        for action in (Action.SENSE, Action.FLEE, Action.PURSUE):
            index = int(action)
            executable[..., index] = False
            reason[..., index] = xp.where(
                alive & legal[..., index],
                int(ReasonCode.NO_EXECUTION_CONTRACT),
                reason[..., index],
            ).astype(xp.int16)
    else:
        if action_transition_context is None:
            raise ValueError("v1 candidate evidence requires action-transition context")
        sense = int(Action.SENSE)
        sense_enabled = bool(action_transition_config.active_sense_enabled)
        can_sense = (
            alive
            & sense_enabled
            & (resource >= float(action_transition_config.active_sense_cost))
        )
        executable[..., sense] = legal[..., sense] & can_sense
        if sense_enabled:
            sense_reason = xp.where(
                resource < float(action_transition_config.active_sense_cost),
                int(ReasonCode.INSUFFICIENT_RESOURCE),
                int(ReasonCode.NONE),
            )
        else:
            sense_reason = xp.full_like(
                reason[..., sense], int(ReasonCode.DISABLED), dtype=xp.int16
            )
        reason[..., sense] = xp.where(
            ~alive,
            int(ReasonCode.NOT_ALIVE),
            xp.where(
                ~legal[..., sense],
                int(ReasonCode.POLICY_ILLEGAL),
                sense_reason,
            ),
        ).astype(xp.int16)
        families = (
            (
                Action.FLEE,
                0,
                bool(action_transition_config.flee_execution_enabled),
                action_transition_context.flee_executable,
            ),
            (
                Action.PURSUE,
                1,
                bool(action_transition_config.pursue_execution_enabled),
                action_transition_context.pursue_executable,
            ),
        )
        for action, family, family_enabled, family_executable in families:
            index = int(action)
            compiled = (
                action_transition_context.flee_compiled_action
                if action == Action.FLEE
                else action_transition_context.pursue_compiled_action
            )
            direction_actions = xp.asarray(
                tuple(int(action) for action in MOVE_DELTAS), dtype=xp.int16
            )
            direction_slot = xp.argmax(
                direction_actions[None, None, :] == compiled[..., None], axis=-1
            ).astype(xp.int32)
            gather = direction_slot[..., None]
            dy = xp.take_along_axis(
                action_transition_context.direction_y[..., family, :], gather, axis=-1
            )[..., 0]
            dx = xp.take_along_axis(
                action_transition_context.direction_x[..., family, :], gather, axis=-1
            )[..., 0]
            has_target = action_transition_context.target_kind[..., family] > 0
            allowed = (
                alive
                & legal[..., index]
                & family_enabled
                & family_executable
                & (compiled >= int(Action.MOVE_N))
            )
            executable[..., index] = allowed
            proposed_y[..., index] = xp.where(has_target, dy, -1)
            proposed_x[..., index] = xp.where(has_target, dx, -1)
            resolved_y[..., index] = xp.where(has_target, dy, -1)
            resolved_x[..., index] = xp.where(has_target, dx, -1)
            target_ow_id[..., index] = action_transition_context.target_ow_id[..., family]
            target_source[..., index] = action_transition_context.target_source[..., family]
            target_distance[..., index] = action_transition_context.target_distance[..., family]
            target_confidence[..., index] = action_transition_context.target_confidence[..., family]
            compiled_action[..., index] = compiled
            opportunity_count[..., index] = has_target.astype(xp.int16)
            safe_y = xp.maximum(dy, 0)
            safe_x = xp.maximum(dx, 0)
            destination_occupancy[..., index] = xp.where(
                has_target, occupancy[safe_y, safe_x], -1
            )
            destination_obstacle[..., index] = xp.where(
                has_target, obstacle[safe_y, safe_x], False
            )
            destination_food[..., index] = xp.where(has_target, food[safe_y, safe_x], 0)
            destination_toxin[..., index] = xp.where(has_target, toxin[safe_y, safe_x], 0)
            if family_enabled:
                family_reason = xp.where(
                    ~has_target,
                    int(ReasonCode.NO_TARGET),
                    xp.where(
                        resource < float(movement_cost),
                        int(ReasonCode.INSUFFICIENT_RESOURCE),
                        xp.where(
                            family_executable,
                            int(ReasonCode.NONE),
                            int(ReasonCode.NO_EXECUTABLE_DIRECTION),
                        ),
                    ),
                )
            else:
                family_reason = xp.full_like(
                    reason[..., index], int(ReasonCode.DISABLED), dtype=xp.int16
                )
            reason[..., index] = xp.where(
                ~alive,
                int(ReasonCode.NOT_ALIVE),
                xp.where(
                    ~legal[..., index],
                    int(ReasonCode.POLICY_ILLEGAL),
                    family_reason,
                ),
            ).astype(xp.int16)

    return CandidateTargetContext(
        target_kind=target_kind,
        proposed_y=proposed_y,
        proposed_x=proposed_x,
        resolved_y=resolved_y,
        resolved_x=resolved_x,
        target_ow_id=target_ow_id,
        destination_occupancy=destination_occupancy,
        destination_obstacle=destination_obstacle,
        destination_food=destination_food,
        destination_toxin=destination_toxin,
        opportunity_count=opportunity_count,
        executable=executable,
        reason_code=reason,
        target_source=target_source,
        target_distance=target_distance,
        target_confidence=target_confidence,
        compiled_action=compiled_action,
    )


def movement_plan(
    readout: Any,
    health: Any,
    obstacle: Any,
    occupancy: Any,
    *,
    boundary_mode: str,
    seed: int,
    tick: int,
    xp: Any,
) -> Any:
    h, w = health.shape
    actions = max(int(a) for a in Action) + 1
    dy = xp.zeros((actions,), dtype=xp.int32)
    dx = xp.zeros((actions,), dtype=xp.int32)
    move_lut = xp.zeros((actions,), dtype=bool)
    for action, (ay, ax) in MOVE_DELTAS.items():
        dy[int(action)] = ay
        dx[int(action)] = ax
        move_lut[int(action)] = True
    live = (health > 0.0) & (~obstacle)
    mover = live & move_lut[readout.astype(xp.int32)]
    sy, sx = xp.nonzero(mover)
    if int(sy.shape[0]) == 0:
        empty_i = xp.zeros((0,), dtype=xp.int32)
        empty_b = xp.zeros((0,), dtype=bool)
        empty_u = xp.zeros((0,), dtype=xp.uint64)
        return MovementPlan(empty_i, empty_i, empty_i, empty_i, empty_b, empty_b, empty_b, empty_u)
    ty = sy.astype(xp.int32) + dy[readout[sy, sx].astype(xp.int32)]
    tx = sx.astype(xp.int32) + dx[readout[sy, sx].astype(xp.int32)]
    if str(boundary_mode) == "toroidal":
        ty %= h
        tx %= w
        in_bounds = xp.ones_like(sy, dtype=bool)
    else:
        in_bounds = (ty >= 0) & (ty < h) & (tx >= 0) & (tx < w)
        ty = xp.clip(ty, 0, h - 1)
        tx = xp.clip(tx, 0, w - 1)
    pre_occupied = (occupancy[ty, tx] >= 0) | (health[ty, tx] > 0.0)
    blocked = (~in_bounds) | obstacle[ty, tx]
    collision = (~blocked) & pre_occupied
    candidate = (~blocked) & (~pre_occupied)
    ow = xp.where(
        occupancy[sy, sx] >= 0, occupancy[sy, sx], sy.astype(xp.int64) * w + sx.astype(xp.int64)
    ).astype(xp.uint64)
    priority = uniform_u64(seed, tick, ow, RNGStream.MOVEMENT_TIE, 0, xp=xp)
    target_flat = ty.astype(xp.int64) * w + tx.astype(xp.int64)
    source_flat = sy.astype(xp.int64) * w + sx.astype(xp.int64)
    accepted = xp.zeros_like(candidate, dtype=bool)
    idx = xp.nonzero(candidate)[0]
    if int(idx.shape[0]):
        # CuPy requires lexsort keys as one ndarray, not a Python tuple.
        # Cast all sort-only keys to uint64 so stacking does not promote the
        # uint64 random priority tie-breaker to float64 and lose precision.
        sort_source = source_flat[idx].astype(xp.uint64)
        sort_priority = priority[idx].astype(xp.uint64)
        sort_target = target_flat[idx].astype(xp.uint64)
        sort_keys = xp.stack((sort_source, sort_priority, sort_target), axis=0)
        order_local = xp.lexsort(sort_keys)
        ordered = idx[order_local]
        ordered_targets = target_flat[ordered]
        first = xp.ones_like(ordered_targets, dtype=bool)
        if int(ordered_targets.shape[0]) > 1:
            first[1:] = ordered_targets[1:] != ordered_targets[:-1]
        accepted[ordered[first]] = True
        # Losers collide with the target-owner winner after commit.
    collision |= candidate & (~accepted)
    return MovementPlan(
        sy.astype(xp.int32),
        sx.astype(xp.int32),
        ty.astype(xp.int32),
        tx.astype(xp.int32),
        accepted,
        blocked,
        collision,
        priority,
    )


@dataclass(frozen=True)
class ReproductionPlan:
    parent_y: Any
    parent_x: Any
    target_y: Any
    target_x: Any
    accepted: Any
    priority: Any
    parent_ow_id: Any
    gate: Any
    has_target: Any
    candidate: Any


def reproduction_plan(
    readout: Any,
    health: Any,
    resource: Any,
    boundary: Any,
    integration: Any,
    reproduction_rate: Any,
    obstacle: Any,
    occupancy: Any,
    *,
    min_resource: float,
    min_health: float,
    min_boundary: float,
    min_integration: float,
    boundary_mode: str,
    seed: int,
    tick: int,
    xp: Any,
) -> Any:
    """Return deterministic simultaneous parent/target proposals.

    The same pre-mutation state is observed by every parent.  A counter-RNG
    gate preserves the configured reproduction trait, a second counter selects
    among currently empty Moore neighbours, and target conflicts are resolved
    by a stable counter priority followed by source identity.
    """
    from owl_raqic.random_contract import uniform01

    h, w = health.shape
    viable = (
        (health > 0.0)
        & (~obstacle)
        & (readout == int(Action.REPRODUCE))
        & (resource >= float(min_resource))
        & (health >= float(min_health))
        & (boundary >= float(min_boundary))
        & (integration >= float(min_integration))
        & (reproduction_rate > 0.0)
    )
    py, px = xp.nonzero(viable)
    n = int(py.shape[0])
    if n == 0:
        empty_i = xp.zeros((0,), dtype=xp.int32)
        empty_b = xp.zeros((0,), dtype=bool)
        empty_u = xp.zeros((0,), dtype=xp.uint64)
        return ReproductionPlan(
            empty_i,
            empty_i,
            empty_i,
            empty_i,
            empty_b,
            empty_u,
            empty_u,
            empty_b,
            empty_b,
            empty_b,
        )

    py = py.astype(xp.int32)
    px = px.astype(xp.int32)
    fallback_id = py.astype(xp.int64) * int(w) + px.astype(xp.int64)
    parent_id = xp.where(occupancy[py, px] >= 0, occupancy[py, px], fallback_id).astype(xp.uint64)

    gate = uniform01(
        seed,
        tick,
        parent_id,
        RNGStream.REPRODUCTION_TIE,
        0,
        xp=xp,
        dtype=xp.float64,
    ) <= xp.clip(reproduction_rate[py, px].astype(xp.float64), 0.0, 1.0)

    neighbors = ((-1, 0), (1, 0), (0, 1), (0, -1), (-1, 1), (-1, -1), (1, 1), (1, -1))
    empty_matrix = xp.zeros((n, len(neighbors)), dtype=bool)
    ys = xp.zeros((n, len(neighbors)), dtype=xp.int32)
    xs = xp.zeros((n, len(neighbors)), dtype=xp.int32)
    for index, (dy, dx) in enumerate(neighbors):
        cy = py + int(dy)
        cx = px + int(dx)
        if str(boundary_mode) == "toroidal":
            cy %= int(h)
            cx %= int(w)
            in_bounds = xp.ones((n,), dtype=bool)
        else:
            in_bounds = (cy >= 0) & (cy < int(h)) & (cx >= 0) & (cx < int(w))
            cy = xp.clip(cy, 0, int(h) - 1)
            cx = xp.clip(cx, 0, int(w) - 1)
        ys[:, index] = cy
        xs[:, index] = cx
        empty_matrix[:, index] = (
            in_bounds & (~obstacle[cy, cx]) & (health[cy, cx] <= 0.0) & (occupancy[cy, cx] < 0)
        )

    counts = xp.sum(empty_matrix, axis=1).astype(xp.int32)
    selector_u = uniform01(
        seed,
        tick,
        parent_id,
        RNGStream.REPRODUCTION_TIE,
        1,
        xp=xp,
        dtype=xp.float64,
    )
    # Rank among the legal neighbours. floor(u * count) is always in range
    # because uniform01 is in [0, 1).
    desired_rank = xp.floor(selector_u * xp.maximum(counts, 1)).astype(xp.int32)
    legal_rank = xp.cumsum(empty_matrix.astype(xp.int32), axis=1) - 1
    chosen_mask = empty_matrix & (legal_rank == desired_rank[:, None])
    chosen_index = xp.argmax(chosen_mask, axis=1).astype(xp.int32)
    rows = xp.arange(n, dtype=xp.int32)
    ty = ys[rows, chosen_index]
    tx = xs[rows, chosen_index]
    candidate = gate & (counts > 0)

    priority = uniform_u64(seed, tick, parent_id, RNGStream.REPRODUCTION_TIE, 2, xp=xp)
    target_flat = ty.astype(xp.int64) * int(w) + tx.astype(xp.int64)
    source_flat = py.astype(xp.int64) * int(w) + px.astype(xp.int64)
    accepted = xp.zeros((n,), dtype=bool)
    idx = xp.nonzero(candidate)[0]
    if int(idx.shape[0]):
        # CuPy requires lexsort keys as one ndarray, not a Python tuple.
        # Cast all sort-only keys to uint64 so stacking does not promote the
        # uint64 random priority tie-breaker to float64 and lose precision.
        sort_source = source_flat[idx].astype(xp.uint64)
        sort_priority = priority[idx].astype(xp.uint64)
        sort_target = target_flat[idx].astype(xp.uint64)
        sort_keys = xp.stack((sort_source, sort_priority, sort_target), axis=0)
        order_local = xp.lexsort(sort_keys)
        ordered = idx[order_local]
        ordered_targets = target_flat[ordered]
        first = xp.ones_like(ordered_targets, dtype=bool)
        if int(ordered_targets.shape[0]) > 1:
            first[1:] = ordered_targets[1:] != ordered_targets[:-1]
        accepted[ordered[first]] = True

    return ReproductionPlan(
        py,
        px,
        ty,
        tx,
        accepted,
        priority,
        parent_id,
        gate,
        counts > 0,
        candidate,
    )
