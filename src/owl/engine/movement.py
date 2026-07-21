"""Movement proposal, boundary, and movement resolution.

Movement is the first embodied spatial action layer. It moves the cell-owned
observer-window arrays in ``WorldState`` while leaving environmental fields
(food, toxin, obstacle, persistent signal, and noise) fixed in world space.

The module is deliberately event-oriented for occupied targets: successful moves
mutate dense arrays; attempted moves into occupied cells enqueue sparse
``COLLISION`` events for ``owl.engine.collision``.
"""

from __future__ import annotations

from typing import cast

import numpy as np

from owl.core.actions import MOVE_DELTAS, REVERSE_MOVE_ACTION, Action, BoundaryMode, EventKind
from owl.core.config import SimulationConfig
from owl.core.constants import CELL_FIELDS_2D, CELL_FIELDS_3D
from owl.core.state import EventRecord, WorldState, field_shape
from owl.engine.events import enqueue_event


def _alive_mask(state: WorldState) -> np.ndarray:
    """Return living, non-obstacle cells."""
    return (state.health > 0.0) & (~state.obstacle)


def _movement_mask(state: WorldState) -> np.ndarray:
    """Return cells whose readout is a movement action and that are alive."""
    mask = np.zeros(field_shape(state), dtype=bool)
    for action in MOVE_DELTAS:
        mask |= state.readout == int(action)
    return cast(np.ndarray, mask & _alive_mask(state))


def _validate_position(
    position: tuple[int, int], shape: tuple[int, int], label: str
) -> tuple[int, int]:
    """Validate and normalize a grid coordinate."""
    y, x = map(int, position)
    height, width = shape
    if not (0 <= y < height and 0 <= x < width):
        raise ValueError(f"{label} position {(y, x)} is outside field shape {(height, width)}")
    return y, x


def _parent_id_for_target(state: WorldState, target: tuple[int, int]) -> int:
    """Infer spatial parent-patch id for ``target`` from current patch tiling."""
    y, x = target
    height, width = field_shape(state)
    patch_h, patch_w = state.patches.integration.shape
    if patch_h <= 0 or patch_w <= 0:
        return -1
    patch_size_y = height // patch_h
    patch_size_x = width // patch_w
    if patch_size_y <= 0 or patch_size_x <= 0:
        return -1
    return int((y // patch_size_y) * patch_w + (x // patch_size_x))


def wrap_position(y: int, x: int, height: int, width: int) -> tuple[int, int]:
    """Apply toroidal wrapping to a cell coordinate.

    Parameters
    ----------
    y, x:
        Candidate row/column coordinate.
    height, width:
        Positive grid dimensions.

    Returns
    -------
    tuple[int, int]
        Coordinate wrapped into ``[0, height) x [0, width)``.
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"height and width must be positive, got {(height, width)}")
    return int(y) % int(height), int(x) % int(width)


def propose_movements(state: WorldState, cfg: SimulationConfig) -> np.ndarray:
    """Generate target positions for movement readouts.

    Parameters
    ----------
    state:
        Current dense world state.
    cfg:
        Validated simulation configuration.

    Returns
    -------
    np.ndarray
        Integer array with shape ``(height, width, 2)``. ``[..., 0]`` is target
        row and ``[..., 1]`` is target column. Non-moving cells propose their
        current coordinate. Invalid off-grid proposals in non-toroidal modes are
        left as off-grid coordinates so validation can reject them explicitly.
    """
    del cfg
    height, width = field_shape(state)
    yy, xx = np.indices((height, width), dtype=np.int32)
    proposals = np.empty((height, width, 2), dtype=np.int32)
    proposals[..., 0] = yy
    proposals[..., 1] = xx

    for action, (dy, dx) in MOVE_DELTAS.items():
        mask = (state.readout == int(action)) & _alive_mask(state)
        proposals[..., 0][mask] = yy[mask] + int(dy)
        proposals[..., 1][mask] = xx[mask] + int(dx)

    return proposals


def validate_movement_targets(
    state: WorldState,
    proposals: np.ndarray,
    cfg: SimulationConfig,
) -> np.ndarray:
    """Validate movement proposals against boundary, obstacles, and occupancy.

    Parameters
    ----------
    state:
        Current dense world state.
    proposals:
        Integer array with shape ``(height, width, 2)`` returned by
        :func:`propose_movements`.
    cfg:
        Validated simulation configuration.

    Returns
    -------
    np.ndarray
        Boolean array with shape ``(height, width)``. ``True`` means the moving
        source cell may move into the proposed target at the time of validation.
        Occupied targets are invalid here and will be routed to collision
        handling in :func:`apply_movement`.
    """
    height, width = field_shape(state)
    proposals = np.asarray(proposals)
    if proposals.shape != (height, width, 2):
        raise ValueError(f"proposals must have shape {(height, width, 2)}, got {proposals.shape}")

    movers = _movement_mask(state)
    target_y = proposals[..., 0].astype(np.int64, copy=False)
    target_x = proposals[..., 1].astype(np.int64, copy=False)

    valid = movers.copy()
    mode = BoundaryMode(cfg.world.boundary_mode)

    if mode == BoundaryMode.TOROIDAL:
        target_y = target_y % height
        target_x = target_x % width
    else:
        in_bounds = (target_y >= 0) & (target_y < height) & (target_x >= 0) & (target_x < width)
        valid &= in_bounds
        # Clip only for safe indexing below; out-of-bounds entries stay invalid.
        target_y = np.clip(target_y, 0, height - 1)
        target_x = np.clip(target_x, 0, width - 1)

    target_obstacle = state.obstacle[target_y, target_x]
    target_occupied = (state.occupancy[target_y, target_x] >= 0) | (
        state.health[target_y, target_x] > 0.0
    )

    # A self-target is not a real movement; all MOVE_DELTAS are nonzero, but this
    # guard prevents future accidental zero-delta movement from being accepted.
    yy, xx = np.indices((height, width), dtype=np.int64)
    self_target = (target_y == yy) & (target_x == xx)

    valid &= ~target_obstacle
    valid &= ~target_occupied
    valid &= ~self_target
    return valid.astype(bool, copy=False)


def _base_move_cell_state(
    state: WorldState, source: tuple[int, int], target: tuple[int, int]
) -> None:
    """Move all cell-owned arrays from ``source`` to ``target``.

    Mutates all fields listed in ``CELL_FIELDS_2D`` and ``CELL_FIELDS_3D``, plus
    readout and occupancy. Environment fields are not moved. The source is
    cleared to an inert/dead state with a one-hot REST possibility vector.
    """
    shape = field_shape(state)
    sy, sx = _validate_position(source, shape, "source")
    ty, tx = _validate_position(target, shape, "target")
    if (sy, sx) == (ty, tx):
        return
    if state.obstacle[ty, tx]:
        raise ValueError(f"cannot move into obstacle target {(ty, tx)}")

    for name in CELL_FIELDS_2D:
        arr = getattr(state, name)
        if arr.shape != shape:
            raise ValueError(f"state.{name} must have shape {shape}, got {arr.shape}")
        arr[ty, tx] = arr[sy, sx]
        arr[sy, sx] = 0

    for name in CELL_FIELDS_3D:
        arr = getattr(state, name)
        if arr.shape[:2] != shape:
            raise ValueError(f"state.{name} must begin with shape {shape}, got {arr.shape}")
        arr[ty, tx, :] = arr[sy, sx, :]
        arr[sy, sx, :] = 0

    state.readout[ty, tx] = state.readout[sy, sx]
    state.readout[sy, sx] = int(Action.REST)

    source_occupancy = int(state.occupancy[sy, sx])
    if source_occupancy < 0:
        source_occupancy = int(ty * shape[1] + tx)
    state.occupancy[ty, tx] = source_occupancy
    state.occupancy[sy, sx] = -1

    state.parent_id[ty, tx] = _parent_id_for_target(state, (ty, tx))
    state.parent_id[sy, sx] = -1
    state.lineage_id[sy, sx] = -1
    state.age[sy, sx] = 0

    # Preserve probability simplex invariant for empty/dead source cells.
    state.possibility[sy, sx, :] = 0.0
    state.possibility[sy, sx, int(Action.REST)] = 1.0


def _normalized_target(
    y: int,
    x: int,
    height: int,
    width: int,
    cfg: SimulationConfig,
) -> tuple[int, int, bool]:
    """Return safe target coordinate and in-bounds status for boundary mode."""
    if BoundaryMode(cfg.world.boundary_mode) == BoundaryMode.TOROIDAL:
        ty, tx = wrap_position(y, x, height, width)
        return ty, tx, True
    in_bounds = 0 <= y < height and 0 <= x < width
    return int(np.clip(y, 0, height - 1)), int(np.clip(x, 0, width - 1)), bool(in_bounds)


def _legacy_apply_movement(
    state: WorldState, cfg: SimulationConfig, rng: np.random.Generator
) -> None:
    """Apply movement readouts and enqueue collisions for occupied targets.

    Mutates dense cell-owned arrays for successful moves, subtracts movement
    costs from moving cells, and appends ``COLLISION`` events for occupied
    targets. Failed boundary/obstacle moves pay a half movement cost.
    """
    height, width = field_shape(state)
    proposals = propose_movements(state, cfg)
    valid_initial = validate_movement_targets(state, proposals, cfg)
    movers = np.column_stack(np.nonzero(_movement_mask(state))).astype(np.int64, copy=False)
    if movers.size == 0:
        return

    order = rng.permutation(len(movers))
    occupied = (state.occupancy >= 0) | ((state.health > 0.0) & (~state.obstacle))

    for index in order:
        sy, sx = map(int, movers[index])
        if state.health[sy, sx] <= 0.0 or state.obstacle[sy, sx]:
            continue

        raw_ty = int(proposals[sy, sx, 0])
        raw_tx = int(proposals[sy, sx, 1])
        ty, tx, in_bounds = _normalized_target(raw_ty, raw_tx, height, width, cfg)

        if not in_bounds or state.obstacle[ty, tx]:
            state.resource[sy, sx] = max(
                0.0, float(state.resource[sy, sx]) - 0.5 * cfg.resources.movement_cost
            )
            continue

        if occupied[ty, tx]:
            if (ty, tx) != (sy, sx):
                enqueue_event(
                    state,
                    EventRecord(
                        kind=str(EventKind.COLLISION),
                        tick=int(state.tick),
                        source=(sy, sx),
                        target=(ty, tx),
                        payload={
                            "action": int(state.readout[sy, sx]),
                            "valid_initial": bool(valid_initial[sy, sx]),
                        },
                    ),
                )
            state.resource[sy, sx] = max(
                0.0, float(state.resource[sy, sx]) - 0.5 * cfg.resources.movement_cost
            )
            continue

        if not valid_initial[sy, sx]:
            state.resource[sy, sx] = max(
                0.0, float(state.resource[sy, sx]) - 0.5 * cfg.resources.movement_cost
            )
            continue

        move_cell_state(state, (sy, sx), (ty, tx))
        state.resource[ty, tx] = max(
            0.0, float(state.resource[ty, tx]) - cfg.resources.movement_cost
        )

        occupied[sy, sx] = False
        occupied[ty, tx] = True

    np.clip(state.resource, 0.0, cfg.resources.max_resource, out=state.resource)


# --- Advanced build overrides ------------------------------------------------
_mvp_move_cell_state = _base_move_cell_state


def move_cell_state(state: WorldState, source: tuple[int, int], target: tuple[int, int]) -> None:
    """Move cell state including optional advanced arrays and movement memory."""
    sy, sx = source
    ty, tx = target
    previous_last = None
    previous_loop = 0.0
    if (
        isinstance(state.last_movement_action, np.ndarray)
        and state.last_movement_action.shape == state.health.shape
    ):
        previous_last = int(state.last_movement_action[sy, sx])
    if (
        isinstance(state.movement_loop_score, np.ndarray)
        and state.movement_loop_score.shape == state.health.shape
    ):
        previous_loop = float(state.movement_loop_score[sy, sx])
    current_action = int(state.readout[sy, sx])

    _mvp_move_cell_state(state, source, target)

    from owl.core.advanced import move_advanced_cell_fields

    move_advanced_cell_fields(state, source, target)

    if (
        isinstance(state.last_movement_action, np.ndarray)
        and state.last_movement_action.shape == state.health.shape
    ):
        state.last_movement_action[ty, tx] = current_action
        state.last_movement_action[sy, sx] = int(Action.REST)

    if (
        isinstance(state.movement_loop_score, np.ndarray)
        and state.movement_loop_score.shape == state.health.shape
    ):
        try:
            reverse = REVERSE_MOVE_ACTION[Action(current_action)]
            was_reverse = 1.0 if previous_last == int(reverse) else 0.0
        except Exception:
            was_reverse = 0.0
        state.movement_loop_score[ty, tx] = np.clip(
            0.90 * previous_loop + 0.10 * was_reverse, 0.0, 1.0
        )
        state.movement_loop_score[sy, sx] = 0.0


# --- Deterministic simultaneous movement handling --------------------------
def apply_movement(state: WorldState, cfg: SimulationConfig, rng: np.random.Generator) -> None:
    """Apply graph/distribution-stable target-owner movement semantics.

    All proposals observe the same pre-mutation occupancy. Empty-target conflicts
    are resolved by the versioned counter RNG. Failed/blocked proposals pay half
    cost; accepted proposals pay full cost after simultaneous scatter.
    """
    del rng
    from owl.science.action_contract import movement_plan

    execution_readout = (
        state.compiled_execution_action
        if bool(cfg.action_transitions.enabled)
        and isinstance(state.compiled_execution_action, np.ndarray)
        else state.readout
    )
    plan = movement_plan(
        execution_readout,
        state.health,
        state.obstacle,
        state.occupancy,
        boundary_mode=str(cfg.world.boundary_mode),
        seed=int(cfg.world.seed),
        tick=int(state.tick),
        xp=np,
    )
    if plan.mover_y.size == 0:
        return
    sy, sx, ty, tx = plan.mover_y, plan.mover_x, plan.target_y, plan.target_x
    failed = ~plan.accepted
    if np.any(failed):
        state.resource[sy[failed], sx[failed]] -= np.float32(0.5 * cfg.resources.movement_cost)
    # Preserve collision events for the collision stage.
    for i in np.nonzero(plan.collision)[0]:
        enqueue_event(
            state,
            EventRecord(
                kind=str(EventKind.COLLISION),
                tick=int(state.tick),
                source=(int(sy[i]), int(sx[i])),
                target=(int(ty[i]), int(tx[i])),
                payload={"simultaneous": True, "priority": int(plan.priority[i])},
            ),
        )
    keep = np.nonzero(plan.accepted)[0]
    if keep.size:
        ay, ax, by, bx = sy[keep], sx[keep], ty[keep], tx[keep]
        # Snapshot all source values before any mutation.
        base2 = list(CELL_FIELDS_2D)
        base3 = list(CELL_FIELDS_3D)
        advanced2 = (
            "raqic_readout",
            "raqic_record_action",
            "raqic_record_readout",
            "raqic_record_confidence",
            "raqic_audit_flags",
            "raqic_trace_error",
            "raqic_min_eigenvalue",
            "raqic_backend_code",
            "raqic_legacy_shadow_readout",
            "raqic_compare_l1",
            "raqic_compare_kl",
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
            "active_sense_food_memory",
            "active_sense_toxin_memory",
            "active_sense_alive_memory",
            "active_sense_ttl",
            "active_sense_new_cell_count",
            "active_sense_new_target_count",
            "flee_compiled_action",
            "pursue_compiled_action",
            "compiled_execution_action",
        )
        advancedn = (
            "raqic_probabilities",
            "raqic_score",
            "raqic_phase",
            "raqic_parent_intention",
            "raqic_legacy_shadow_possibility",
            "raqic_debug_density_diag",
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
            "neighbor_trust",
            "same_scale_weight",
            "genome",
            "action_target_y",
            "action_target_x",
            "action_target_ow_id",
            "action_target_kind",
            "action_target_source",
            "action_target_distance",
            "action_target_confidence",
            "action_direction_y",
            "action_direction_x",
            "action_direction_executable",
            "action_direction_score",
            "action_direction_distance_delta",
            "action_direction_hazard",
            "action_direction_opportunity",
        )
        for name in (*base2, *advanced2):
            arr = getattr(state, name, None)
            if isinstance(arr, np.ndarray) and arr.shape == state.health.shape:
                vals = arr[ay, ax].copy()
                arr[ay, ax] = 0
                arr[by, bx] = vals
        for name in (*base3, *advancedn):
            arr = getattr(state, name, None)
            if isinstance(arr, np.ndarray) and arr.shape[:2] == state.health.shape:
                vals = arr[ay, ax, ...].copy()
                arr[ay, ax, ...] = 0
                arr[by, bx, ...] = vals
        old_occ = state.occupancy[ay, ax].copy()
        old_read = state.readout[ay, ax].copy()
        state.occupancy[ay, ax] = -1
        state.occupancy[by, bx] = np.where(old_occ >= 0, old_occ, by * state.health.shape[1] + bx)
        state.readout[ay, ax] = int(Action.REST)
        state.readout[by, bx] = old_read
        for name in (
            "raqic_readout",
            "raqic_record_action",
            "raqic_record_readout",
            "raqic_legacy_shadow_readout",
        ):
            arr = getattr(state, name, None)
            if isinstance(arr, np.ndarray) and arr.shape == state.health.shape:
                arr[ay, ax] = int(Action.REST)
        for name in ("raqic_probabilities", "raqic_legacy_shadow_possibility"):
            arr = getattr(state, name, None)
            if isinstance(arr, np.ndarray) and arr.shape[:2] == state.health.shape:
                arr[ay, ax, :] = 0.0
                arr[ay, ax, int(Action.REST)] = 1.0
        state.parent_id[ay, ax] = -1
        state.lineage_id[ay, ax] = -1
        state.age[ay, ax] = 0
        ph, pw = state.patches.integration.shape
        psy = state.health.shape[0] // ph
        psx = state.health.shape[1] // pw
        state.parent_id[by, bx] = (by // psy) * pw + (bx // psx)
        state.possibility[ay, ax, :] = 0.0
        state.possibility[ay, ax, int(Action.REST)] = 1.0
        if isinstance(state.last_movement_action, np.ndarray):
            state.last_movement_action[ay, ax] = int(Action.REST)
            state.last_movement_action[by, bx] = old_read
        state.resource[by, bx] -= np.float32(cfg.resources.movement_cost)
    np.clip(state.resource, 0.0, cfg.resources.max_resource, out=state.resource)
