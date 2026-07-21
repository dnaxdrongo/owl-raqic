"""Death, clearing, release, and residue handling.

Death is a physical/topological consequence: cell-owned arrays are cleared, a
small amount of residue is returned to the environment, and distress emission is
left in the communication substrate. Environment fields such as food, toxin,
noise, obstacle, and the persistent signal field are not cell-owned and are not
cleared by ``clear_cell``.
"""

from __future__ import annotations

import numpy as np

from owl.core.actions import Action, EventKind, SignalChannel
from owl.core.config import SimulationConfig
from owl.core.constants import CELL_FIELDS_2D, CELL_FIELDS_3D
from owl.core.state import EventRecord, WorldState, field_shape
from owl.engine.feeding import deposit_resource_residue


def _cell_present_mask(state: WorldState) -> np.ndarray:
    """Return positions that appear to contain a cell identity."""
    return (
        (state.occupancy >= 0)
        | (state.health > 0.0)
        | (state.resource > 0.0)
        | (state.boundary > 0.0)
        | (state.memory > 0.0)
        | (state.integration > 0.0)
    ) & (~state.obstacle)


def detect_dead_cells(state: WorldState, cfg: SimulationConfig) -> np.ndarray:
    """Return a boolean mask of cells that fail survival conditions.

    A position is considered for death only if it appears to contain a cell
    identity. Death is triggered by nonpositive health, resource, or boundary.
    Integration collapse is represented conservatively as a negative value,
    because low-but-positive integration can be a valid early-world condition.
    """
    shape = field_shape(state)
    for name in ("resource", "health", "boundary", "integration", "occupancy"):
        if getattr(state, name).shape != shape:
            raise ValueError(
                f"state.{name} must have shape {shape}, got {getattr(state, name).shape}"
            )

    present = _cell_present_mask(state)
    starvation_debt = getattr(state, "starvation_debt", None)
    if isinstance(starvation_debt, np.ndarray) and starvation_debt.shape == shape:
        starvation_failed = (starvation_debt >= 1.0) & (state.health <= 0.05)
    else:
        starvation_failed = np.zeros(shape, dtype=bool)

    # Zero resources begin the starvation process; death occurs only after the
    # accumulated debt and health thresholds are satisfied.
    dead = present & (
        (state.health <= 0.0)
        | starvation_failed
        | (state.boundary <= 0.0)
        | (state.integration < 0.0)
    )
    dead[state.obstacle] = False
    return dead.astype(bool, copy=False)


def release_internal_ows(
    state: WorldState, position: tuple[int, int], cfg: SimulationConfig
) -> None:
    """Record release events for nested child observer windows after parent death.

    The early baseline has no active nested ``OWRecord`` containment dynamics. This
    function therefore only records a sparse RELEASE event when a mobile OW at
    the position has listed children, leaving actual child placement to a later
    topology pass.
    """
    del cfg
    y, x = position
    for record in list(state.mobile_ows.values()):
        if record.alive and record.pos_y == y and record.pos_x == x and record.children:
            state.event_queue.append(
                EventRecord(
                    kind=str(EventKind.RELEASE),
                    tick=int(state.tick),
                    source=(int(y), int(x)),
                    payload={"children": list(record.children), "parent_id": int(record.id)},
                )
            )


def _base_clear_cell(state: WorldState, position: tuple[int, int]) -> None:
    """Clear all cell-owned arrays at ``position``.

    Mutates physical, possibility, communication-trait, identity, lineage, and
    readout fields. Environment fields ``food``, ``toxin``, ``noise``,
    ``obstacle``, and persistent ``signal`` are not cleared.
    """
    y, x = map(int, position)
    h, w = field_shape(state)
    if not (0 <= y < h and 0 <= x < w):
        raise ValueError(f"position {(y, x)} is outside field shape {(h, w)}")

    for name in CELL_FIELDS_2D:
        arr = getattr(state, name)
        if arr.shape != (h, w):
            raise ValueError(f"state.{name} must have shape {(h, w)}, got {arr.shape}")
        arr[y, x] = 0

    for name in CELL_FIELDS_3D:
        arr = getattr(state, name)
        if arr.shape[:2] != (h, w):
            raise ValueError(f"state.{name} must begin with shape {(h, w)}, got {arr.shape}")
        arr[y, x, :] = 0

    state.signal_reception[y, x, :] = 0
    state.signal_emission[y, x, :] = 0
    state.readout[y, x] = int(Action.REST)
    state.occupancy[y, x] = -1
    state.parent_id[y, x] = -1
    state.lineage_id[y, x] = -1
    state.age[y, x] = 0

    # Preserve the probability simplex invariant even for dead cells by making
    # their possibility distribution one-hot REST.
    state.possibility[y, x, :] = 0
    state.possibility[y, x, int(Action.REST)] = 1.0


def _base_apply_death(state: WorldState, cfg: SimulationConfig) -> None:
    """Clear dead cells and return residue to environment/communication fields.

    Mutates
    -------
    state.food:
        Receives a small residue deposit from dead cells.
    state.signal_emission:
        Receives a DISTRESS pulse at dead-cell positions when the channel exists.
    cell-owned arrays:
        Cleared through :func:`clear_cell`.
    """
    dead = detect_dead_cells(state, cfg)
    if not np.any(dead):
        return

    positions = np.column_stack(np.nonzero(dead)).astype(np.int64, copy=False)
    residue = (
        0.20
        * np.clip(state.resource, 0.0, cfg.resources.max_resource)
        / max(float(cfg.resources.max_resource), cfg.actions.epsilon)
        + 0.05 * np.clip(state.boundary, 0.0, 1.0)
    ).astype(np.float32, copy=False)
    deposit_resource_residue(state, residue, positions)

    distress_idx = int(SignalChannel.DISTRESS)
    if cfg.communication.enabled and distress_idx < state.signal_emission.shape[-1]:
        state.signal_emission[dead, distress_idx] += 0.10
        np.clip(
            state.signal_emission[..., distress_idx],
            0.0,
            1.0,
            out=state.signal_emission[..., distress_idx],
        )

    for y, x in positions:
        release_internal_ows(state, (int(y), int(x)), cfg)
        clear_cell(state, (int(y), int(x)))


# --- Advanced build overrides ------------------------------------------------
_mvp_clear_cell = _base_clear_cell
_mvp_apply_death = _base_apply_death


def _advanced_clear_cell(state: WorldState, position: tuple[int, int]) -> None:
    """Clear cell including optional advanced-owned arrays."""
    _mvp_clear_cell(state, position)
    y, x = position
    for name in (
        "digestion",
        "age_stress",
        "last_intake",
        "development_stage",
        "symbiosis",
        "prediction_error",
        "starvation_debt",
        "movement_loop_score",
    ):
        arr = getattr(state, name, None)
        if isinstance(arr, np.ndarray) and arr.shape == state.health.shape:
            arr[y, x] = 0
    for name in (
        "action_cooldown",
        "last_utilities",
        "last_logits",
        "last_action_probabilities",
        "deception_memory",
        "source_confidence",
        "genome",
    ):
        arr = getattr(state, name, None)
        if isinstance(arr, np.ndarray) and arr.shape[:2] == state.health.shape:
            arr[y, x, ...] = 0
    for name in (
        "active_sense_food_memory",
        "active_sense_toxin_memory",
        "active_sense_alive_memory",
        "active_sense_ttl",
        "active_sense_new_cell_count",
        "active_sense_new_target_count",
        "action_target_distance",
        "action_target_confidence",
        "action_direction_executable",
        "action_direction_score",
        "action_direction_distance_delta",
        "action_direction_hazard",
        "action_direction_opportunity",
    ):
        arr = getattr(state, name, None)
        if isinstance(arr, np.ndarray) and arr.shape[:2] == state.health.shape:
            arr[y, x, ...] = 0
    for name in (
        "flee_compiled_action",
        "pursue_compiled_action",
        "compiled_execution_action",
        "action_target_y",
        "action_target_x",
        "action_target_ow_id",
        "action_target_kind",
        "action_target_source",
        "action_direction_y",
        "action_direction_x",
    ):
        arr = getattr(state, name, None)
        if isinstance(arr, np.ndarray) and arr.shape[:2] == state.health.shape:
            arr[y, x, ...] = -1
    if (
        isinstance(state.neighbor_trust, np.ndarray)
        and state.neighbor_trust.shape[:2] == state.health.shape
    ):
        state.neighbor_trust[y, x, ...] = 1.0
    if (
        isinstance(state.last_movement_action, np.ndarray)
        and state.last_movement_action.shape == state.health.shape
    ):
        state.last_movement_action[y, x] = int(Action.REST)


def apply_death(state: WorldState, cfg: SimulationConfig) -> None:
    """Apply death and remember death mask for metrics."""
    from owl.core.advanced import ensure_advanced_fields

    ensure_advanced_fields(state, cfg)
    assert state.last_death_mask is not None
    before_dead = detect_dead_cells(state, cfg)
    state.last_death_mask[...] = before_dead
    _mvp_apply_death(state, cfg)


# --- RAQIC terminal-state handling -----------------------------------------
_scientific_clear_cell_v091 = _advanced_clear_cell


def clear_cell(state: WorldState, position: tuple[int, int]) -> None:
    """Clear one OW and place every decision representation in REST.

     makes the terminal-state law explicit: dead cells cannot retain an
    authoritative non-REST RAQIC readout or stale action distribution.
    """
    _scientific_clear_cell_v091(state, position)
    y, x = map(int, position)
    rest = int(Action.REST)
    for name in (
        "raqic_readout",
        "raqic_record_action",
        "raqic_record_readout",
        "raqic_legacy_shadow_readout",
    ):
        arr = getattr(state, name, None)
        if isinstance(arr, np.ndarray) and arr.shape == state.health.shape:
            arr[y, x] = rest
    for name in (
        "raqic_probabilities",
        "last_action_probabilities",
        "last_macro_probabilities",
        "raqic_legacy_shadow_possibility",
    ):
        arr = getattr(state, name, None)
        if isinstance(arr, np.ndarray) and arr.shape[:2] == state.health.shape:
            arr[y, x, ...] = 0
            if arr.shape[-1] > rest:
                arr[y, x, rest] = 1
    for name in ("raqic_score", "raqic_phase"):
        arr = getattr(state, name, None)
        if isinstance(arr, np.ndarray) and arr.shape[:2] == state.health.shape:
            arr[y, x, ...] = 0
    parent_intention = getattr(state, "raqic_parent_intention", None)
    if (
        isinstance(parent_intention, np.ndarray)
        and parent_intention.shape[:2] == state.health.shape
    ):
        parent_intention[y, x, ...] = 0
        if parent_intention.shape[-1] > rest:
            parent_intention[y, x, rest] = 1
    for name in (
        "raqic_record_confidence",
        "raqic_trace_error",
        "raqic_min_eigenvalue",
        "raqic_backend_code",
        "raqic_compare_l1",
        "raqic_compare_kl",
    ):
        arr = getattr(state, name, None)
        if isinstance(arr, np.ndarray) and arr.shape == state.health.shape:
            arr[y, x] = 0
    audit_flags = getattr(state, "raqic_audit_flags", None)
    if isinstance(audit_flags, np.ndarray) and audit_flags.shape[:2] == state.health.shape:
        audit_flags[y, x, ...] = 0
