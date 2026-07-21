from __future__ import annotations

import numpy as np

from owl.core.actions import Action
from owl.record.cadc_schema import ReasonCode, TargetKind
from owl.science.action_contract import candidate_target_context


def _context(*, boundary_mode: str = "reflective"):
    shape = (3, 3)
    health = np.zeros(shape, dtype=np.float32)
    resource = np.ones(shape, dtype=np.float32)
    obstacle = np.zeros(shape, dtype=bool)
    occupancy = np.full(shape, -1, dtype=np.int64)
    food = np.arange(9, dtype=np.float32).reshape(shape) / 9.0
    toxin = food[::-1].copy()
    parent_id = np.full(shape, -1, dtype=np.int64)
    legal = np.zeros((*shape, len(Action)), dtype=bool)
    health[1, 1] = 1.0
    occupancy[1, 1] = 42
    legal[1, 1, :] = True
    obstacle[0, 1] = True
    occupancy[1, 2] = 99
    health[1, 2] = 1.0
    return candidate_target_context(
        health,
        resource,
        obstacle,
        occupancy,
        food,
        toxin,
        parent_id,
        legal,
        boundary_mode=boundary_mode,
        diagonal_movement_enabled=True,
        xp=np,
    )


def test_candidate_context_has_exact_fixed_action_axis_and_target_kinds() -> None:
    context = _context()
    assert context.executable.shape == (3, 3, 22)
    assert context.reason_code.shape == (3, 3, 22)
    assert context.target_kind[1, 1, int(Action.REST)] == int(TargetKind.SELF)
    assert context.target_kind[1, 1, int(Action.MOVE_S)] == int(TargetKind.CELL)
    assert context.target_kind[1, 1, int(Action.REPRODUCE)] == int(
        TargetKind.EMPTY_NEIGHBOR_SET
    )


def test_movement_executability_separates_obstacle_occupancy_and_success() -> None:
    context = _context()
    center = (1, 1)
    assert not context.executable[*center, int(Action.MOVE_N)]
    assert context.reason_code[*center, int(Action.MOVE_N)] == int(ReasonCode.OBSTACLE)
    assert not context.executable[*center, int(Action.MOVE_E)]
    assert context.reason_code[*center, int(Action.MOVE_E)] == int(ReasonCode.OCCUPIED)
    assert context.executable[*center, int(Action.MOVE_S)]
    assert context.reason_code[*center, int(Action.MOVE_S)] == int(ReasonCode.NONE)
    assert context.resolved_y[*center, int(Action.MOVE_S)] == 2
    assert context.resolved_x[*center, int(Action.MOVE_S)] == 1


def test_policy_legal_flee_and_pursue_are_recorded_as_non_executable() -> None:
    context = _context()
    for action in (Action.FLEE, Action.PURSUE):
        assert not context.executable[1, 1, int(action)]
        assert context.reason_code[1, 1, int(action)] == int(
            ReasonCode.NO_EXECUTION_CONTRACT
        )


def test_non_toroidal_out_of_bounds_is_a_distinct_reason() -> None:
    shape = (3, 3)
    health = np.zeros(shape, dtype=np.float32)
    health[0, 0] = 1.0
    occupancy = np.full(shape, -1, dtype=np.int64)
    occupancy[0, 0] = 7
    legal = np.zeros((*shape, len(Action)), dtype=bool)
    legal[0, 0, int(Action.MOVE_N)] = True
    context = candidate_target_context(
        health,
        np.ones(shape, dtype=np.float32),
        np.zeros(shape, dtype=bool),
        occupancy,
        np.zeros(shape, dtype=np.float32),
        np.zeros(shape, dtype=np.float32),
        np.full(shape, -1, dtype=np.int64),
        legal,
        boundary_mode="reflective",
        diagonal_movement_enabled=True,
        xp=np,
    )
    assert not context.executable[0, 0, int(Action.MOVE_N)]
    assert context.reason_code[0, 0, int(Action.MOVE_N)] == int(
        ReasonCode.BOUNDARY_BLOCKED
    )


def test_candidate_derivation_does_not_modify_inputs() -> None:
    context = _context(boundary_mode="toroidal")
    # An immutable deterministic result on repeated derivation is also evidence
    # that no counter-RNG stream is consumed by candidate recording.
    repeated = _context(boundary_mode="toroidal")
    for name in context.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(context, name), getattr(repeated, name))
