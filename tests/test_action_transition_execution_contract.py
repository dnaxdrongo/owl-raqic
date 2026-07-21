from __future__ import annotations

import copy

import numpy as np

from owl.core.actions import Action
from owl.core.advanced import ensure_action_transition_fields
from owl.core.config import SimulationConfig
from owl.core.init import initialize_world
from owl.engine.action_transitions import (
    apply_active_sense_transition,
    compile_selected_action_transition,
    prepare_action_transition_context,
)
from owl.engine.movement import apply_movement
from owl.gpu.device_state import OWLDeviceState
from owl.gpu.stages.action_transitions_gpu import (
    apply_active_sense_transition_gpu,
    compile_selected_action_transition_gpu,
    prepare_action_transition_context_gpu,
)
from owl.gpu.stages.movement_gpu import apply_movement_gpu


def _cfg() -> SimulationConfig:
    return SimulationConfig.model_validate(
        {
            "world": {
                "height": 10,
                "width": 10,
                "patch_size": 5,
                "seed": 19,
            },
            "initialization": {
                "population_density": 0.0,
                "food_patch_count": 0,
                "toxin_patch_count": 0,
                "background_food": 0.0,
            },
            "action_transitions": {
                "enabled": True,
                "action_contract_version": "owl.action-transitions.v1",
                "legacy_unsupported_action_recovery": False,
                "active_sense_enabled": True,
                "flee_execution_enabled": True,
                "pursue_execution_enabled": True,
            },
        }
    )


def _state(cfg: SimulationConfig, *, pursuit: bool) -> object:
    state = initialize_world(cfg, np.random.default_rng(cfg.world.seed))
    state.health.fill(0)
    state.resource.fill(0)
    state.boundary.fill(0)
    state.occupancy.fill(-1)
    state.obstacle.fill(False)
    state.toxin.fill(0)
    state.food.fill(0)
    state.mobility.fill(0)
    state.predation.fill(0)
    state.aggression.fill(0)
    state.readout.fill(int(Action.REST))
    y, x = 5, 5
    state.health[y, x] = 1.0
    state.resource[y, x] = 0.8
    state.boundary[y, x] = 1.0
    state.mobility[y, x] = 1.0
    state.occupancy[y, x] = 55
    if pursuit:
        state.predation[y, x] = 1.0
        state.health[3, 7] = 1.0
        state.resource[3, 7] = 0.8
        state.boundary[3, 7] = 1.0
        state.occupancy[3, 7] = 37
    else:
        state.toxin[5, 3] = 1.0
    ensure_action_transition_fields(state, cfg)
    state.tick = 1
    return state


def _agent_position(occupancy: np.ndarray) -> tuple[int, int]:
    found = np.argwhere(occupancy == 55)
    assert found.shape == (1, 2)
    return int(found[0, 0]), int(found[0, 1])


def _distance(position: tuple[int, int], target: tuple[int, int]) -> int:
    return max(abs(position[0] - target[0]), abs(position[1] - target[1]))


def _run_cpu_and_backend(action: Action) -> tuple[object, OWLDeviceState, object, object]:
    cfg = _cfg()
    initial = _state(cfg, pursuit=action == Action.PURSUE)
    cpu = copy.deepcopy(initial)
    backend_state = copy.deepcopy(initial)

    cpu_context = prepare_action_transition_context(cpu, cfg)
    cpu.readout[5, 5] = int(action)
    compile_selected_action_transition(cpu, cfg)
    apply_movement(cpu, cfg, np.random.default_rng(cfg.world.seed))

    ds = OWLDeviceState.from_world_state(
        backend_state, cfg, force_backend="numpy", strict=False, allow_fallback=True
    )
    device_context = prepare_action_transition_context_gpu(ds, cfg)
    ds.readout[5, 5] = int(action)
    compile_selected_action_transition_gpu(ds, cfg)
    apply_movement_gpu(ds, cfg)
    return cpu, ds, cpu_context, device_context


def test_flee_is_authoritative_preserves_identity_and_increases_separation() -> None:
    cpu, ds, cpu_context, device_context = _run_cpu_and_backend(Action.FLEE)
    assert cpu_context is not None
    assert int(cpu_context.target_y[5, 5, 0]) == 5
    assert int(cpu_context.target_x[5, 5, 0]) == 3
    assert int(cpu_context.flee_compiled_action[5, 5]) == int(Action.MOVE_E)
    np.testing.assert_array_equal(
        cpu_context.flee_compiled_action, device_context.flee_compiled_action
    )
    cpu_position = _agent_position(cpu.occupancy)
    device_position = _agent_position(np.asarray(ds.occupancy))
    assert cpu_position == device_position == (5, 6)
    assert _distance(cpu_position, (5, 3)) > _distance((5, 5), (5, 3))
    assert int(cpu.readout[cpu_position]) == int(Action.FLEE)
    assert int(ds.readout[device_position]) == int(Action.FLEE)
    assert int(cpu.compiled_execution_action[cpu_position]) == int(Action.MOVE_E)
    assert int(ds.compiled_execution_action[device_position]) == int(Action.MOVE_E)
    np.testing.assert_allclose(cpu.resource, ds.resource, atol=0, rtol=0)


def test_pursue_is_authoritative_preserves_identity_and_decreases_distance() -> None:
    cpu, ds, cpu_context, device_context = _run_cpu_and_backend(Action.PURSUE)
    assert cpu_context is not None
    assert int(cpu_context.target_ow_id[5, 5, 1]) == 37
    assert int(cpu_context.pursue_compiled_action[5, 5]) == int(Action.MOVE_NE)
    np.testing.assert_array_equal(
        cpu_context.pursue_compiled_action, device_context.pursue_compiled_action
    )
    cpu_position = _agent_position(cpu.occupancy)
    device_position = _agent_position(np.asarray(ds.occupancy))
    assert cpu_position == device_position == (4, 6)
    assert _distance(cpu_position, (3, 7)) < _distance((5, 5), (3, 7))
    assert int(cpu.readout[cpu_position]) == int(Action.PURSUE)
    assert int(ds.readout[device_position]) == int(Action.PURSUE)
    assert int(cpu.compiled_execution_action[cpu_position]) == int(Action.MOVE_NE)
    assert int(ds.compiled_execution_action[device_position]) == int(Action.MOVE_NE)
    np.testing.assert_allclose(cpu.resource, ds.resource, atol=0, rtol=0)


def test_active_sense_cost_memory_and_numpy_backend_parity() -> None:
    cfg = _cfg()
    initial = _state(cfg, pursuit=False)
    initial.food[3:8, 3:8] = np.arange(25, dtype=np.float32).reshape(5, 5) / 25
    cpu = copy.deepcopy(initial)
    device_state = copy.deepcopy(initial)
    cpu.readout[5, 5] = int(Action.SENSE)
    before = float(cpu.resource[5, 5])
    cpu_result = apply_active_sense_transition(cpu, cfg)

    ds = OWLDeviceState.from_world_state(
        device_state, cfg, force_backend="numpy", strict=False, allow_fallback=True
    )
    ds.readout[5, 5] = int(Action.SENSE)
    device_result = apply_active_sense_transition_gpu(ds, cfg)
    assert cpu_result.success[5, 5]
    assert device_result.success[5, 5]
    assert float(cpu.resource[5, 5]) == float(
        np.float32(before - cfg.action_transitions.active_sense_cost)
    )
    for name in (
        "resource",
        "active_sense_food_memory",
        "active_sense_toxin_memory",
        "active_sense_alive_memory",
        "active_sense_ttl",
        "active_sense_new_cell_count",
        "active_sense_new_target_count",
    ):
        np.testing.assert_array_equal(getattr(cpu, name), ds.arrays[name])


def test_active_sense_success_with_no_new_information_is_not_failure() -> None:
    cfg = _cfg()
    state = _state(cfg, pursuit=False)
    yy, xx = np.indices(state.health.shape)
    outside_ordinary = np.maximum(abs(yy - 5), abs(xx - 5)) > 1
    state.obstacle[outside_ordinary] = True
    state.readout[5, 5] = int(Action.SENSE)
    result = apply_active_sense_transition(state, cfg)
    assert result.success[5, 5]
    assert result.no_new_information[5, 5]
    assert result.newly_observed_count[5, 5] == 0


def test_no_target_and_no_direction_fail_closed() -> None:
    cfg = _cfg()
    no_target = _state(cfg, pursuit=False)
    no_target.toxin.fill(0)
    context = prepare_action_transition_context(no_target, cfg)
    assert context is not None
    assert not context.flee_executable[5, 5]
    assert int(context.flee_compiled_action[5, 5]) == -1

    blocked = _state(cfg, pursuit=False)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if (dy, dx) != (0, 0):
                blocked.obstacle[5 + dy, 5 + dx] = True
    context = prepare_action_transition_context(blocked, cfg)
    assert context is not None
    assert not context.flee_executable[5, 5]
    assert int(context.flee_compiled_action[5, 5]) == -1
