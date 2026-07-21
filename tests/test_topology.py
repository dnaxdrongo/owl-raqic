"""Topology, movement, collision, inhibition, and ingestion tests."""

from __future__ import annotations

import numpy as np

from owl.core.actions import Action, EventKind
from owl.core.config import SimulationConfig, load_config
from owl.core.init import initialize_world
from owl.core.state import EventRecord
from owl.engine.collision import (
    apply_inhibition,
    attempt_ingestion,
    compute_ingestion_probability,
    resolve_collisions,
)
from owl.engine.events import dequeue_events, enqueue_event, route_events
from owl.engine.movement import (
    apply_movement,
    move_cell_state,
    propose_movements,
    validate_movement_targets,
    wrap_position,
)


def make_small_cfg(height: int = 20, width: int = 20) -> SimulationConfig:
    cfg = load_config("configs/mvp.yaml")
    data = cfg.model_dump()
    data["world"]["height"] = height
    data["world"]["width"] = width
    data["world"]["patch_size"] = 5
    data["world"]["boundary_mode"] = "toroidal"
    data["initialization"]["population_density"] = 0.0
    data["visualization"]["enabled"] = False
    data["recording"]["enabled"] = False
    return SimulationConfig.model_validate(data)


def seed_cell(state, y: int, x: int, *, resource: float = 0.8, health: float = 0.9) -> None:
    """Place one living cell with safe defaults for movement/collision tests."""
    h, w = state.health.shape
    flat_id = y * w + x
    state.health[y, x] = health
    state.resource[y, x] = resource
    state.boundary[y, x] = 0.8
    state.activation[y, x] = 0.2
    state.memory[y, x] = 0.3
    state.integration[y, x] = 0.5
    state.threshold[y, x] = 0.4
    state.phase[y, x] = 1.0
    state.mobility[y, x] = 1.0
    state.metabolism[y, x] = 0.2
    state.predation[y, x] = 0.0
    state.grazing[y, x] = 0.7
    state.cooperation[y, x] = 0.5
    state.aggression[y, x] = 0.2
    state.curiosity[y, x] = 0.3
    state.reproduction_rate[y, x] = 0.2
    state.toxin_resistance[y, x] = 0.2
    state.memory_capacity[y, x] = 0.5
    state.coupling_strength[y, x] = 0.5
    state.emit_strength[y, x] = 0.5
    state.emit_efficiency[y, x] = 0.5
    state.receive_sensitivity[y, x] = 0.5
    state.signal_precision[y, x] = 0.5
    state.honesty_bias[y, x] = 0.8
    state.deception_bias[y, x] = 0.1
    state.lineage_id[y, x] = flat_id
    state.occupancy[y, x] = flat_id
    state.parent_id[y, x] = 0
    state.possibility[y, x, :] = 0.0
    state.possibility[y, x, int(Action.REST)] = 1.0
    state.channel_receptivity[y, x, :] = 0.5
    state.channel_emission_bias[y, x, :] = 0.5
    state.channel_trust_local[y, x, :] = 1.0
    state.signal_memory[y, x, :] = 0.1


def test_event_queue_enqueue_dequeue_and_route_preserve_unmatched_events() -> None:
    cfg = make_small_cfg()
    state = initialize_world(cfg, np.random.default_rng(1))

    collision = EventRecord(kind=str(EventKind.COLLISION), tick=0, source=(1, 1), target=(1, 2))
    ingestion = EventRecord(kind=str(EventKind.INGESTION), tick=0, source=(2, 2), target=(2, 3))
    enqueue_event(state, collision)
    enqueue_event(state, ingestion)

    routed = route_events(state)
    assert set(routed) == {str(EventKind.COLLISION), str(EventKind.INGESTION)}
    assert len(state.event_queue) == 2

    collisions = dequeue_events(state, str(EventKind.COLLISION))
    assert collisions == [collision]
    assert state.event_queue == [ingestion]

    remaining = dequeue_events(state)
    assert remaining == [ingestion]
    assert state.event_queue == []


def test_wrap_position_uses_toroidal_modulo() -> None:
    assert wrap_position(-1, 20, 10, 20) == (9, 0)
    assert wrap_position(10, -1, 10, 20) == (0, 19)


def test_propose_and_validate_movements_detect_empty_obstacle_and_occupied_targets() -> None:
    cfg = make_small_cfg()
    state = initialize_world(cfg, np.random.default_rng(2))

    seed_cell(state, 5, 5)
    state.readout[5, 5] = int(Action.MOVE_E)
    proposals = propose_movements(state, cfg)
    assert tuple(proposals[5, 5]) == (5, 6)

    valid = validate_movement_targets(state, proposals, cfg)
    assert valid[5, 5]

    state.obstacle[5, 6] = True
    valid = validate_movement_targets(state, proposals, cfg)
    assert not valid[5, 5]
    state.obstacle[5, 6] = False

    seed_cell(state, 5, 6)
    valid = validate_movement_targets(state, proposals, cfg)
    assert not valid[5, 5]


def test_move_cell_state_moves_all_cell_owned_fields_and_preserves_source_rest_probability() -> (
    None
):
    cfg = make_small_cfg()
    state = initialize_world(cfg, np.random.default_rng(3))
    seed_cell(state, 3, 3)
    state.readout[3, 3] = int(Action.MOVE_S)
    old_channel = state.channel_receptivity[3, 3].copy()
    old_lineage = int(state.lineage_id[3, 3])

    move_cell_state(state, (3, 3), (4, 3))

    assert state.health[4, 3] > 0.0
    assert state.health[3, 3] == 0.0
    assert state.lineage_id[4, 3] == old_lineage
    assert state.occupancy[4, 3] >= 0
    assert state.occupancy[3, 3] == -1
    assert np.array_equal(state.channel_receptivity[4, 3], old_channel)
    assert state.readout[3, 3] == int(Action.REST)
    assert np.allclose(state.possibility[3, 3].sum(), 1.0)
    assert state.possibility[3, 3, int(Action.REST)] == 1.0


def test_apply_movement_moves_successes_and_queues_collisions() -> None:
    cfg = make_small_cfg()
    state = initialize_world(cfg, np.random.default_rng(4))
    seed_cell(state, 5, 5, resource=0.9)
    seed_cell(state, 6, 5, resource=0.8)

    state.readout[5, 5] = int(Action.MOVE_E)
    state.readout[6, 5] = int(Action.MOVE_N)  # target occupied by alternate source at (5,5)

    apply_movement(state, cfg, np.random.default_rng(5))

    assert state.health[5, 6] > 0.0
    assert state.health[5, 5] == 0.0
    assert any(event.kind == str(EventKind.COLLISION) for event in state.event_queue)
    assert state.resource[5, 6] < 0.9


def test_compute_ingestion_probability_is_bounded_and_trait_sensitive() -> None:
    cfg = make_small_cfg()
    state = initialize_world(cfg, np.random.default_rng(6))
    seed_cell(state, 2, 2)
    seed_cell(state, 2, 3)

    state.predation[2, 2] = 1.0
    state.aggression[2, 2] = 1.0
    state.integration[2, 2] = 1.0
    state.resource[2, 2] = 1.0
    state.health[2, 3] = 0.2
    state.boundary[2, 3] = 0.2

    strong = compute_ingestion_probability(state, (2, 2), (2, 3), cfg)
    state.predation[2, 2] = 0.0
    weak = compute_ingestion_probability(state, (2, 2), (2, 3), cfg)

    assert 0.0 <= strong <= 1.0
    assert strong > 0.5
    assert weak == 0.0


def test_attempt_ingestion_success_transfers_resource_and_clears_target() -> None:
    cfg = make_small_cfg()
    data = cfg.model_dump()
    data["predation"]["min_predation_trait"] = 0.1
    cfg = SimulationConfig.model_validate(data)
    state = initialize_world(cfg, np.random.default_rng(7))
    seed_cell(state, 4, 4, resource=0.2)
    seed_cell(state, 4, 5, resource=0.8)

    state.predation[4, 4] = 1.0
    state.integration[4, 4] = 1.0
    state.aggression[4, 4] = 1.0
    state.health[4, 5] = 0.05
    state.boundary[4, 5] = 0.05

    success = attempt_ingestion(state, (4, 4), (4, 5), cfg, np.random.default_rng(1))

    assert success
    assert state.resource[4, 4] > 0.2
    assert state.health[4, 5] == 0.0
    assert state.occupancy[4, 5] == -1
    assert state.possibility[4, 5, int(Action.REST)] == 1.0
    assert any(event.kind == str(EventKind.INGESTION) for event in state.event_queue)


def test_resolve_collisions_routes_ingest_readout_to_ingestion_attempt() -> None:
    cfg = make_small_cfg()
    data = cfg.model_dump()
    data["predation"]["min_predation_trait"] = 0.1
    cfg = SimulationConfig.model_validate(data)
    state = initialize_world(cfg, np.random.default_rng(8))
    seed_cell(state, 7, 7, resource=0.3)
    seed_cell(state, 7, 8, resource=0.8)

    state.readout[7, 7] = int(Action.INGEST)
    state.predation[7, 7] = 1.0
    state.integration[7, 7] = 1.0
    state.aggression[7, 7] = 1.0
    state.health[7, 8] = 0.05
    state.boundary[7, 8] = 0.05
    enqueue_event(
        state,
        EventRecord(kind=str(EventKind.COLLISION), tick=state.tick, source=(7, 7), target=(7, 8)),
    )

    resolve_collisions(state, cfg, np.random.default_rng(1))

    assert state.health[7, 8] == 0.0
    assert any(event.kind == str(EventKind.INGESTION) for event in state.event_queue)


def test_apply_inhibition_reduces_neighbor_activation_and_costs_resource() -> None:
    cfg = make_small_cfg()
    state = initialize_world(cfg, np.random.default_rng(9))
    seed_cell(state, 10, 10, resource=0.9)
    seed_cell(state, 10, 11, resource=0.8)
    state.readout[10, 10] = int(Action.INHIBIT)
    state.aggression[10, 10] = 1.0
    state.integration[10, 10] = 1.0
    state.cooperation[10, 10] = 1.0
    state.activation[10, 11] = 0.8

    before_neighbor_activation = float(state.activation[10, 11])
    before_resource = float(state.resource[10, 10])
    apply_inhibition(state, cfg)

    assert state.activation[10, 11] < before_neighbor_activation
    assert state.resource[10, 10] < before_resource
    assert np.all(state.activation >= 0.0)
    assert np.all(state.integration >= 0.0)
