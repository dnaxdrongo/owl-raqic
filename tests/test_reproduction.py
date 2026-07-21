"""Reproduction, lineage, and topology hook tests."""

from __future__ import annotations

import numpy as np

from owl.core.actions import Action, EventKind
from owl.core.config import SimulationConfig, load_config
from owl.core.init import initialize_world
from owl.core.state import EventRecord, OWRecord
from owl.engine.events import enqueue_event
from owl.engine.reproduction import (
    apply_reproduction,
    copy_child_from_parent,
    find_empty_neighbor_positions,
    update_lineage,
)
from owl.engine.topology import (
    apply_topology_events,
    detect_topology_events,
    expel_child_ow,
    merge_ows,
    split_ow,
)


def make_pass09_cfg(height: int = 20, width: int = 20) -> SimulationConfig:
    """Return a small deterministic config for reproduction/topology tests."""
    data = load_config("configs/mvp.yaml").model_dump()
    data["world"]["height"] = height
    data["world"]["width"] = width
    data["world"]["patch_size"] = 5
    data["world"]["boundary_mode"] = "toroidal"
    data["initialization"]["population_density"] = 0.0
    data["initialization"]["food_patch_count"] = 0
    data["initialization"]["toxin_patch_count"] = 0
    data["visualization"]["enabled"] = False
    data["recording"]["enabled"] = False
    data["reproduction"]["enabled"] = True
    data["reproduction"]["min_resource"] = 0.6
    data["reproduction"]["min_health"] = 0.6
    data["reproduction"]["min_boundary"] = 0.5
    data["reproduction"]["min_integration"] = 0.4
    data["reproduction"]["mutation_sigma"] = 0.0
    data["reproduction"]["channel_mutation_sigma"] = 0.0
    data["reproduction"]["child_resource_fraction"] = 0.25
    return SimulationConfig.model_validate(data)


def seed_cell(state, y: int, x: int, *, resource: float = 0.9, health: float = 0.9) -> None:
    """Place one living cell with stable reproduction-ready fields."""
    h, w = state.health.shape
    flat_id = y * w + x
    state.health[y, x] = health
    state.resource[y, x] = resource
    state.boundary[y, x] = 0.8
    state.activation[y, x] = 0.3
    state.memory[y, x] = 0.4
    state.integration[y, x] = 0.7
    state.threshold[y, x] = 0.5
    state.phase[y, x] = 1.25
    state.readout[y, x] = int(Action.REST)
    state.ow_type[y, x] = 1
    state.lineage_id[y, x] = 123
    state.occupancy[y, x] = flat_id
    state.parent_id[y, x] = 0
    state.mobility[y, x] = 0.5
    state.metabolism[y, x] = 0.5
    state.predation[y, x] = 0.2
    state.grazing[y, x] = 0.7
    state.cooperation[y, x] = 0.6
    state.aggression[y, x] = 0.1
    state.curiosity[y, x] = 0.4
    state.reproduction_rate[y, x] = 1.0
    state.toxin_resistance[y, x] = 0.3
    state.memory_capacity[y, x] = 0.8
    state.coupling_strength[y, x] = 0.6
    state.emit_strength[y, x] = 0.5
    state.emit_efficiency[y, x] = 0.6
    state.receive_sensitivity[y, x] = 0.7
    state.signal_precision[y, x] = 0.8
    state.honesty_bias[y, x] = 0.9
    state.deception_bias[y, x] = 0.05
    state.possibility[y, x, :] = 0.0
    state.possibility[y, x, int(Action.REST)] = 1.0
    state.channel_receptivity[y, x, :] = 0.55
    state.channel_emission_bias[y, x, :] = 0.45
    state.channel_trust_local[y, x, :] = 0.80
    state.signal_memory[y, x, :] = 0.20


def make_state(seed: int = 0):
    cfg = make_pass09_cfg()
    state = initialize_world(cfg, np.random.default_rng(seed))
    # initialize_world with population_density=0 still keeps empty fields valid.
    return cfg, state


def test_find_empty_neighbor_positions_respects_occupancy_obstacles_and_toroidal_boundary() -> None:
    cfg, state = make_state()
    seed_cell(state, 0, 0)
    # Occupy one toroidal neighbor and block another.
    seed_cell(state, 19, 0)
    state.obstacle[0, 1] = True

    empty = find_empty_neighbor_positions(state, (0, 0), cfg)

    assert (1, 0) in empty
    assert (0, 19) in empty
    assert (19, 0) not in empty
    assert (0, 1) not in empty
    assert len(empty) == 2


def test_copy_child_from_parent_transfers_resource_inherits_traits_and_preserves_simplex() -> None:
    cfg, state = make_state()
    seed_cell(state, 5, 5, resource=0.8)
    child = (5, 6)

    before_resource = float(state.resource[5, 5])
    copy_child_from_parent(state, (5, 5), child, cfg, np.random.default_rng(123))

    expected_child_resource = cfg.reproduction.child_resource_fraction * before_resource
    assert np.isclose(state.resource[child], expected_child_resource)
    assert np.isclose(state.resource[5, 5], before_resource - expected_child_resource)
    assert state.health[child] == cfg.reproduction.initial_child_health
    assert state.boundary[child] == cfg.reproduction.initial_child_boundary
    assert state.memory[child] == cfg.reproduction.memory_inheritance * state.memory[5, 5]
    assert state.lineage_id[child] == state.lineage_id[5, 5]
    assert state.occupancy[child] == child[0] * state.health.shape[1] + child[1]
    assert state.readout[child] == int(Action.REST)
    assert np.allclose(state.possibility[child].sum(), 1.0)
    assert state.possibility[child][int(Action.REST)] == 1.0
    assert np.all(
        (state.channel_receptivity[child] >= 0.0) & (state.channel_receptivity[child] <= 1.0)
    )
    assert np.all(
        (state.channel_emission_bias[child] >= 0.0) & (state.channel_emission_bias[child] <= 1.0)
    )


def test_update_lineage_uses_parent_lineage_or_parent_flat_id() -> None:
    cfg, state = make_state()
    del cfg
    seed_cell(state, 2, 2)
    state.lineage_id[2, 2] = -1
    update_lineage(state, (2, 2), (2, 3))

    assert state.lineage_id[2, 3] == 2 * state.health.shape[1] + 2
    assert state.age[2, 3] == 0


def test_apply_reproduction_places_child_and_records_event() -> None:
    cfg, state = make_state()
    seed_cell(state, 8, 8, resource=0.9)
    state.readout[8, 8] = int(Action.REPRODUCE)
    state.reproduction_rate[8, 8] = 1.0

    before_resource = float(state.resource[8, 8])
    apply_reproduction(state, cfg, np.random.default_rng(10))

    born_positions = np.argwhere((state.health > 0.0) & (state.lineage_id == 123))
    # Parent plus one child.
    assert len(born_positions) == 2
    assert state.resource[8, 8] < before_resource
    assert any(event.kind == str(EventKind.REPRODUCTION) for event in state.event_queue)
    assert np.all(np.isfinite(state.health))
    assert np.all((state.health >= 0.0) & (state.health <= 1.0))
    assert np.allclose(state.possibility.sum(axis=-1), 1.0, atol=1e-6)


def test_apply_reproduction_does_nothing_when_disabled_or_not_viable() -> None:
    cfg, state = make_state()
    seed_cell(state, 4, 4, resource=0.2)
    state.readout[4, 4] = int(Action.REPRODUCE)

    apply_reproduction(state, cfg, np.random.default_rng(0))
    assert np.count_nonzero(state.health > 0.0) == 1

    data = cfg.model_dump()
    data["reproduction"]["enabled"] = False
    disabled = SimulationConfig.model_validate(data)
    state.resource[4, 4] = 0.9
    apply_reproduction(state, disabled, np.random.default_rng(0))
    assert np.count_nonzero(state.health > 0.0) == 1


def test_detect_and_apply_topology_events_are_safe_hooks() -> None:
    cfg, state = make_state()
    seed_cell(state, 6, 6)
    seed_cell(state, 6, 7)
    state.readout[6, 6] = int(Action.MERGE)
    state.readout[6, 7] = int(Action.SPLIT)

    health_before = state.health.copy()
    detect_topology_events(state, cfg)

    assert any(event.kind == str(EventKind.MERGE) for event in state.event_queue)
    assert any(event.kind == str(EventKind.SPLIT) for event in state.event_queue)

    apply_topology_events(state, cfg)

    assert not any(
        event.kind in {str(EventKind.MERGE), str(EventKind.SPLIT), str(EventKind.EXPULSION)}
        for event in state.event_queue
    )
    assert np.array_equal(state.health, health_before)


def test_topology_noop_hooks_validate_positions_and_sparse_expulsion_updates_mobile_records() -> (
    None
):
    cfg, state = make_state()
    del cfg
    seed_cell(state, 1, 1)
    seed_cell(state, 1, 2)

    merge_ows(state, (1, 1), (1, 2))
    split_ow(state, (1, 1))

    parent = OWRecord(
        id=10,
        type_id=0,
        pos_y=1,
        pos_x=1,
        occupied_cells=[(1, 1)],
        parent_id=None,
        children=[11],
        traits=np.zeros((3,), dtype=np.float32),
        alive=True,
    )
    child = OWRecord(
        id=11,
        type_id=0,
        pos_y=1,
        pos_x=2,
        occupied_cells=[(1, 2)],
        parent_id=10,
        children=[],
        traits=np.zeros((3,), dtype=np.float32),
        alive=True,
    )
    state.mobile_ows[10] = parent
    state.mobile_ows[11] = child

    expel_child_ow(state, 10, 11)

    assert parent.children == []
    assert child.parent_id is None


def test_apply_topology_events_preserves_unrelated_events() -> None:
    cfg, state = make_state()
    seed_cell(state, 3, 3)
    enqueue_event(state, EventRecord(kind=str(EventKind.DEATH), tick=0, source=(3, 3)))
    enqueue_event(state, EventRecord(kind=str(EventKind.SPLIT), tick=0, source=(3, 3)))

    apply_topology_events(state, cfg)

    assert [event.kind for event in state.event_queue] == [str(EventKind.DEATH)]
