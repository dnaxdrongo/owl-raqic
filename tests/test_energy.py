"""Initialization, trait mutation, and resource seeding tests."""

from __future__ import annotations

import numpy as np

from owl.core.actions import Action
from owl.core.config import SimulationConfig, load_config
from owl.core.init import (
    initialize_world,
)
from owl.core.state import OWRecord
from owl.core.traits import (
    copy_traits_with_mutation,
    default_trait_presets,
    mutate_scalar_trait,
    mutate_trait_vector,
)
from owl.engine.death import apply_death, clear_cell, detect_dead_cells, release_internal_ows
from owl.engine.feeding import apply_feeding, compute_intake, deposit_resource_residue
from owl.engine.health import apply_metabolism_damage, apply_repair_and_integrate
from owl.engine.loop import run_headless
from owl.engine.memory import compute_identity_persistence, encode_experience, update_memory


def test_default_trait_presets_are_bounded_and_include_core_roles() -> None:
    presets = default_trait_presets()
    assert {"grazer", "cooperator", "proto_carnivore", "scavenger"}.issubset(presets)

    for preset in presets.values():
        for field in [
            "mobility",
            "metabolism",
            "predation",
            "grazing",
            "cooperation",
            "aggression",
            "curiosity",
            "reproduction_rate",
            "toxin_resistance",
            "emit_strength",
            "emit_efficiency",
            "receive_sensitivity",
            "honesty_bias",
            "deception_bias",
            "memory_capacity",
            "coupling_strength",
            "signal_precision",
        ]:
            value = getattr(preset, field)
            assert 0.0 <= value <= 1.0, (preset.name, field)


def test_mutate_scalar_and_vector_traits_are_clipped() -> None:
    rng = np.random.default_rng(123)
    assert 0.0 <= mutate_scalar_trait(0.5, 0.1, rng) <= 1.0
    assert mutate_scalar_trait(3.0, 0.0, rng) == 1.0
    assert mutate_scalar_trait(-2.0, 0.0, rng) == 0.0

    values = np.array([-1.0, 0.25, 2.0], dtype=np.float32)
    mutated = mutate_trait_vector(values, 0.0, rng)
    assert mutated.dtype == np.float32
    assert np.all(mutated >= 0.0)
    assert np.all(mutated <= 1.0)
    assert np.array_equal(mutated, np.array([0.0, 0.25, 1.0], dtype=np.float32))


def test_population_traits_food_and_possibilities_can_be_initialized_in_steps() -> None:
    cfg = load_config("configs/mvp.yaml")
    rng = np.random.default_rng(7)
    state = initialize_world(cfg, rng)

    living = state.health > 0.0
    assert living.any()
    assert state.food.sum() > 0.0
    assert np.allclose(state.possibility.sum(axis=-1), 1.0, atol=1e-6)
    assert np.all(state.possibility[~living, int(Action.REST)] == 1.0)

    # Living cells should have universal communication traits, not a single signaler role.
    assert np.any(state.emit_strength[living] > 0.0)
    assert np.any(state.receive_sensitivity[living] > 0.0)
    assert np.all(state.channel_trust_local[living] >= 0.0)
    assert np.all(state.channel_trust_local[living] <= 1.0)


def test_copy_traits_with_mutation_only_changes_target_traits() -> None:
    cfg = load_config("configs/mvp.yaml")
    rng = np.random.default_rng(9)
    state = initialize_world(cfg, rng)
    living = np.argwhere(state.health > 0.0)
    source = tuple(map(int, living[0]))
    target = (0, 0)
    if source == target:
        target = (0, 1)

    state.health[target] = 0.8
    before_source_mobility = float(state.mobility[source])
    before_source_channel = state.channel_receptivity[source].copy()

    copy_traits_with_mutation(state, source, target, cfg, np.random.default_rng(10))

    assert np.isclose(state.mobility[source], before_source_mobility)
    assert np.array_equal(state.channel_receptivity[source], before_source_channel)
    assert 0.0 <= state.mobility[target] <= 1.0
    assert np.all(state.channel_receptivity[target] >= 0.0)
    assert np.all(state.channel_receptivity[target] <= 1.0)


def test_initialize_world_rejects_unknown_type_weights() -> None:
    cfg = load_config("configs/mvp.yaml")
    data = cfg.model_dump()
    data["initialization"]["type_weights"] = {"not_a_role": 1.0}
    bad_cfg = type(cfg).model_validate(data)

    try:
        initialize_world(bad_cfg, np.random.default_rng(1))
    except ValueError as exc:
        assert "unknown initialization type_weights" in str(exc)
    else:
        raise AssertionError("unknown type weight should fail during initialization")


# Feeding, health, memory, and death tests.


def make_pass07_cfg(height: int = 20, width: int = 20):
    data = load_config("configs/mvp.yaml").model_dump()
    data["world"]["height"] = height
    data["world"]["width"] = width
    data["world"]["patch_size"] = 5
    data["initialization"]["population_density"] = 0.75
    data["initialization"]["food_patch_count"] = 1
    data["initialization"]["food_patch_radius"] = 2
    data["visualization"]["enabled"] = False
    data["recording"]["enabled"] = False
    return type(load_config("configs/mvp.yaml")).model_validate(data)


def make_pass07_state(seed: int = 123):
    cfg = make_pass07_cfg()
    state = initialize_world(cfg, np.random.default_rng(seed))
    return cfg, state


def first_living_position(state):
    living = np.argwhere((state.health > 0.0) & (~state.obstacle))
    assert living.size > 0
    return tuple(map(int, living[0]))


def test_compute_intake_is_pure_and_apply_feeding_transfers_food_to_resource() -> None:
    cfg, state = make_pass07_state()
    y, x = first_living_position(state)

    state.food.fill(0.0)
    state.resource.fill(0.0)
    state.grazing.fill(0.0)
    state.readout.fill(int(Action.REST))

    state.food[y, x] = 1.0
    state.resource[y, x] = 0.20
    state.grazing[y, x] = 1.0
    state.readout[y, x] = int(Action.FEED)

    food_before = state.food.copy()
    resource_before = state.resource.copy()
    intake = compute_intake(state, cfg)

    assert intake.shape == state.health.shape
    assert intake.dtype == np.float32
    assert np.all((intake >= 0.0) & (intake <= 1.0))
    assert intake[y, x] > 0.0
    assert np.array_equal(state.food, food_before)
    assert np.array_equal(state.resource, resource_before)

    apply_feeding(state, cfg)
    assert np.isclose(state.resource[y, x], resource_before[y, x] + intake[y, x])
    assert np.isclose(state.food[y, x], food_before[y, x] - intake[y, x])
    assert np.all((state.resource >= 0.0) & (state.resource <= cfg.resources.max_resource))
    assert np.all((state.food >= 0.0) & (state.food <= 1.0))


def test_deposit_resource_residue_accumulates_duplicates_and_clips() -> None:
    cfg, state = make_pass07_state()
    del cfg
    state.food.fill(0.0)
    positions = np.array([[1, 1], [1, 1], [2, 3]], dtype=np.int64)
    amount = np.array([0.4, 0.7, 0.2], dtype=np.float32)

    deposit_resource_residue(state, amount, positions)

    assert np.isclose(state.food[1, 1], 1.0)
    assert np.isclose(state.food[2, 3], 0.2)
    assert np.all((state.food >= 0.0) & (state.food <= 1.0))


def test_apply_repair_and_integrate_spends_resource_and_improves_state() -> None:
    cfg, state = make_pass07_state()
    y, x = first_living_position(state)
    y2, x2 = tuple(map(int, np.argwhere((state.health > 0.0) & (~state.obstacle))[1]))

    state.resource[y, x] = 0.8
    state.health[y, x] = 0.4
    state.boundary[y, x] = 0.3
    state.readout[y, x] = int(Action.REPAIR)

    state.resource[y2, x2] = 0.8
    state.memory[y2, x2] = 0.1
    state.boundary[y2, x2] = 0.3
    state.integration[y2, x2] = 0.2
    state.readout[y2, x2] = int(Action.INTEGRATE)

    repair_before = (
        float(state.resource[y, x]),
        float(state.health[y, x]),
        float(state.boundary[y, x]),
    )
    integrate_before = (
        float(state.resource[y2, x2]),
        float(state.memory[y2, x2]),
        float(state.boundary[y2, x2]),
    )

    apply_repair_and_integrate(state, cfg)

    assert state.resource[y, x] < repair_before[0]
    assert state.health[y, x] > repair_before[1]
    assert state.boundary[y, x] > repair_before[2]
    assert state.resource[y2, x2] < integrate_before[0]
    assert state.memory[y2, x2] >= integrate_before[1]
    assert state.boundary[y2, x2] >= integrate_before[2]
    assert np.all((state.health >= 0.0) & (state.health <= 1.0))
    assert np.all((state.boundary >= 0.0) & (state.boundary <= 1.0))


def test_apply_metabolism_damage_reduces_resource_and_toxin_hurts_health() -> None:
    cfg, state = make_pass07_state()
    y, x = first_living_position(state)

    state.resource[y, x] = 0.5
    state.metabolism[y, x] = 1.0
    state.toxin[y, x] = 1.0
    state.toxin_resistance[y, x] = 0.0
    state.health[y, x] = 0.9
    state.boundary[y, x] = 0.9

    before = (float(state.resource[y, x]), float(state.health[y, x]), float(state.boundary[y, x]))
    apply_metabolism_damage(state, cfg)

    assert state.resource[y, x] < before[0]
    assert state.health[y, x] < before[1]
    assert state.boundary[y, x] < before[2]
    assert np.all((state.resource >= 0.0) & (state.resource <= cfg.resources.max_resource))
    assert np.all((state.health >= 0.0) & (state.health <= 1.0))
    assert np.all((state.boundary >= 0.0) & (state.boundary <= 1.0))


def test_encode_experience_update_memory_and_identity_are_bounded() -> None:
    cfg, state = make_pass07_state()
    y, x = first_living_position(state)
    state.readout[y, x] = int(Action.INTEGRATE)
    state.signal_reception[y, x, :] = 0.5
    state.memory[y, x] = 0.1

    experience = encode_experience(state, cfg)
    identity = compute_identity_persistence(state, cfg)
    before = state.memory.copy()
    update_memory(state, cfg)

    assert experience.shape == state.health.shape
    assert identity.shape == state.health.shape
    assert experience.dtype == np.float32
    assert identity.dtype == np.float32
    assert np.all((experience >= 0.0) & (experience <= 1.0))
    assert np.all((identity >= 0.0) & (identity <= 1.0))
    assert state.memory[y, x] != before[y, x]
    assert np.all((state.memory >= 0.0) & (state.memory <= 1.0))


def test_detect_dead_cells_apply_death_and_clear_cell_reset_cell_owned_fields() -> None:
    cfg, state = make_pass07_state()
    y, x = first_living_position(state)

    state.food[y, x] = 0.0
    state.resource[y, x] = 0.4
    state.health[y, x] = 0.0
    state.boundary[y, x] = 0.5
    state.memory[y, x] = 0.7
    state.integration[y, x] = 0.6
    state.signal_memory[y, x, :] = 1.0
    state.channel_trust_local[y, x, :] = 1.0

    dead = detect_dead_cells(state, cfg)
    assert dead[y, x]

    apply_death(state, cfg)

    assert state.health[y, x] == 0.0
    assert state.resource[y, x] == 0.0
    assert state.boundary[y, x] == 0.0
    assert state.memory[y, x] == 0.0
    assert state.integration[y, x] == 0.0
    assert state.occupancy[y, x] == -1
    assert state.readout[y, x] == int(Action.REST)
    assert state.possibility[y, x, int(Action.REST)] == 1.0
    assert np.isclose(state.possibility[y, x].sum(), 1.0)
    assert np.all(state.signal_memory[y, x, :] == 0.0)
    assert np.all(state.channel_trust_local[y, x, :] == 0.0)
    assert state.food[y, x] > 0.0


def test_clear_cell_validates_position_and_release_internal_ows_adds_event() -> None:
    cfg, state = make_pass07_state()
    y, x = first_living_position(state)

    state.mobile_ows[99] = OWRecord(
        id=99,
        type_id=1,
        pos_y=y,
        pos_x=x,
        occupied_cells=[(y, x)],
        parent_id=None,
        children=[100, 101],
        traits=np.ones(3, dtype=np.float32),
        alive=True,
    )

    release_internal_ows(state, (y, x), cfg)
    assert state.event_queue
    assert state.event_queue[-1].payload["children"] == [100, 101]

    clear_cell(state, (y, x))
    assert state.health[y, x] == 0.0
    assert state.possibility[y, x, int(Action.REST)] == 1.0

    try:
        clear_cell(state, (-1, 0))
    except ValueError as exc:
        assert "outside field shape" in str(exc)
    else:
        raise AssertionError("out-of-bounds clear_cell should fail")


# Loop resource and energy tests.


def make_pass10_energy_cfg() -> SimulationConfig:
    data = load_config("configs/mvp.yaml").model_dump()
    data["world"]["height"] = 20
    data["world"]["width"] = 20
    data["world"]["patch_size"] = 5
    data["world"]["max_steps"] = 12
    data["initialization"]["population_density"] = 0.40
    data["initialization"]["food_patch_count"] = 2
    data["visualization"]["enabled"] = False
    data["recording"]["enabled"] = False
    return SimulationConfig.model_validate(data)


def test_loop_resource_and_food_fields_remain_bounded() -> None:
    cfg = make_pass10_energy_cfg()
    state, metrics = run_headless(cfg, max_steps=12)

    assert len(metrics) == 12
    assert np.all((state.resource >= 0.0) & (state.resource <= cfg.resources.max_resource + 1e-6))
    assert np.all((state.food >= 0.0) & (state.food <= 1.0 + 1e-6))
    assert np.all((state.health >= 0.0) & (state.health <= 1.0 + 1e-6))
    assert all("mean_resource" in row for row in metrics)
