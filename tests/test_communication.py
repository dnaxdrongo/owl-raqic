"""Environment, sensing, and passive communication tests."""

from __future__ import annotations

import numpy as np

from owl.core.actions import SignalChannel
from owl.core.config import SimulationConfig, load_config
from owl.core.init import initialize_world
from owl.engine.communication import (
    choose_signal_intent,
    compute_automatic_signal_intents,
    compute_signal_conflict,
    emit_signals,
    update_channel_trust,
    update_signal_memory,
)
from owl.engine.environment import (
    apply_obstacle_mask,
    update_environment,
    update_signal_fields,
)
from owl.engine.loop import step
from owl.engine.sensing import (
    compute_crowding,
    compute_local_food_pressure,
    compute_local_toxin_pressure,
    compute_novelty,
    compute_signal_reception,
)


def make_cfg(height: int = 20, width: int = 20) -> SimulationConfig:
    """Return a small validated config for deterministic pass tests."""
    data = load_config("configs/mvp.yaml").model_dump()
    data["world"]["height"] = height
    data["world"]["width"] = width
    data["world"]["patch_size"] = 5
    data["initialization"]["population_density"] = 0.50
    data["initialization"]["food_patch_count"] = 2
    data["initialization"]["food_patch_radius"] = 3
    data["initialization"]["toxin_patch_count"] = 1
    data["visualization"]["enabled"] = False
    data["recording"]["enabled"] = False
    return SimulationConfig.model_validate(data)


def make_state(seed: int = 123):
    """Create a small initialized state."""
    cfg = make_cfg()
    rng = np.random.default_rng(seed)
    state = initialize_world(cfg, rng)
    return cfg, state


def first_living_cell(state) -> tuple[int, int]:
    living = np.argwhere((state.health > 0.0) & (~state.obstacle))
    assert len(living) > 0
    return tuple(map(int, living[0]))


def test_environment_updates_food_toxin_signal_and_preserves_bounds() -> None:
    cfg, state = make_state()
    state.food.fill(0.0)
    state.toxin.fill(0.0)
    state.signal.fill(0.0)
    state.signal_emission.fill(0.0)

    cy, cx = first_living_cell(state)
    state.food[cy, cx] = 1.0
    state.toxin[cy, cx] = 1.0
    state.signal_emission[cy, cx, int(SignalChannel.FOOD)] = 0.6

    update_environment(state, cfg)

    assert np.all(np.isfinite(state.food))
    assert np.all(np.isfinite(state.toxin))
    assert np.all(np.isfinite(state.signal))
    assert np.all((state.food >= 0.0) & (state.food <= 1.0))
    assert np.all((state.toxin >= 0.0) & (state.toxin <= 1.0))
    assert np.all((state.signal >= 0.0) & (state.signal <= 1.0))
    assert state.signal[cy, cx, int(SignalChannel.FOOD)] > 0.0
    assert np.all(state.signal_emission == 0.0)


def test_obstacle_mask_zeroes_environment_but_not_traits() -> None:
    cfg, state = make_state()
    y, x = 0, 0
    old_mobility = float(state.mobility[y, x])
    state.obstacle[y, x] = True
    state.food[y, x] = 1.0
    state.toxin[y, x] = 1.0
    state.noise[y, x] = 1.0
    state.signal[y, x, :] = 1.0
    state.signal_memory[y, x, :] = 1.0
    state.occupancy[y, x] = 42

    apply_obstacle_mask(state)

    assert state.food[y, x] == 0.0
    assert state.toxin[y, x] == 0.0
    assert state.noise[y, x] == 0.0
    assert np.all(state.signal[y, x, :] == 0.0)
    assert np.all(state.signal_memory[y, x, :] == 0.0)
    assert state.occupancy[y, x] == -1
    assert state.mobility[y, x] == old_mobility


def test_local_pressure_and_crowding_are_pure_bounded_fields() -> None:
    cfg, state = make_state()
    food_before = state.food.copy()
    toxin_before = state.toxin.copy()
    pressure_food = compute_local_food_pressure(state, cfg)
    pressure_toxin = compute_local_toxin_pressure(state, cfg)
    crowding = compute_crowding(state)

    assert pressure_food.shape == state.health.shape
    assert pressure_toxin.shape == state.health.shape
    assert crowding.shape == state.health.shape
    assert np.all((pressure_food >= 0.0) & (pressure_food <= 1.0))
    assert np.all((pressure_toxin >= 0.0) & (pressure_toxin <= 1.0))
    assert np.all((crowding >= 0.0) & (crowding <= 1.0))
    assert np.array_equal(state.food, food_before)
    assert np.array_equal(state.toxin, toxin_before)


def test_signal_reception_uses_sensitivity_receptivity_trust_and_boundary() -> None:
    cfg, state = make_state()
    y, x = first_living_cell(state)
    state.signal.fill(0.0)
    state.signal[y, x, int(SignalChannel.FOOD)] = 1.0
    state.receive_sensitivity[y, x] = 1.0
    state.channel_receptivity[y, x, int(SignalChannel.FOOD)] = 1.0
    state.channel_trust_local[y, x, int(SignalChannel.FOOD)] = 1.0
    state.boundary[y, x] = 1.0

    compute_signal_reception(state, cfg)

    assert state.signal_reception.shape == state.signal.shape
    assert state.signal_reception[y, x, int(SignalChannel.FOOD)] > 0.0
    assert np.all((state.signal_reception >= 0.0) & (state.signal_reception <= 1.0))

    state.channel_trust_local[y, x, int(SignalChannel.FOOD)] = 0.0
    compute_signal_reception(state, cfg)
    assert state.signal_reception[y, x, int(SignalChannel.FOOD)] == 0.0


def test_automatic_signal_intents_cover_core_channels_and_choose_intent() -> None:
    cfg, state = make_state()
    y, x = first_living_cell(state)

    state.food[y, x] = 1.0
    state.grazing[y, x] = 1.0
    state.toxin[y, x] = 0.8
    state.health[y, x] = 0.2
    state.cooperation[y, x] = 1.0
    state.integration[y, x] = 1.0
    state.aggression[y, x] = 1.0
    state.emit_strength[y, x] = 1.0
    state.signal_precision[y, x] = 1.0
    state.channel_emission_bias[y, x, :] = 1.0

    intents = compute_automatic_signal_intents(state, cfg)
    chosen = choose_signal_intent(state, cfg)

    assert intents.shape == state.signal.shape
    assert np.all((intents >= 0.0) & (intents <= 1.0))
    assert intents[y, x, int(SignalChannel.FOOD)] > 0.0
    assert intents[y, x, int(SignalChannel.DANGER)] > 0.0
    assert intents[y, x, int(SignalChannel.COORDINATION)] > 0.0
    assert chosen.shape == state.health.shape
    assert chosen[y, x] in range(cfg.communication.num_channels)


def test_emit_signals_costs_resource_and_adds_emissions() -> None:
    cfg, state = make_state()
    y, x = first_living_cell(state)
    state.signal_emission.fill(0.0)
    state.food[y, x] = 1.0
    state.grazing[y, x] = 1.0
    state.emit_strength[y, x] = 1.0
    state.emit_efficiency[y, x] = 1.0
    state.signal_precision[y, x] = 1.0
    state.channel_emission_bias[y, x, int(SignalChannel.FOOD)] = 1.0
    state.resource[y, x] = 1.0

    before_resource = float(state.resource[y, x])
    emit_signals(state, cfg)

    assert state.signal_emission[y, x, int(SignalChannel.FOOD)] > 0.0
    assert state.resource[y, x] < before_resource
    assert np.all((state.signal_emission >= 0.0) & (state.signal_emission <= 1.0))
    assert np.all((state.resource >= 0.0) & (state.resource <= cfg.resources.max_resource))


def test_signal_memory_trust_novelty_and_conflict_are_bounded() -> None:
    cfg, state = make_state()
    y, x = first_living_cell(state)
    state.signal_reception.fill(0.0)
    state.signal_reception[y, x, int(SignalChannel.FOOD)] = 0.8
    state.signal_reception[y, x, int(SignalChannel.DANGER)] = 0.6

    novelty = compute_novelty(state, cfg)
    conflict = compute_signal_conflict(state, cfg)
    update_signal_memory(state, cfg)

    assert novelty.shape == state.health.shape
    assert conflict.shape == state.health.shape
    assert conflict[y, x] > 0.0
    assert state.signal_memory[y, x, int(SignalChannel.FOOD)] > 0.0
    assert np.all((novelty >= 0.0) & (novelty <= 1.0))
    assert np.all((conflict >= 0.0) & (conflict <= 1.0))
    assert np.all((state.signal_memory >= 0.0) & (state.signal_memory <= 1.0))

    prev_resource = state.resource.copy()
    prev_health = state.health.copy()
    prev_integration = state.integration.copy()
    state.resource[y, x] = min(1.0, state.resource[y, x] + 0.2)
    state.health[y, x] = min(1.0, state.health[y, x] + 0.1)

    before_trust = state.channel_trust_local[y, x, int(SignalChannel.FOOD)]
    update_channel_trust(state, prev_resource, prev_health, prev_integration, cfg)

    assert state.channel_trust_local[y, x, int(SignalChannel.FOOD)] >= before_trust
    assert np.all((state.channel_trust_local >= 0.0) & (state.channel_trust_local <= 1.0))


def test_disabled_communication_zeroes_reception_and_emission() -> None:
    cfg, state = make_state()
    cfg.communication.enabled = False
    state.signal.fill(1.0)
    state.signal_emission.fill(1.0)
    state.signal_reception.fill(1.0)

    update_signal_fields(state, cfg)
    compute_signal_reception(state, cfg)
    emit_signals(state, cfg)

    assert np.all(state.signal == 0.0)
    assert np.all(state.signal_emission == 0.0)
    assert np.all(state.signal_reception == 0.0)


# Loop communication integration tests.


def test_step_keeps_communication_fields_bounded_and_channel_compatible() -> None:
    cfg, state = make_state(seed=789)
    rng = np.random.default_rng(789)

    step(state, cfg, rng)

    assert state.signal.shape == (cfg.world.height, cfg.world.width, cfg.communication.num_channels)
    assert state.signal_reception.shape == state.signal.shape
    assert state.signal_memory.shape == state.signal.shape
    assert state.channel_trust_local.shape == state.signal.shape
    for field in (
        state.signal,
        state.signal_emission,
        state.signal_reception,
        state.signal_memory,
        state.channel_trust_local,
    ):
        assert np.all(np.isfinite(field))
        assert np.nanmin(field) >= -1e-6
        assert np.nanmax(field) <= 1.0 + 1e-6
