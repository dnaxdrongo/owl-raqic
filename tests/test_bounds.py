"""Bounds and state-shape tests."""

from __future__ import annotations

import numpy as np

from owl.core.actions import Action, SignalChannel
from owl.core.config import SimulationConfig, load_config
from owl.core.init import create_empty_global_state, create_empty_patch_state, initialize_world
from owl.core.state import (
    GlobalState,
    PatchState,
    WorldState,
    action_shape,
    channel_shape,
    clone_for_baseline,
    field_shape,
)
from owl.engine.actualization import actualize_actions
from owl.engine.aggregation import aggregate_patches, upsample_patch_bias
from owl.engine.authority import apply_enabled_action_mask, compute_authority
from owl.engine.integration import compute_conflict, update_integration
from owl.engine.loop import assert_invariants, run_headless
from owl.engine.phase import (
    compute_cell_coherence,
    compute_cross_scale_coupling,
    compute_local_synchrony,
    compute_meaning_alignment,
    update_phase,
)
from owl.engine.scheduler import should_record, should_update_global, should_update_patches
from owl.engine.topdown import compute_patch_intention, patch_policy_to_bias
from owl.engine.utility import (
    add_communication_utilities,
    add_movement_utilities,
    compute_internal_drives,
    compute_utilities,
)
from owl.kernels.numpy_kernels import (
    gradient_wrap,
    laplacian_wrap,
    neighbor_mean_wrap,
)
from owl.kernels.scipy_kernels import convolve_field, diffuse_with_obstacles, local_mean


def make_dummy_state(
    height: int = 10, width: int = 10, actions: int = 22, channels: int = 8
) -> WorldState:
    """Create a small valid WorldState without depending on later init code."""

    def f(shape):
        return np.zeros(shape, dtype=np.float32)

    def ones(shape):
        return np.ones(shape, dtype=np.float32)

    def ints(shape):
        return np.zeros(shape, dtype=np.int32)

    readouts = np.zeros((height, width), dtype=np.int16)

    patch_h = height // 5
    patch_w = width // 5
    patches = PatchState(
        activation=f((patch_h, patch_w)),
        memory=f((patch_h, patch_w)),
        phase=f((patch_h, patch_w)),
        possibility=ones((patch_h, patch_w, actions)) / actions,
        integration=f((patch_h, patch_w)),
        resource=f((patch_h, patch_w)),
        health=f((patch_h, patch_w)),
        boundary=f((patch_h, patch_w)),
        signal_pressure=f((patch_h, patch_w, channels)),
        synchrony=f((patch_h, patch_w)),
        coherence=f((patch_h, patch_w)),
        cross_scale=f((patch_h, patch_w)),
        intention=ints((patch_h, patch_w)),
        policy_bias=f((patch_h, patch_w, actions)),
    )
    global_state = GlobalState(
        integration=0.0,
        readout=int(Action.REST),
        intention=0,
        fragmentation=0.0,
        diversity=0.0,
        complexity=0.0,
        signal_pressure=f((channels,)),
        policy_bias=f((actions,)),
    )

    possibility = ones((height, width, actions)) / actions
    return WorldState(
        activation=f((height, width)),
        memory=f((height, width)),
        phase=f((height, width)),
        threshold=f((height, width)),
        readout=readouts,
        integration=f((height, width)),
        resource=ones((height, width)),
        health=ones((height, width)),
        boundary=ones((height, width)),
        age=ints((height, width)),
        ow_type=ints((height, width)),
        lineage_id=ints((height, width)),
        parent_id=ints((height, width)),
        possibility=possibility.astype(np.float32),
        signal=f((height, width, channels)),
        signal_emission=f((height, width, channels)),
        signal_reception=f((height, width, channels)),
        signal_memory=f((height, width, channels)),
        channel_receptivity=ones((height, width, channels)),
        channel_emission_bias=ones((height, width, channels)),
        channel_trust_local=ones((height, width, channels)),
        food=f((height, width)),
        toxin=f((height, width)),
        obstacle=np.zeros((height, width), dtype=bool),
        occupancy=np.full((height, width), -1, dtype=np.int32),
        noise=f((height, width)),
        mobility=ones((height, width)),
        metabolism=ones((height, width)),
        predation=f((height, width)),
        grazing=ones((height, width)),
        cooperation=f((height, width)),
        aggression=f((height, width)),
        curiosity=f((height, width)),
        reproduction_rate=f((height, width)),
        toxin_resistance=f((height, width)),
        memory_capacity=ones((height, width)),
        coupling_strength=ones((height, width)),
        emit_strength=ones((height, width)),
        emit_efficiency=ones((height, width)),
        receive_sensitivity=ones((height, width)),
        signal_precision=ones((height, width)),
        honesty_bias=ones((height, width)),
        deception_bias=f((height, width)),
        patches=patches,
        global_state=global_state,
    )


def test_core_enums_have_expected_indices() -> None:
    assert Action.REST == 0
    assert SignalChannel.FOOD == 0
    assert SignalChannel.INTEGRATION == 7


def test_shape_helpers_return_expected_shapes() -> None:
    state = make_dummy_state()
    assert field_shape(state) == (10, 10)
    assert action_shape(state) == (10, 10, 22)
    assert channel_shape(state) == (10, 10, 8)


def test_clone_for_baseline_deep_copies_arrays() -> None:
    state = make_dummy_state()
    cloned = clone_for_baseline(state)

    assert cloned is not state
    assert cloned.health is not state.health
    assert np.array_equal(cloned.health, state.health)

    cloned.health[0, 0] = 0.25
    assert state.health[0, 0] == 1.0


def test_mvp_config_loads_and_validates() -> None:
    cfg = load_config("configs/mvp.yaml")
    assert isinstance(cfg, SimulationConfig)
    assert cfg.world.height == 50
    assert cfg.world.width == 50
    assert cfg.world.height % cfg.world.patch_size == 0
    assert cfg.communication.num_channels == len(cfg.communication.diffusion)
    assert cfg.communication.num_channels == len(cfg.communication.decay)


def test_config_rejects_bad_channel_lengths() -> None:
    data = load_config("configs/mvp.yaml").model_dump()
    data["communication"]["diffusion"] = [0.1]
    try:
        SimulationConfig.model_validate(data)
    except ValueError as exc:
        assert "communication.diffusion length" in str(exc)
    else:
        raise AssertionError("bad communication.diffusion length should fail validation")


def test_laplacian_wrap_preserves_shape_and_supports_channels() -> None:
    field = np.zeros((3, 3), dtype=np.float32)
    field[1, 1] = 1.0

    lap = laplacian_wrap(field)

    assert lap.shape == field.shape
    assert lap.dtype == np.float32
    assert np.isclose(lap[1, 1], -4.0)
    assert np.isclose(lap[0, 1], 1.0)
    assert np.isclose(lap[1, 0], 1.0)

    channel_field = np.stack([field, 2.0 * field], axis=-1)
    channel_lap = laplacian_wrap(channel_field)
    assert channel_lap.shape == channel_field.shape
    assert np.isclose(channel_lap[1, 1, 1], -8.0)


def test_neighbor_mean_wrap_preserves_shape_and_excludes_center() -> None:
    field = np.zeros((3, 3), dtype=np.float32)
    field[1, 1] = 1.0

    mean = neighbor_mean_wrap(field)

    assert mean.shape == field.shape
    assert np.isclose(mean[1, 1], 0.0)
    assert np.isclose(mean[0, 0], 1.0 / 8.0)
    assert np.isclose(mean.sum(), 1.0)


def test_gradient_wrap_returns_y_and_x_components() -> None:
    field = np.tile(np.arange(5, dtype=np.float32), (5, 1))
    grad_y, grad_x = gradient_wrap(field)

    assert grad_y.shape == field.shape
    assert grad_x.shape == field.shape
    assert np.allclose(grad_y, 0.0)
    assert np.isclose(grad_x[2, 2], 1.0)


def test_scipy_convolve_field_identity_and_local_mean() -> None:
    field = np.arange(9, dtype=np.float32).reshape(3, 3)
    identity = np.zeros((3, 3), dtype=np.float32)
    identity[1, 1] = 1.0

    out = convolve_field(field, identity, mode="reflect")
    assert out.shape == field.shape
    assert out.dtype == np.float32
    assert np.array_equal(out, field)

    ones = np.ones((5, 5), dtype=np.float32)
    mean = local_mean(ones, radius=1, mode="reflect")
    assert mean.shape == ones.shape
    assert mean.dtype == np.float32
    assert np.allclose(mean, 1.0)


def test_diffuse_with_obstacles_freezes_obstacle_cells() -> None:
    field = np.zeros((5, 5), dtype=np.float32)
    field[2, 2] = 1.0
    obstacle = np.zeros_like(field, dtype=bool)
    obstacle[2, 2] = True

    out = diffuse_with_obstacles(field, obstacle, rate=0.1, mode="reflect")

    assert out.shape == field.shape
    assert out.dtype == np.float32
    assert np.isclose(out[2, 2], 1.0)
    assert np.all(np.isfinite(out))


def test_create_empty_patch_and_global_state_shapes() -> None:
    cfg = load_config("configs/mvp.yaml")
    patches = create_empty_patch_state(cfg)
    global_state = create_empty_global_state(cfg)

    patch_shape = (
        cfg.world.height // cfg.world.patch_size,
        cfg.world.width // cfg.world.patch_size,
    )
    assert patches.integration.shape == patch_shape
    assert patches.possibility.shape == (*patch_shape, len(Action))
    assert patches.signal_pressure.shape == (*patch_shape, cfg.communication.num_channels)
    assert np.allclose(patches.possibility.sum(axis=-1), 1.0)

    assert global_state.signal_pressure.shape == (cfg.communication.num_channels,)
    assert global_state.policy_bias.shape == (len(Action),)


def test_initialize_world_returns_valid_bounded_state() -> None:
    cfg = load_config("configs/mvp.yaml")
    rng = np.random.default_rng(123)
    state = initialize_world(cfg, rng)

    assert field_shape(state) == (cfg.world.height, cfg.world.width)
    assert action_shape(state) == (cfg.world.height, cfg.world.width, len(Action))
    assert channel_shape(state) == (
        cfg.world.height,
        cfg.world.width,
        cfg.communication.num_channels,
    )
    assert state.patches.integration.shape == (
        cfg.world.height // cfg.world.patch_size,
        cfg.world.width // cfg.world.patch_size,
    )
    assert state.global_state.policy_bias.shape == (len(Action),)

    bounded_names = [
        "activation",
        "memory",
        "integration",
        "resource",
        "health",
        "boundary",
        "food",
        "toxin",
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
        "signal_precision",
        "honesty_bias",
        "deception_bias",
    ]
    for name in bounded_names:
        arr = getattr(state, name)
        assert np.all(np.isfinite(arr)), name
        assert np.all(arr >= 0.0), name
        assert np.all(arr <= 1.0), name

    assert np.allclose(state.possibility.sum(axis=-1), 1.0, atol=1e-6)

    living = state.health > 0.0
    assert living.any()
    assert np.all(state.lineage_id[living] >= 0)
    assert np.all(state.occupancy[living] >= 0)
    assert np.all(state.parent_id[living] >= 0)

    dead = ~living
    if dead.any():
        assert np.all(state.readout[dead] == int(Action.REST))
        assert np.allclose(state.possibility[dead, int(Action.REST)], 1.0)


def test_initialize_world_is_deterministic_for_same_seed() -> None:
    cfg = load_config("configs/mvp.yaml")
    state_a = initialize_world(cfg, np.random.default_rng(cfg.world.seed))
    state_b = initialize_world(cfg, np.random.default_rng(cfg.world.seed))

    for name in [
        "health",
        "resource",
        "boundary",
        "food",
        "phase",
        "ow_type",
        "channel_emission_bias",
    ]:
        assert np.array_equal(getattr(state_a, name), getattr(state_b, name)), name


def test_all_config_presets_load_and_initialize() -> None:
    for name in [
        "default.yaml",
        "mvp.yaml",
        "communication.yaml",
        "carnivore_world.yaml",
        "scarce_food.yaml",
        "high_noise.yaml",
        "overcontrolled.yaml",
        "rivalry.yaml",
        "fragmented.yaml",
    ]:
        cfg = load_config(f"configs/{name}")
        state = initialize_world(cfg, np.random.default_rng(1))
        assert field_shape(state) == (cfg.world.height, cfg.world.width)
        assert np.allclose(state.possibility.sum(axis=-1), 1.0, atol=1e-6)


def test_phase_update_and_diagnostics_are_bounded_and_shape_safe() -> None:
    cfg = load_config("configs/mvp.yaml")
    rng = np.random.default_rng(123)
    state = initialize_world(cfg, rng)
    parent_phase = np.zeros(field_shape(state), dtype=np.float32)

    old_phase = state.phase.copy()
    old_health = state.health.copy()

    update_phase(state, parent_phase, np.random.default_rng(456), cfg)

    assert state.phase.shape == old_phase.shape
    assert state.phase.dtype == np.float32
    assert np.all(np.isfinite(state.phase))
    assert np.all(state.phase >= 0.0)
    assert np.all(state.phase < 2.0 * np.pi)

    # Phase update should not mutate physical viability fields.
    assert np.array_equal(state.health, old_health)

    synchrony = compute_local_synchrony(state, cfg)
    coherence = compute_cell_coherence(state, cfg)
    cross_scale = compute_cross_scale_coupling(state, parent_phase, cfg)
    alignment = compute_meaning_alignment(state, coherence, cross_scale, cfg)

    for name, arr in {
        "synchrony": synchrony,
        "coherence": coherence,
        "cross_scale": cross_scale,
        "alignment": alignment,
    }.items():
        assert arr.shape == field_shape(state), name
        assert arr.dtype == np.float32, name
        assert np.all(np.isfinite(arr)), name
        assert np.all(arr >= 0.0), name
        assert np.all(arr <= 1.0), name


def test_cross_scale_and_phase_functions_reject_bad_parent_shape() -> None:
    cfg = load_config("configs/mvp.yaml")
    state = initialize_world(cfg, np.random.default_rng(123))
    bad_parent = np.zeros((3, 7), dtype=np.float32)

    for func in (
        lambda: update_phase(state, bad_parent, np.random.default_rng(1), cfg),
        lambda: compute_cross_scale_coupling(state, bad_parent, cfg),
    ):
        try:
            func()
        except ValueError as exc:
            assert "parent_phase" in str(exc)
        else:
            raise AssertionError("bad parent_phase shape should fail")


def test_conflict_and_integration_update_are_bounded() -> None:
    cfg = load_config("configs/mvp.yaml")
    state = initialize_world(cfg, np.random.default_rng(321))
    h, w = field_shape(state)
    parent_bias = np.zeros((h, w, len(Action)), dtype=np.float32)
    parent_bias[..., Action.INTEGRATE] = 0.5

    # Create communication conflict: food and danger signals together.
    state.signal_reception[..., SignalChannel.FOOD] = 1.0
    state.signal_reception[..., SignalChannel.DANGER] = 1.0

    synchrony = compute_local_synchrony(state, cfg)
    coherence = compute_cell_coherence(state, cfg)
    cross_scale = compute_cross_scale_coupling(state, np.zeros((h, w), dtype=np.float32), cfg)
    conflict = compute_conflict(state, parent_bias, cfg)

    assert conflict.shape == (h, w)
    assert conflict.dtype == np.float32
    assert np.all(conflict >= 0.0)
    assert np.all(conflict <= 1.0)
    assert conflict.mean() > 0.0

    update_integration(state, synchrony, coherence, cross_scale, conflict, cfg)
    assert state.integration.shape == (h, w)
    assert state.integration.dtype == np.float32
    assert np.all(np.isfinite(state.integration))
    assert np.all(state.integration >= 0.0)
    assert np.all(state.integration <= 1.0)


def test_integration_update_sets_dead_cells_to_zero() -> None:
    state = make_dummy_state()
    cfg = load_config("configs/mvp.yaml")
    h, w = field_shape(state)
    state.health[0, 0] = 0.0
    state.integration[0, 0] = 1.0

    ones = np.ones((h, w), dtype=np.float32)
    parent_bias = np.zeros((h, w, len(Action)), dtype=np.float32)
    conflict = compute_conflict(state, parent_bias, cfg)
    update_integration(state, ones, ones, ones, conflict, cfg)

    assert state.integration[0, 0] == 0.0


# Bounds tests for utility, authority, and actualization.


def test_pass06_authority_masks_are_bounded_and_dead_cells_rest_only() -> None:
    cfg = load_config("configs/mvp.yaml")
    data = cfg.model_dump()
    data["world"]["height"] = 20
    data["world"]["width"] = 20
    data["world"]["patch_size"] = 5
    data["initialization"]["population_density"] = 0.5
    data["visualization"]["enabled"] = False
    data["recording"]["enabled"] = False
    cfg = SimulationConfig.model_validate(data)
    rng = np.random.default_rng(321)
    state = initialize_world(cfg, rng)

    state.health[0, 0] = 0.0
    authority = compute_authority(state, cfg)

    assert authority.shape == state.possibility.shape
    assert np.all(np.isfinite(authority))
    assert np.all((authority >= 0.0) & (authority <= 1.0))
    assert authority[0, 0, int(Action.REST)] == 1.0
    assert np.sum(authority[0, 0]) == 1.0


def test_pass06_enabled_action_mask_rejects_unknown_actions() -> None:
    cfg = load_config("configs/mvp.yaml")
    authority = np.ones((2, 2, len(Action)), dtype=np.float32)
    cfg.actions.enabled_actions = ["REST", "NOT_A_REAL_ACTION"]

    try:
        apply_enabled_action_mask(authority, cfg)
    except ValueError as exc:
        assert "unknown enabled action" in str(exc)
    else:
        raise AssertionError("unknown action should fail")


def test_pass06_utilities_are_finite_and_do_not_mutate_state() -> None:
    cfg = load_config("configs/mvp.yaml")
    data = cfg.model_dump()
    data["world"]["height"] = 20
    data["world"]["width"] = 20
    data["world"]["patch_size"] = 5
    data["initialization"]["population_density"] = 0.5
    data["visualization"]["enabled"] = False
    data["recording"]["enabled"] = False
    cfg = SimulationConfig.model_validate(data)
    rng = np.random.default_rng(123)
    state = initialize_world(cfg, rng)
    patches = aggregate_patches(state, cfg)
    compute_patch_intention(patches, cfg)
    parent_bias = upsample_patch_bias(patch_policy_to_bias(patches, cfg), cfg.world.patch_size)

    before_readout = state.readout.copy()
    before_possibility = state.possibility.copy()

    drives = compute_internal_drives(state, cfg)
    utilities = compute_utilities(state, parent_bias, cfg)
    utilities2 = add_movement_utilities(np.zeros_like(utilities), state, drives, cfg)
    utilities3 = add_communication_utilities(np.zeros_like(utilities), state, drives, cfg)

    assert utilities.shape == state.possibility.shape
    assert utilities2.shape == state.possibility.shape
    assert utilities3.shape == state.possibility.shape
    assert np.all(np.isfinite(utilities))
    assert np.array_equal(state.readout, before_readout)
    assert np.array_equal(state.possibility, before_possibility)


def test_pass06_actualization_keeps_probability_bounds() -> None:
    cfg = load_config("configs/mvp.yaml")
    data = cfg.model_dump()
    data["world"]["height"] = 20
    data["world"]["width"] = 20
    data["world"]["patch_size"] = 5
    data["initialization"]["population_density"] = 0.5
    data["visualization"]["enabled"] = False
    data["recording"]["enabled"] = False
    cfg = SimulationConfig.model_validate(data)
    rng = np.random.default_rng(55)
    state = initialize_world(cfg, rng)
    patches = aggregate_patches(state, cfg)
    compute_patch_intention(patches, cfg)
    parent_bias = upsample_patch_bias(patch_policy_to_bias(patches, cfg), cfg.world.patch_size)

    utilities = compute_utilities(state, parent_bias, cfg)
    authority = compute_authority(state, cfg)
    actualize_actions(state, utilities, authority, parent_bias, rng, cfg)

    assert np.all(np.isfinite(state.possibility))
    assert np.all((state.possibility >= 0.0) & (state.possibility <= 1.0))
    alive = (state.health > 0.0) & (~state.obstacle)
    assert np.allclose(state.possibility[alive].sum(axis=-1), 1.0, atol=1e-6)
    assert np.all((state.readout >= 0) & (state.readout < len(Action)))


# Scheduler and bounded-run loop tests.


def make_pass10_cfg(height: int = 20, width: int = 20, steps: int = 20) -> SimulationConfig:
    data = load_config("configs/mvp.yaml").model_dump()
    data["world"]["height"] = height
    data["world"]["width"] = width
    data["world"]["patch_size"] = 5
    data["world"]["max_steps"] = steps
    data["initialization"]["population_density"] = 0.35
    data["initialization"]["food_patch_count"] = 2
    data["initialization"]["food_patch_radius"] = 3
    data["visualization"]["enabled"] = False
    data["recording"]["enabled"] = False
    data["debug"]["assert_invariants"] = True
    return SimulationConfig.model_validate(data)


def test_scheduler_cadences_are_deterministic_and_config_driven() -> None:
    cfg = make_pass10_cfg()
    data = cfg.model_dump()
    data["recording"]["enabled"] = True
    data["recording"]["record_every"] = 5
    data["topdown"]["apex_update_every"] = 4
    cfg = SimulationConfig.model_validate(data)

    assert should_update_patches(0, cfg)
    assert should_update_patches(7, cfg)
    assert should_update_global(0, cfg)
    assert should_update_global(4, cfg)
    assert not should_update_global(5, cfg)
    assert should_record(0, cfg)
    assert should_record(10, cfg)
    assert not should_record(11, cfg)


def test_run_headless_50_by_50_completes_100_ticks_with_bounded_fields() -> None:
    cfg = make_pass10_cfg(height=50, width=50, steps=100)
    state, metrics = run_headless(cfg, max_steps=100)

    assert state.tick == 100
    assert len(metrics) == 100
    assert metrics[-1]["tick"] == 100
    for field in (
        state.activation,
        state.memory,
        state.integration,
        state.resource,
        state.health,
        state.boundary,
        state.food,
        state.toxin,
        state.signal,
        state.signal_reception,
        state.signal_memory,
        state.channel_trust_local,
    ):
        assert np.all(np.isfinite(field))
        assert np.nanmin(field) >= -1e-6
        assert np.nanmax(field) <= 1.0 + 1e-6

    assert state.patches.integration.shape == (10, 10)
    assert state.global_state.policy_bias.shape == (len(Action),)
    assert_invariants(state, cfg)


def test_assert_invariants_helper_accepts_valid_completed_state_and_rejects_drift() -> None:
    cfg = make_pass10_cfg(height=20, width=20, steps=3)
    state, _metrics = run_headless(cfg, max_steps=3)

    assert_invariants(state, cfg)

    state.possibility[0, 0, :] = 0.0
    state.health[0, 0] = 1.0
    try:
        assert_invariants(state, cfg)
    except AssertionError as exc:
        assert "possibility" in str(exc)
    else:
        raise AssertionError("invalid possibility simplex should fail invariant check")
