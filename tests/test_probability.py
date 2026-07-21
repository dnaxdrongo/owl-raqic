"""Probability, actualization, and numerical-kernel tests."""

from __future__ import annotations

import numpy as np

from owl.core.actions import Action, SignalChannel
from owl.core.config import SimulationConfig, load_config
from owl.core.init import initialize_world
from owl.core.state import clone_for_baseline
from owl.engine.actualization import (
    actualize_actions,
    compute_action_logits,
    deterministic_actions,
    sample_actions,
)
from owl.engine.aggregation import aggregate_global, aggregate_patches, upsample_patch_bias
from owl.engine.authority import compute_authority
from owl.engine.integration import entropy_normalized, possibility_flexibility
from owl.engine.loop import run_headless
from owl.engine.topdown import compute_patch_intention, patch_policy_to_bias
from owl.engine.utility import compute_internal_drives, compute_utilities
from owl.kernels.numba_kernels import (
    collision_scan_kernel,
    ingestion_attempt_kernel,
    move_cells_kernel,
    sample_categorical_grid,
)
from owl.kernels.numpy_kernels import (
    circular_mean,
    normalize_last_axis,
    sigmoid,
    softmax_stable,
)
from tests.test_bounds import make_dummy_state


def test_dummy_state_possibilities_are_normalized() -> None:
    state = make_dummy_state()
    assert np.allclose(state.possibility.sum(axis=-1), 1.0)
    assert np.all(state.possibility >= 0.0)


def test_normalize_last_axis_projects_to_simplex_and_repairs_zero_rows() -> None:
    values = np.array(
        [
            [[1.0, 1.0, 2.0], [-1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )

    out = normalize_last_axis(values)

    assert out.shape == values.shape
    assert out.dtype == np.float32
    assert np.all(out >= 0.0)
    assert np.allclose(out.sum(axis=-1), 1.0)
    assert np.allclose(out[1, 0], np.full(3, 1.0 / 3.0, dtype=np.float32))


def test_softmax_stable_handles_large_logits_and_normalizes() -> None:
    logits = np.array([[1000.0, 1001.0, 1002.0], [-1000.0, -999.0, -998.0]], dtype=np.float32)
    probs = softmax_stable(logits, axis=-1)

    assert probs.shape == logits.shape
    assert probs.dtype == np.float32
    assert np.all(np.isfinite(probs))
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    assert np.allclose(probs.sum(axis=-1), 1.0, atol=1e-6)
    assert np.argmax(probs[0]) == 2
    assert np.argmax(probs[1]) == 2


def test_sigmoid_is_bounded_and_stable_for_extreme_values() -> None:
    x = np.array([-1000.0, -10.0, 0.0, 10.0, 1000.0], dtype=np.float64)
    y = sigmoid(x)

    assert y.shape == x.shape
    assert np.all(np.isfinite(y))
    assert np.all(y >= 0.0)
    assert np.all(y <= 1.0)
    assert np.isclose(y[2], 0.5)
    assert sigmoid(0.0) == 0.5


def test_circular_mean_handles_phase_wraparound() -> None:
    phases = np.array([0.0, 2.0 * np.pi], dtype=np.float64)
    mean = circular_mean(phases)

    assert np.isclose(mean, 0.0, atol=1e-12)

    two_by_two = np.array([[0.0, np.pi / 2], [np.pi, 3 * np.pi / 2]])
    row_means = circular_mean(two_by_two, axis=1)
    assert row_means.shape == (2,)
    assert np.all(np.isfinite(row_means))


def test_sample_categorical_grid_is_deterministic_for_fixed_random_values() -> None:
    probabilities = np.array(
        [
            [[0.2, 0.3, 0.5], [1.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.25, 0.25, 0.5]],
        ],
        dtype=np.float32,
    )
    random_values = np.array([[0.1, 0.99], [0.5, 0.75]], dtype=np.float32)

    readout = sample_categorical_grid(probabilities, random_values)

    assert readout.shape == probabilities.shape[:2]
    assert readout.dtype == np.int16
    assert np.array_equal(readout, np.array([[0, 0], [1, 2]], dtype=np.int16))


def test_move_cells_kernel_moves_2d_and_3d_fields_without_mutating_input() -> None:
    field_2d = np.arange(9, dtype=np.float32).reshape(3, 3)
    moved_2d = move_cells_kernel(
        field_2d,
        source_y=np.array([0]),
        source_x=np.array([0]),
        target_y=np.array([2]),
        target_x=np.array([2]),
        clear_value=-1.0,
    )

    assert field_2d[0, 0] == 0.0
    assert moved_2d[2, 2] == 0.0
    assert moved_2d[0, 0] == -1.0
    assert moved_2d.dtype == field_2d.dtype

    field_3d = np.arange(18, dtype=np.float32).reshape(3, 3, 2)
    moved_3d = move_cells_kernel(field_3d, [1], [1], [0], [2], clear_value=0.0)

    assert np.array_equal(moved_3d[0, 2], field_3d[1, 1])
    assert np.array_equal(moved_3d[1, 1], np.zeros(2, dtype=np.float32))
    assert np.array_equal(field_3d[1, 1], np.array([8.0, 9.0], dtype=np.float32))


def test_collision_scan_kernel_detects_occupied_targets() -> None:
    occupied = np.zeros((4, 4), dtype=bool)
    occupied[1, 1] = True
    occupied[3, 2] = True

    collisions = collision_scan_kernel(
        target_y=np.array([1, 2, 3]),
        target_x=np.array([1, 2, 2]),
        occupied=occupied,
    )

    assert collisions.dtype == np.bool_
    assert np.array_equal(collisions, np.array([True, False, True]))


def test_ingestion_attempt_kernel_returns_bounded_advantage_probabilities() -> None:
    shape = (3, 3)
    predation = np.zeros(shape, dtype=np.float32)
    integration = np.zeros(shape, dtype=np.float32)
    resource = np.zeros(shape, dtype=np.float32)
    aggression = np.zeros(shape, dtype=np.float32)
    health = np.ones(shape, dtype=np.float32) * 0.5
    boundary = np.ones(shape, dtype=np.float32) * 0.5

    predation[0, 0] = 1.0
    integration[0, 0] = 1.0
    resource[0, 0] = 1.0
    aggression[0, 0] = 1.0

    probs = ingestion_attempt_kernel(
        predation,
        integration,
        resource,
        aggression,
        health,
        boundary,
        predator_y=np.array([0, 2]),
        predator_x=np.array([0, 2]),
        target_y=np.array([1, 1]),
        target_x=np.array([1, 1]),
    )

    assert probs.shape == (2,)
    assert probs.dtype == np.float32
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    assert probs[0] > probs[1]


def test_entropy_normalized_returns_expected_bounds_for_simple_distributions() -> None:
    deterministic = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32)
    uniform = np.array([[[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]]], dtype=np.float32)

    h_det = entropy_normalized(deterministic)
    h_uni = entropy_normalized(uniform)

    assert h_det.shape == (1, 1)
    assert h_det.dtype == np.float32
    assert np.allclose(h_det, 0.0, atol=1e-6)
    assert np.allclose(h_uni, 1.0, atol=1e-5)
    assert np.all(h_uni >= 0.0)
    assert np.all(h_uni <= 1.0)


def test_entropy_normalized_repairs_non_normalized_inputs() -> None:
    raw = np.array([[[2.0, 2.0], [0.0, 0.0]]], dtype=np.float32)
    h = entropy_normalized(raw)

    assert h.shape == raw.shape[:-1]
    assert np.all(np.isfinite(h))
    assert np.all(h >= 0.0)
    assert np.all(h <= 1.0)
    assert np.allclose(h, 1.0, atol=1e-5)


def test_possibility_flexibility_peaks_near_target_entropy() -> None:
    uniform = np.array([[[0.25, 0.25, 0.25, 0.25]]], dtype=np.float32)
    deterministic = np.array([[[1.0, 0.0, 0.0, 0.0]]], dtype=np.float32)

    flex_uniform = possibility_flexibility(uniform, target=1.0, sigma=0.2)
    flex_deterministic = possibility_flexibility(deterministic, target=1.0, sigma=0.2)

    assert flex_uniform.shape == (1, 1)
    assert flex_uniform.dtype == np.float32
    assert np.all(flex_uniform > flex_deterministic)
    assert np.all(flex_uniform <= 1.0)
    assert np.all(flex_deterministic >= 0.0)


def test_possibility_flexibility_rejects_bad_parameters() -> None:
    P = np.ones((2, 2, 3), dtype=np.float32) / 3.0

    for kwargs in ({"target": -0.1, "sigma": 0.2}, {"target": 0.5, "sigma": 0.0}):
        try:
            possibility_flexibility(P, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"bad flexibility parameters should fail: {kwargs}")


# Utility, authority, and actualization tests.


def make_pass06_cfg(height: int = 20, width: int = 20) -> SimulationConfig:
    data = load_config("configs/mvp.yaml").model_dump()
    data["world"]["height"] = height
    data["world"]["width"] = width
    data["world"]["patch_size"] = 5
    data["initialization"]["population_density"] = 0.65
    data["initialization"]["food_patch_count"] = 2
    data["initialization"]["food_patch_radius"] = 3
    data["visualization"]["enabled"] = False
    data["recording"]["enabled"] = False
    return SimulationConfig.model_validate(data)


def make_pass06_state(seed: int = 123):
    cfg = make_pass06_cfg()
    rng = np.random.default_rng(seed)
    state = initialize_world(cfg, rng)
    patches = aggregate_patches(state, cfg)
    compute_patch_intention(patches, cfg)
    parent_bias = upsample_patch_bias(patch_policy_to_bias(patches, cfg), cfg.world.patch_size)
    state.patches = patches
    state.global_state = aggregate_global(patches, cfg)
    return cfg, state, rng, parent_bias


def test_compute_internal_drives_returns_bounded_cell_fields() -> None:
    cfg, state, _, _ = make_pass06_state()
    drives = compute_internal_drives(state, cfg)

    expected_keys = {
        "hunger",
        "pain",
        "boundary_stress",
        "crowding",
        "food_pressure",
        "toxin_pressure",
        "novelty",
        "social_need",
    }
    assert expected_keys <= set(drives)
    for key in expected_keys:
        assert drives[key].shape == state.health.shape
        assert drives[key].dtype == np.float32
        assert np.all(np.isfinite(drives[key]))
        assert np.all((drives[key] >= 0.0) & (drives[key] <= 1.0))


def test_compute_utilities_authority_logits_and_deterministic_actualization() -> None:
    cfg, state, rng, parent_bias = make_pass06_state()
    cfg.actions.stochastic = False
    state.signal_reception[..., int(SignalChannel.FOOD)] = 0.3
    state.signal_reception[..., int(SignalChannel.DANGER)] = 0.1

    utilities = compute_utilities(state, parent_bias, cfg)
    authority = compute_authority(state, cfg)
    logits = compute_action_logits(state, utilities, authority, parent_bias, cfg)

    assert utilities.shape == state.possibility.shape
    assert authority.shape == state.possibility.shape
    assert logits.shape == state.possibility.shape
    assert utilities.dtype == np.float32
    assert authority.dtype == np.float32
    assert logits.dtype == np.float32
    assert np.all(np.isfinite(utilities))
    assert np.all(np.isfinite(authority))
    assert np.all(np.isfinite(logits))
    assert np.all((authority >= 0.0) & (authority <= 1.0))

    actualize_actions(state, utilities, authority, parent_bias, rng, cfg)

    alive = (state.health > 0.0) & (~state.obstacle)
    assert np.allclose(state.possibility[alive].sum(axis=-1), 1.0, atol=1e-6)
    assert np.all(state.possibility >= 0.0)
    assert state.readout.shape == state.health.shape
    assert state.readout.dtype == np.int16
    assert np.all((state.readout >= 0) & (state.readout < len(Action)))
    assert np.array_equal(state.readout, deterministic_actions(state.possibility))


def test_stochastic_macro_movement_is_utility_weighted_not_argmax_only() -> None:
    cfg, state, rng, parent_bias = make_pass06_state()
    cfg.actions.stochastic = True
    cfg.actions.movement_macro_enabled = True
    cfg.actions.diagonal_movement_enabled = True
    cfg.actions.action_temperature = 0.50
    cfg.actions.movement_temperature = 0.65
    cfg.actions.enabled_actions = [
        "REST",
        "MOVE_N",
        "MOVE_S",
        "MOVE_E",
        "MOVE_W",
        "MOVE_NE",
        "MOVE_NW",
        "MOVE_SE",
        "MOVE_SW",
        "INTEGRATE",
    ]

    utilities = np.full_like(state.possibility, -10.0, dtype=np.float32)
    authority = np.zeros_like(state.possibility, dtype=np.float32)
    parent_bias = np.zeros_like(state.possibility, dtype=np.float32)
    alive = (state.health > 0.0) & (~state.obstacle)
    for action in (
        Action.REST,
        Action.INTEGRATE,
        Action.MOVE_N,
        Action.MOVE_S,
        Action.MOVE_E,
        Action.MOVE_W,
    ):
        authority[..., int(action)] = alive.astype(np.float32)
    utilities[..., int(Action.INTEGRATE)] = 0.10
    for action in (Action.MOVE_N, Action.MOVE_S, Action.MOVE_E, Action.MOVE_W):
        utilities[..., int(action)] = 0.08

    from owl.engine.actualization import actualize_actions

    counts = []
    for seed in range(25):
        test_state = clone_for_baseline(state)
        actualize_actions(
            test_state, utilities, authority, parent_bias, np.random.default_rng(seed), cfg
        )
        move_count = sum(
            np.count_nonzero((test_state.readout == int(a)) & alive)
            for a in (Action.MOVE_N, Action.MOVE_S, Action.MOVE_E, Action.MOVE_W)
        )
        counts.append(move_count / max(int(np.count_nonzero(alive)), 1))
    assert float(np.mean(counts)) > 0.15


def test_actualization_stochastic_sampling_is_seeded_and_normalized() -> None:
    cfg, state, _, parent_bias = make_pass06_state()
    cfg.actions.stochastic = True
    rng1 = np.random.default_rng(99)
    rng2 = np.random.default_rng(99)

    utilities = compute_utilities(state, parent_bias, cfg)
    authority = compute_authority(state, cfg)

    actualize_actions(state, utilities, authority, parent_bias, rng1, cfg)
    readout1 = state.readout.copy()

    # Recompute on a fresh equivalent state and seed to test deterministic replay.
    cfg2, state2, _, parent_bias2 = make_pass06_state()
    cfg2.actions.stochastic = True
    utilities2 = compute_utilities(state2, parent_bias2, cfg2)
    authority2 = compute_authority(state2, cfg2)
    actualize_actions(state2, utilities2, authority2, parent_bias2, rng2, cfg2)

    assert np.array_equal(readout1, state2.readout)
    assert np.allclose(state.possibility.sum(axis=-1), 1.0, atol=1e-6)


def test_sample_actions_and_deterministic_actions_validate_shapes() -> None:
    rng = np.random.default_rng(7)
    probabilities = np.zeros((2, 3, len(Action)), dtype=np.float32)
    probabilities[..., int(Action.FEED)] = 1.0

    sampled = sample_actions(probabilities, rng)
    deterministic = deterministic_actions(probabilities)

    assert sampled.shape == (2, 3)
    assert deterministic.shape == (2, 3)
    assert sampled.dtype == np.int16
    assert deterministic.dtype == np.int16
    assert np.all(sampled == int(Action.FEED))
    assert np.all(deterministic == int(Action.FEED))

    bad = np.ones((2, 3), dtype=np.float32)
    try:
        deterministic_actions(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("2D probability array should fail")


def test_dead_and_obstacle_cells_are_forced_to_rest() -> None:
    cfg, state, rng, parent_bias = make_pass06_state()
    state.health[0, 0] = 0.0
    state.obstacle[0, 1] = True

    utilities = compute_utilities(state, parent_bias, cfg)
    authority = compute_authority(state, cfg)
    actualize_actions(state, utilities, authority, parent_bias, rng, cfg)

    assert authority[0, 0, int(Action.REST)] == 1.0
    assert np.all(authority[0, 0, np.arange(len(Action)) != int(Action.REST)] == 0.0)
    assert state.readout[0, 0] == int(Action.REST)
    assert state.readout[0, 1] == int(Action.REST)
    assert state.possibility[0, 0, int(Action.REST)] == 1.0
    assert state.possibility[0, 1, int(Action.REST)] == 1.0


def test_enabled_action_mask_suppresses_disabled_actions_in_actualization() -> None:
    cfg, state, rng, parent_bias = make_pass06_state()
    cfg.actions.enabled_actions = ["REST", "FEED"]

    utilities = np.zeros_like(state.possibility, dtype=np.float32)
    utilities[..., int(Action.INGEST)] = 100.0
    utilities[..., int(Action.FEED)] = 1.0

    authority = compute_authority(state, cfg)
    actualize_actions(state, utilities, authority, parent_bias * 0.0, rng, cfg)

    assert not np.any(state.readout == int(Action.INGEST))
    assert np.all(np.isclose(state.possibility[..., int(Action.INGEST)], 0.0, atol=1e-7))


# Loop probability and replay tests.


def make_pass10_probability_cfg() -> SimulationConfig:
    data = load_config("configs/mvp.yaml").model_dump()
    data["world"]["height"] = 20
    data["world"]["width"] = 20
    data["world"]["patch_size"] = 5
    data["world"]["max_steps"] = 8
    data["world"]["seed"] = 12345
    data["initialization"]["population_density"] = 0.45
    data["visualization"]["enabled"] = False
    data["recording"]["enabled"] = False
    return SimulationConfig.model_validate(data)


def test_loop_preserves_living_possibility_simplex() -> None:
    cfg = make_pass10_probability_cfg()
    state, metrics = run_headless(cfg, max_steps=8)
    del metrics

    living = (state.health > 0.0) & (~state.obstacle)
    assert living.any()
    sums = state.possibility.sum(axis=-1)
    assert np.allclose(sums[living], 1.0, atol=1e-4)
    assert np.all(state.possibility[living] >= -1e-7)


def test_run_headless_is_deterministic_for_fixed_seed() -> None:
    cfg = make_pass10_probability_cfg()
    state_a, metrics_a = run_headless(cfg, max_steps=6)
    state_b, metrics_b = run_headless(cfg, max_steps=6)

    assert metrics_a == metrics_b
    assert np.allclose(state_a.integration, state_b.integration)
    assert np.array_equal(state_a.readout, state_b.readout)
