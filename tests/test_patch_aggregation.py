"""Patch and global aggregation and top-down-bias tests."""

from __future__ import annotations

import numpy as np
import pytest

from owl.core.actions import Action, GlobalIntention, PatchIntention, SignalChannel
from owl.core.config import SimulationConfig, load_config
from owl.core.init import initialize_world
from owl.engine.aggregation import (
    aggregate_global,
    aggregate_patches,
    block_view_2d,
    upsample_patch_bias,
    upsample_patch_field,
)
from owl.engine.loop import step
from owl.engine.topdown import (
    apply_threshold_modulation,
    compute_global_intention,
    compute_patch_intention,
    global_policy_to_bias,
    patch_policy_to_bias,
)


def make_cfg(height: int = 20, width: int = 20, patch_size: int = 5) -> SimulationConfig:
    """Return a small deterministic config for patch/top-down tests."""
    data = load_config("configs/mvp.yaml").model_dump()
    data["world"]["height"] = height
    data["world"]["width"] = width
    data["world"]["patch_size"] = patch_size
    data["initialization"]["population_density"] = 0.60
    data["initialization"]["food_patch_count"] = 2
    data["initialization"]["food_patch_radius"] = 3
    data["visualization"]["enabled"] = False
    data["recording"]["enabled"] = False
    return SimulationConfig.model_validate(data)


def make_state(seed: int = 123):
    """Create a small initialized state for pass tests."""
    cfg = make_cfg()
    rng = np.random.default_rng(seed)
    return cfg, initialize_world(cfg, rng)


def test_block_view_2d_tiles_exactly_and_preserves_patch_layout() -> None:
    field = np.arange(16, dtype=np.float32).reshape(4, 4)
    blocks = block_view_2d(field, 2)

    assert blocks.shape == (2, 2, 2, 2)
    assert np.array_equal(blocks[0, 0], np.array([[0, 1], [4, 5]], dtype=np.float32))
    assert np.array_equal(blocks[1, 1], np.array([[10, 11], [14, 15]], dtype=np.float32))

    with pytest.raises(ValueError, match="divisible"):
        block_view_2d(np.zeros((5, 4), dtype=np.float32), 2)
    with pytest.raises(ValueError, match="2D"):
        block_view_2d(np.zeros((4, 4, 1), dtype=np.float32), 2)
    with pytest.raises(ValueError, match="positive"):
        block_view_2d(field, 0)


def test_aggregate_patches_returns_bounded_expected_shapes() -> None:
    cfg, state = make_state()
    # Make a few signals visible so signal_pressure is not trivially zero.
    state.signal_reception[..., int(SignalChannel.FOOD)] = 0.2
    state.signal_reception[:5, :5, int(SignalChannel.DANGER)] = 0.8

    patches = aggregate_patches(state, cfg)

    ph = cfg.world.height // cfg.world.patch_size
    pw = cfg.world.width // cfg.world.patch_size

    assert patches.activation.shape == (ph, pw)
    assert patches.phase.shape == (ph, pw)
    assert patches.possibility.shape == (ph, pw, len(Action))
    assert patches.signal_pressure.shape == (ph, pw, cfg.communication.num_channels)
    assert patches.policy_bias.shape == (ph, pw, len(Action))
    assert np.all(np.isfinite(patches.phase))
    assert np.all((patches.activation >= 0.0) & (patches.activation <= 1.0))
    assert np.all((patches.integration >= 0.0) & (patches.integration <= 1.0))
    assert np.all((patches.synchrony >= 0.0) & (patches.synchrony <= 1.0))
    assert np.all((patches.coherence >= 0.0) & (patches.coherence <= 1.0))
    assert np.all((patches.cross_scale >= 0.0) & (patches.cross_scale <= 1.0))
    assert np.all((patches.signal_pressure >= 0.0) & (patches.signal_pressure <= 1.0))
    assert np.allclose(patches.possibility.sum(axis=-1), 1.0, atol=1e-6)


def test_aggregate_patches_does_not_mutate_world_state() -> None:
    cfg, state = make_state()
    before_integration = state.integration.copy()
    before_possibility = state.possibility.copy()

    _ = aggregate_patches(state, cfg)

    assert np.array_equal(state.integration, before_integration)
    assert np.array_equal(state.possibility, before_possibility)


def test_aggregate_global_produces_bounded_apex_summary() -> None:
    cfg, state = make_state()
    patches = aggregate_patches(state, cfg)
    compute_patch_intention(patches, cfg)
    patches.policy_bias[...] = patch_policy_to_bias(patches, cfg)

    global_state = aggregate_global(patches, cfg)

    assert 0.0 <= global_state.integration <= 1.0
    assert 0.0 <= global_state.fragmentation <= 1.0
    assert 0.0 <= global_state.diversity <= 1.0
    assert 0.0 <= global_state.complexity <= 1.0
    assert global_state.signal_pressure.shape == (cfg.communication.num_channels,)
    assert global_state.policy_bias.shape == (len(Action),)
    assert global_state.readout in range(len(Action))


def test_upsample_patch_field_and_bias_shapes() -> None:
    patch_field = np.array([[1, 2], [3, 4]], dtype=np.float32)
    up = upsample_patch_field(patch_field, 2)
    assert up.shape == (4, 4)
    assert np.array_equal(up[:2, :2], np.ones((2, 2), dtype=np.float32))

    patch_bias = np.zeros((2, 2, len(Action)), dtype=np.float32)
    patch_bias[0, 1, int(Action.FEED)] = 0.5
    bias_up = upsample_patch_bias(patch_bias, 3)

    assert bias_up.shape == (6, 6, len(Action))
    assert np.allclose(bias_up[:3, 3:, int(Action.FEED)], 0.5)

    with pytest.raises(ValueError, match="3D"):
        upsample_patch_bias(np.zeros((2, 2), dtype=np.float32), 2)


def test_compute_patch_intention_and_policy_bias_are_bounded() -> None:
    cfg, state = make_state()
    patches = aggregate_patches(state, cfg)

    patches.signal_pressure.fill(0.0)
    patches.integration.fill(0.8)
    patches.health.fill(1.0)
    patches.resource.fill(0.2)
    patches.signal_pressure[0, 0, int(SignalChannel.FOOD)] = 1.0
    patches.signal_pressure[0, 1, int(SignalChannel.DANGER)] = 1.0
    patches.signal_pressure[1, 0, int(SignalChannel.COORDINATION)] = 1.0

    compute_patch_intention(patches, cfg)
    bias = patch_policy_to_bias(patches, cfg)

    assert patches.intention.shape == patches.integration.shape
    assert patches.intention[0, 0] == int(PatchIntention.SEEK_FOOD)
    assert patches.intention[0, 1] == int(PatchIntention.AVOID_DANGER)
    assert patches.intention[1, 0] == int(PatchIntention.COORDINATE)
    assert bias.shape == patches.policy_bias.shape
    assert np.array_equal(bias, patches.policy_bias)
    assert np.all(np.abs(bias) <= cfg.topdown.max_parent_control + 1e-7)
    assert bias[0, 0, int(Action.FEED)] > 0.0
    assert bias[0, 1, int(Action.FLEE)] > 0.0
    assert bias[1, 0, int(Action.INTEGRATE)] > 0.0


def test_global_intention_and_policy_bias_are_weak_and_bounded() -> None:
    cfg, state = make_state()
    patches = aggregate_patches(state, cfg)
    global_state = aggregate_global(patches, cfg)

    global_state.signal_pressure.fill(0.0)
    global_state.integration = 0.9
    global_state.fragmentation = 0.8
    global_state.signal_pressure[int(SignalChannel.COORDINATION)] = 1.0

    intention = compute_global_intention(global_state, cfg)
    global_state.intention = intention
    bias = global_policy_to_bias(global_state, cfg)

    assert intention == int(GlobalIntention.COORDINATE)
    assert bias.shape == (len(Action),)
    assert np.array_equal(bias, global_state.policy_bias)
    assert np.all(np.abs(bias) <= cfg.topdown.max_parent_control + 1e-7)
    assert bias[int(Action.INTEGRATE)] > 0.0


def test_apply_threshold_modulation_mutates_only_threshold_and_stays_bounded() -> None:
    cfg, state = make_state()
    patches = aggregate_patches(state, cfg)
    patches.integration.fill(1.0)
    patches.health.fill(1.0)
    patches.intention.fill(int(PatchIntention.COORDINATE))

    before_threshold = state.threshold.copy()
    before_integration = state.integration.copy()
    before_readout = state.readout.copy()

    apply_threshold_modulation(state, patches, cfg)

    assert not np.array_equal(state.threshold, before_threshold)
    assert np.all((state.threshold >= 0.0) & (state.threshold <= 1.0))
    assert np.array_equal(state.integration, before_integration)
    assert np.array_equal(state.readout, before_readout)

    # Repeated modulation should remain bounded and asymptotic, not run away.
    for _ in range(20):
        apply_threshold_modulation(state, patches, cfg)
    assert np.all((state.threshold >= 0.0) & (state.threshold <= 1.0))


def test_apply_threshold_modulation_rejects_shape_mismatch() -> None:
    cfg, state = make_state()
    bad_cfg = make_cfg(height=20, width=20, patch_size=4)
    patches = aggregate_patches(state, cfg)

    with pytest.raises(ValueError):
        apply_threshold_modulation(state, patches, bad_cfg)


# Loop patch and global refresh tests.


def test_step_refreshes_patch_and_global_state_after_mutations() -> None:
    cfg, state = make_state(seed=456)
    rng = np.random.default_rng(456)
    before_tick = state.tick

    step(state, cfg, rng)

    assert state.tick == before_tick + 1
    ph = cfg.world.height // cfg.world.patch_size
    pw = cfg.world.width // cfg.world.patch_size
    assert state.patches.integration.shape == (ph, pw)
    assert state.patches.policy_bias.shape == (ph, pw, len(Action))
    assert state.global_state.policy_bias.shape == (len(Action),)
    assert 0.0 <= state.global_state.integration <= 1.0
    assert np.all((state.patches.integration >= 0.0) & (state.patches.integration <= 1.0))
