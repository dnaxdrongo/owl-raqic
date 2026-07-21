from __future__ import annotations

import numpy as np
import pytest

from owl.core.actions import Action
from owl.core.config import SimulationConfig, load_config
from owl.core.init import initialize_world
from owl.engine.authority import compute_authority
from owl.engine.loop import run, step
from owl.engine.utility import compute_utilities
from owl.raqic.action_adapter import assert_action_basis_compatible
from owl.raqic.config import build_raqic_action_set, convert_owl_cfg_to_raqic_cfg
from owl.raqic.feature_extraction import build_feature_packets
from owl.raqic.invariants import assert_raqic_invariants
from owl.raqic.state import ensure_raqic_fields
from owl_raqic.algorithms.feature_pipeline import score_weights_for_actions


def small_cfg(policy: str = "raqic", steps: int = 3) -> SimulationConfig:
    return SimulationConfig.model_validate(
        {
            "world": {"height": 10, "width": 10, "patch_size": 5, "max_steps": steps, "seed": 123},
            "recording": {"enabled": False},
            "visualization": {"enabled": False, "backend": "none"},
            "debug": {"assert_invariants": True},
            "raqic": {
                "enabled": True,
                "mode": "cpu_audit",
                "decision_policy": policy,
                "epsilon_raqic": 1.0,
                "epsilon_adelic": 1.0,
                "rounds_per_tick": 1,
                "shots": 64,
                "active_primes": [2, 3, 5],
                "max_cells_per_tick": 20,
            },
        }
    )


def test_raqic_config_default_disabled():
    cfg = SimulationConfig()
    assert cfg.raqic.enabled is False
    assert cfg.raqic.decision_policy == "legacy"


def test_raqic_config_files_load():
    assert load_config("configs/raqic_cpu_audit.yaml").raqic.enabled
    assert (
        load_config("configs/raqic_hybrid_compare.yaml").raqic.decision_policy == "hybrid_compare"
    )


def test_raqic_invalid_prime_rejected():
    with pytest.raises(ValueError):
        SimulationConfig.model_validate({"raqic": {"active_primes": [2, 4]}})


def test_raqic_action_basis_matches_owl():
    action_set = build_raqic_action_set()
    assert len(action_set) == len(Action)
    assert_action_basis_compatible(action_set)
    rq = convert_owl_cfg_to_raqic_cfg(small_cfg())
    assert rq.registers.n_actions == len(Action)
    weights = score_weights_for_actions(len(Action), tuple(a.name for a in Action))
    assert weights.shape == (len(Action), 11)


def test_ensure_raqic_fields_shapes():
    cfg = small_cfg()
    rng = np.random.default_rng(cfg.world.seed)
    state = initialize_world(cfg, rng)
    ensure_raqic_fields(state, cfg)
    assert state.raqic_probabilities.shape == (*state.health.shape, len(Action))
    assert state.raqic_parent_intention.shape == (*state.health.shape, len(Action))
    assert state.raqic_patch_intention.shape[-1] == len(Action)


def test_feature_extraction_no_mutation_and_masks():
    cfg = small_cfg()
    rng = np.random.default_rng(cfg.world.seed)
    state = initialize_world(cfg, rng)
    ensure_raqic_fields(state, cfg)
    parent_bias = np.zeros((*state.health.shape, len(Action)), dtype=np.float32)
    compute_utilities(state, parent_bias, cfg)
    authority = compute_authority(state, cfg)
    before = state.resource.copy()
    packets = build_feature_packets(state, cfg, authority, state.raqic_parent_intention)
    assert np.array_equal(state.resource, before)
    assert packets
    assert packets[0].authority_mask.shape == (len(Action),)


def test_step_raqic_runs_one_tick_probability_simplex():
    cfg = small_cfg(steps=1)
    rng = np.random.default_rng(cfg.world.seed)
    state = initialize_world(cfg, rng)
    step(state, cfg, rng)
    alive = (state.health > 0.0) & (~state.obstacle)
    assert np.allclose(state.possibility[alive].sum(axis=-1), 1.0, atol=1e-4)
    assert hasattr(state, "raqic_probabilities")
    assert_raqic_invariants(state, cfg)


def test_hybrid_compare_records_both_paths():
    cfg = small_cfg(policy="hybrid_compare", steps=1)
    rng = np.random.default_rng(cfg.world.seed)
    state = initialize_world(cfg, rng)
    step(state, cfg, rng)
    assert hasattr(state, "raqic_legacy_shadow_possibility")
    assert hasattr(state, "raqic_compare_l1")
    assert np.all(np.isfinite(state.raqic_compare_l1))


def test_raqic_recovery_epsilon_zero_uses_legacy_path():
    cfg = small_cfg(policy="raqic", steps=1)
    cfg.raqic.epsilon_raqic = 0.0
    rng1 = np.random.default_rng(cfg.world.seed)
    rng2 = np.random.default_rng(cfg.world.seed)
    s1 = initialize_world(cfg, rng1)
    s2 = initialize_world(cfg, rng2)
    legacy_cfg = cfg.model_copy(deep=True)
    legacy_cfg.raqic.enabled = False
    step(s1, cfg, rng1)
    step(s2, legacy_cfg, rng2)
    assert np.allclose(s1.possibility, s2.possibility)
    assert np.array_equal(s1.readout, s2.readout)


def test_run_raqic_100_ticks_small_headless():
    cfg = small_cfg(steps=100)
    cfg.raqic.max_cells_per_tick = 8
    state, metrics = run(cfg)
    assert len(metrics) == 100
    alive = (state.health > 0.0) & (~state.obstacle)
    if np.any(alive):
        assert np.allclose(state.possibility[alive].sum(axis=-1), 1.0, atol=1e-4)
    assert_raqic_invariants(state, cfg)
