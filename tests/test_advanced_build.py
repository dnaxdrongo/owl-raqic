"""Advanced-build augmentation tests."""

from __future__ import annotations

import numpy as np

from owl.core.actions import Action
from owl.core.advanced import action_entropy, ensure_advanced_fields
from owl.core.config import SimulationConfig, load_config
from owl.core.init import initialize_world
from owl.engine.feeding import apply_feeding, compute_intake
from owl.engine.loop import assert_invariants, run_headless
from owl.record.metrics import collect_metrics
from owl.record.snapshots import load_snapshot, save_snapshot
from owl.viz.palettes import genome_palette, trust_palette, waste_palette


def advanced_cfg() -> SimulationConfig:
    data = load_config("configs/mvp.yaml").model_dump()
    data["world"]["height"] = 20
    data["world"]["width"] = 20
    data["world"]["patch_size"] = 5
    data["world"]["max_steps"] = 10
    data["debug"]["assert_invariants"] = True
    data["ecology"]["advanced_enabled"] = True
    data["possibility"]["advanced_enabled"] = True
    data["communication"]["source_tracking_enabled"] = True
    data["hierarchy"]["dynamic_patches"] = True
    data["hierarchy"]["predictive_topdown"] = True
    data["reproduction"]["advanced_enabled"] = True
    data["reproduction"]["min_resource"] = 0.30
    data["recording"]["enabled"] = False
    data["visualization"]["enabled"] = False
    return SimulationConfig.model_validate(data)


def test_advanced_fields_allocate_and_config_validates() -> None:
    cfg = advanced_cfg()
    state = initialize_world(cfg, np.random.default_rng(123))
    ensure_advanced_fields(state, cfg)

    assert state.digestion.shape == state.health.shape
    assert state.action_cooldown.shape == state.possibility.shape
    assert state.neighbor_trust.shape[:2] == state.health.shape
    assert state.neighbor_trust.shape[2] == 8
    assert state.genome.shape == (*state.health.shape, cfg.reproduction.genome_length)
    assert np.all((state.genome >= 0.0) & (state.genome <= 1.0))
    assert_invariants(state, cfg)


def test_advanced_feeding_is_saturating_and_uses_digestion_buffer() -> None:
    cfg = advanced_cfg()
    state = initialize_world(cfg, np.random.default_rng(0))
    ensure_advanced_fields(state, cfg)
    living = np.argwhere((state.health > 0.0) & (~state.obstacle))
    y, x = map(int, living[0])
    state.food.fill(0.0)
    state.resource[y, x] = 0.2
    state.grazing[y, x] = 1.0
    state.readout[y, x] = int(Action.FEED)

    state.food[y, x] = 0.1
    low = compute_intake(state, cfg)[y, x]
    state.food[y, x] = 1.0
    high = compute_intake(state, cfg)[y, x]

    assert high > low
    assert high <= cfg.resources.feed_efficiency + 1e-6

    before_resource = float(state.resource[y, x])
    apply_feeding(state, cfg)
    assert state.digestion[y, x] > 0.0
    assert state.resource[y, x] > before_resource  # immediate assimilation repair
    assert state.last_intake[y, x] > 0.0


def test_resource_zero_does_not_instantly_kill_healthy_cell() -> None:
    cfg = advanced_cfg()
    state = initialize_world(cfg, np.random.default_rng(123))
    ensure_advanced_fields(state, cfg)
    living = np.argwhere((state.health > 0.0) & (~state.obstacle))
    y, x = map(int, living[0])
    state.resource[y, x] = 0.0
    state.health[y, x] = 0.90
    state.boundary[y, x] = 0.80
    state.integration[y, x] = 0.10
    state.starvation_debt[y, x] = 0.50

    from owl.engine.death import detect_dead_cells

    assert not bool(detect_dead_cells(state, cfg)[y, x])


def test_starvation_debt_accumulates_and_damages_health() -> None:
    cfg = advanced_cfg()
    state = initialize_world(cfg, np.random.default_rng(123))
    ensure_advanced_fields(state, cfg)
    living = np.argwhere((state.health > 0.0) & (~state.obstacle))
    y, x = map(int, living[0])
    state.resource[y, x] = 0.0
    state.health[y, x] = 0.90
    before = float(state.health[y, x])

    from owl.engine.health import apply_metabolism_damage

    apply_metabolism_damage(state, cfg)
    assert state.starvation_debt[y, x] > 0.0
    assert state.health[y, x] < before


def test_diagonal_movement_enabled_and_sampled() -> None:
    cfg = advanced_cfg()
    state = initialize_world(cfg, np.random.default_rng(123))
    ensure_advanced_fields(state, cfg)
    cfg.actions.stochastic = True
    cfg.actions.movement_macro_enabled = True
    cfg.actions.diagonal_movement_enabled = True
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
        "FEED",
        "INTEGRATE",
    ]
    parent_bias = np.zeros_like(state.possibility)
    y, x = np.argwhere((state.health > 0.0) & (~state.obstacle))[0]
    state.food.fill(0.0)
    state.food[(int(y) - 1) % state.food.shape[0], (int(x) + 1) % state.food.shape[1]] = 1.0
    state.resource[int(y), int(x)] = 0.05
    state.mobility[int(y), int(x)] = 1.0

    from owl.engine.actualization import compute_action_logits, movement_direction_probabilities
    from owl.engine.authority import compute_authority
    from owl.engine.utility import compute_utilities

    utilities = compute_utilities(state, parent_bias, cfg)
    authority = compute_authority(state, cfg)
    logits = compute_action_logits(state, utilities, authority, parent_bias, cfg)
    probs, actions = movement_direction_probabilities(state, logits, int(y), int(x), cfg)
    assert "MOVE_NE" in [a.name for a in actions]
    assert probs[[a.name for a in actions].index("MOVE_NE")] == probs.max()


def test_action_entropy_bounds() -> None:
    p = np.array([[[1.0, 0.0], [0.5, 0.5]]], dtype=np.float32)
    h = action_entropy(p)
    assert h.shape == (1, 2)
    assert np.all((h >= 0.0) & (h <= 1.0))
    assert h[0, 1] > h[0, 0]


def test_advanced_run_metrics_snapshot_and_palettes(tmp_path) -> None:
    cfg = advanced_cfg()
    state, rows = run_headless(cfg, max_steps=10)
    assert len(rows) == 10
    assert_invariants(state, cfg)

    metrics = collect_metrics(state, cfg)
    for key in ("waste_mean", "digestion_mean", "genome_diversity", "neighbor_trust_mean"):
        assert key in metrics

    path = tmp_path / "advanced_snapshot.npz"
    save_snapshot(state, path)
    loaded = load_snapshot(path)
    ensure_advanced_fields(loaded, cfg)
    assert loaded.genome.shape == state.genome.shape
    assert loaded.neighbor_trust.shape == state.neighbor_trust.shape

    for palette in (waste_palette, trust_palette, genome_palette):
        rgb = palette(state)
        assert rgb.shape == (cfg.world.width, cfg.world.height, 3)
        assert rgb.dtype == np.uint8
