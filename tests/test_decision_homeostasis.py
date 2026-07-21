"""Decision-homeostasis and cross-scale regulation tests."""

from __future__ import annotations

import numpy as np

from owl.core.actions import Action, GlobalIntention, PatchIntention
from owl.core.advanced import ensure_advanced_fields
from owl.core.config import load_config
from owl.core.init import initialize_world
from owl.engine.aggregation import aggregate_global, aggregate_patches
from owl.engine.authority import compute_authority
from owl.engine.loop import assert_invariants, step
from owl.engine.topdown import compute_global_intention, compute_patch_intention
from owl.engine.utility import compute_utilities, reproduction_viability_field
from owl.record.zarr_recorder import create_recorder


def homeostasis_cfg():
    cfg = load_config("configs/mvp.yaml")
    cfg.world.height = 20
    cfg.world.width = 20
    cfg.world.patch_size = 5
    cfg.world.max_steps = 30
    cfg.ecology.advanced_enabled = True
    cfg.possibility.advanced_enabled = True
    cfg.decision_homeostasis.enabled = True
    cfg.cross_scale_homeostasis.enabled = True
    cfg.identity.enabled = True
    cfg.hierarchy.dynamic_patches = True
    cfg.hierarchy.predictive_topdown = True
    cfg.reproduction.advanced_enabled = True
    cfg.actions.stochastic = True
    cfg.actions.enabled_actions = [
        "REST",
        "SENSE",
        "MOVE_N",
        "MOVE_S",
        "MOVE_E",
        "MOVE_W",
        "MOVE_NE",
        "MOVE_NW",
        "MOVE_SE",
        "MOVE_SW",
        "FEED",
        "COMMUNICATE",
        "INHIBIT",
        "INTEGRATE",
        "REPAIR",
        "REPRODUCE",
        "INGEST",
    ]
    cfg.initialization.background_food = 0.08
    cfg.initialization.food_patch_count = 8
    return cfg


def living_ids_are_unique(state) -> bool:
    alive = (state.health > 0.0) & (~state.obstacle)
    ids = state.occupancy[alive]
    ids = ids[ids >= 0]
    if ids.size == 0:
        return True
    _, counts = np.unique(ids, return_counts=True)
    return int(counts.max()) == 1


def test_identity_continuity_unique_after_reproduction_and_movement() -> None:
    cfg = homeostasis_cfg()
    rng = np.random.default_rng(1)
    state = initialize_world(cfg, rng)
    ensure_advanced_fields(state, cfg)
    for _ in range(30):
        step(state, cfg, rng)
    assert_invariants(state, cfg)
    assert living_ids_are_unique(state)


def test_urgent_feasible_actions_concentrate_probability() -> None:
    cfg = homeostasis_cfg()
    rng = np.random.default_rng(2)
    state = initialize_world(cfg, rng)
    ensure_advanced_fields(state, cfg)
    y, x = np.argwhere(state.health > 0.0)[0]
    state.resource[y, x] = 0.01
    state.starvation_debt[y, x] = 0.95
    state.food[y, x] = 0.90
    state.boundary[y, x] = 0.90
    state.health[y, x] = 0.90
    state.patches = aggregate_patches(state, cfg)
    parent_bias = np.zeros((*state.health.shape, len(Action)), dtype=np.float32)
    utilities = compute_utilities(state, parent_bias, cfg)
    authority = compute_authority(state, cfg)
    from owl.engine.actualization import actualize_actions
    from owl.engine.loop import capture_pre_decision_state

    capture_pre_decision_state(state, cfg, authority, utilities, parent_bias)
    actualize_actions(state, utilities, authority, parent_bias, rng, cfg)
    assert state.possibility[y, x, int(Action.FEED)] > 0.55
    assert state.possibility[y, x, int(Action.REPRODUCE)] < 0.05


def test_reproduction_suppressed_by_patch_crisis() -> None:
    cfg = homeostasis_cfg()
    rng = np.random.default_rng(3)
    state = initialize_world(cfg, rng)
    ensure_advanced_fields(state, cfg)
    alive = state.health > 0.0
    state.food.fill(0.0)
    state.resource[alive] = 0.95
    state.starvation_debt[alive] = 0.9
    state.health[alive] = 0.95
    state.boundary[alive] = 0.95
    state.integration[alive] = 0.9
    state.patches = aggregate_patches(state, cfg)
    viability = reproduction_viability_field(state, cfg)
    assert float(np.mean(viability[alive])) < 0.35
    g = aggregate_global(state.patches, cfg)
    assert g.crisis > 0.25
    assert compute_global_intention(g, cfg) != int(GlobalIntention.REPRODUCE)


def test_patch_and_apex_intentions_respond_to_lower_state() -> None:
    cfg = homeostasis_cfg()
    rng = np.random.default_rng(4)
    state = initialize_world(cfg, rng)
    ensure_advanced_fields(state, cfg)
    alive = state.health > 0.0
    state.food.fill(0.0)
    state.resource[alive] = 0.02
    state.starvation_debt[alive] = 1.0
    state.patches = aggregate_patches(state, cfg)
    compute_patch_intention(state.patches, cfg)
    assert int(PatchIntention.REPRODUCE) not in set(np.unique(state.patches.intention).tolist())
    g = aggregate_global(state.patches, cfg)
    intention = GlobalIntention(compute_global_intention(g, cfg))
    assert intention in {
        GlobalIntention.CONSERVE,
        GlobalIntention.SEEK_FOOD,
        GlobalIntention.REPAIR,
        GlobalIntention.COORDINATE,
    }


def test_noetic_decomposition_bounds() -> None:
    cfg = homeostasis_cfg()
    rng = np.random.default_rng(5)
    state = initialize_world(cfg, rng)
    ensure_advanced_fields(state, cfg)
    state.patches = aggregate_patches(state, cfg)
    for name in (
        "noetic_B",
        "noetic_M",
        "noetic_P",
        "noetic_C",
        "noetic_K",
        "noetic_Theta",
        "noetic_N",
    ):
        arr = getattr(state, name)
        assert np.all(np.isfinite(arr))
        assert float(np.min(arr)) >= -1e-6
        assert float(np.max(arr)) <= 1.0 + 1e-6


def test_zarr_records_decision_audit_fields(tmp_path) -> None:
    cfg = homeostasis_cfg()
    cfg.recording.enabled = True
    cfg.recording.zarr_path = str(tmp_path / "run.zarr")
    cfg.recording.record_every = 1
    cfg.world.max_steps = 3
    rng = np.random.default_rng(6)
    state = initialize_world(cfg, rng)
    ensure_advanced_fields(state, cfg)
    recorder = create_recorder(cfg, state, max_steps=3)
    assert recorder is not None
    for _ in range(3):
        step(state, cfg, rng)
        recorder.maybe_record(state)
    names = set(recorder._arrays)
    recorder.close()
    required = {
        "state/occupancy",
        "state/pre_utilities",
        "state/pre_authority",
        "state/last_survival_value",
        "state/last_decision_urgency",
        "state/last_macro_probabilities",
        "state/noetic_N",
        "patch/crisis",
        "patch/carrying_pressure",
        "global/crisis",
        "global/carrying_pressure",
    }
    assert required <= names
