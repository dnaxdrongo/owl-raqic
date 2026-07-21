from __future__ import annotations

import numpy as np

from owl.core.config import SimulationConfig
from owl.core.init import initialize_world
from owl.engine.authority import compute_authority
from owl.raqic.dense_feature_extraction import build_dense_feature_batch_numpy
from owl.raqic.gpu_engine import OWLRAQICGPUEngine
from owl.raqic.state import ensure_raqic_fields


def _cfg(mode="gpu_batch", fallback=True):
    return SimulationConfig.model_validate(
        {
            "world": {"height": 10, "width": 10, "patch_size": 5, "max_steps": 2, "seed": 222},
            "raqic": {
                "enabled": True,
                "decision_policy": "raqic",
                "mode": mode,
                "strict_gpu": False,
                "fallback_on_backend_error": fallback,
                "gpu_all_cells_required": True,
                "gpu_precision": "audit64",
                "gpu_audit_limit": 2,
            },
        }
    )


def test_new_gpu_config_modes_validate():
    cfg = _cfg("gpu_batch")
    assert cfg.raqic.mode == "gpu_batch"
    cfg = _cfg("gpu_hybrid_audit")
    assert cfg.raqic.mode == "gpu_hybrid_audit"


def test_dense_feature_batch_processes_all_eligible_cells():
    cfg = _cfg()
    rng = np.random.default_rng(cfg.world.seed)
    state = initialize_world(cfg, rng)
    ensure_raqic_fields(state, cfg)
    authority = compute_authority(state, cfg)
    batch = build_dense_feature_batch_numpy(state, cfg, authority, state.raqic_parent_intention)
    eligible = int(np.sum((state.health > 0.0) & (~state.obstacle)))
    assert batch.metadata["processed_cells"] == eligible
    assert batch.features.shape[0] == eligible
    assert batch.authority_mask.shape[1] == len(state.possibility[0, 0])


def test_gpu_engine_dense_numpy_fallback_runs_all_cells_when_gpu_absent_or_fallback_allowed():
    cfg = _cfg()
    rng = np.random.default_rng(cfg.world.seed)
    state = initialize_world(cfg, rng)
    ensure_raqic_fields(state, cfg)
    engine = OWLRAQICGPUEngine()
    engine.prepare_cross_scale_context(state, cfg)
    authority = compute_authority(state, cfg)
    result = engine.decide_cells(state, cfg, authority, rng)
    assert result.probabilities.shape == state.possibility.shape
    alive = (state.health > 0.0) & (~state.obstacle)
    assert np.allclose(result.probabilities[alive].sum(axis=1), 1.0, atol=1e-8)
    assert result.audit["all_cells_satisfied"] is True
