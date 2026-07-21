from __future__ import annotations

from typing import Any

import numpy as np

from owl.core.actions import Action
from owl.core.config import load_config
from owl.core.init import initialize_world
from owl.gpu.device_state import OWLDeviceState
from owl.raqic.engine import OWLRAQICDecisionBatch, apply_raqic_decisions
from owl.raqic.precision import (
    RAQIC_AUDIT_REAL_FIELDS,
    RAQIC_RECURSIVE_REAL_FIELDS,
    raqic_numpy_dtype,
)
from owl.raqic.state import ensure_raqic_fields


def _config(precision: str = "audit64") -> Any:
    cfg = load_config("configs/gpu_v096_utility_authoritative.yaml")
    cfg.world.height = 10
    cfg.world.width = 10
    cfg.world.patch_size = 5
    cfg.world.max_steps = 1
    cfg.raqic.full_gpu_precision = precision
    cfg.raqic.strict_gpu = False
    cfg.raqic.full_gpu_strict = False
    cfg.raqic.fallback_on_backend_error = True
    cfg.raqic.full_gpu_no_silent_fallback = False
    return cfg


def _state(cfg: Any) -> Any:
    state = initialize_world(cfg, np.random.default_rng(cfg.world.seed))
    ensure_raqic_fields(state, cfg)
    return state


def test_recursive_raqic_fields_follow_audit64_policy() -> None:
    cfg = _config("audit64")
    state = _state(cfg)
    assert raqic_numpy_dtype(cfg) == np.dtype(np.float64)
    for field in RAQIC_AUDIT_REAL_FIELDS | RAQIC_RECURSIVE_REAL_FIELDS:
        value = getattr(state, field, None)
        if isinstance(value, np.ndarray):
            assert value.dtype == np.float64, field
    assert state.possibility.dtype == np.float32


def test_balanced32_keeps_recursive_fields_float32() -> None:
    cfg = _config("balanced32")
    state = _state(cfg)
    assert raqic_numpy_dtype(cfg) == np.dtype(np.float32)
    for field in RAQIC_AUDIT_REAL_FIELDS | RAQIC_RECURSIVE_REAL_FIELDS:
        value = getattr(state, field, None)
        if isinstance(value, np.ndarray):
            assert value.dtype == np.float32, field


def test_numpy_device_state_matches_recursive_precision_policy() -> None:
    for precision, expected in (("audit64", np.float64), ("balanced32", np.float32)):
        cfg = _config(precision)
        state = _state(cfg)
        device = OWLDeviceState.from_world_state(state, cfg, force_backend="numpy")
        for field in RAQIC_AUDIT_REAL_FIELDS | RAQIC_RECURSIVE_REAL_FIELDS:
            value = device.arrays.get(field)
            if isinstance(value, np.ndarray):
                assert value.dtype == expected, (precision, field)
        assert device.arrays["possibility"].dtype == np.float32


def test_dtype_migration_preserves_existing_recursive_values() -> None:
    cfg = _config("balanced32")
    state = _state(cfg)
    assert state.raqic_probabilities is not None
    marker = np.float32(0.1234567)
    state.raqic_probabilities[0, 0, int(Action.REST)] = marker

    cfg.raqic.full_gpu_precision = "audit64"
    ensure_raqic_fields(state, cfg)
    assert state.raqic_probabilities.dtype == np.float64
    assert state.raqic_probabilities[0, 0, int(Action.REST)] == np.float64(marker)


def test_authoritative_probability_does_not_alias_physical_projection(monkeypatch: Any) -> None:
    cfg = _config("audit64")
    state = _state(cfg)
    h, w = state.health.shape
    actions = len(Action)
    authoritative = np.zeros((h, w, actions), dtype=np.float64)
    authoritative[..., int(Action.REST)] = np.float64(0.5000000001)
    authoritative[..., int(Action.SENSE)] = np.float64(0.4999999999)
    readout = np.full((h, w), int(Action.REST), dtype=np.int16)

    class FakeEngine:
        def prepare_cross_scale_context(self, state: Any, cfg: Any) -> np.ndarray:
            return np.asarray(state.raqic_parent_intention)

        def decide_cells(self, *args: Any, **kwargs: Any) -> OWLRAQICDecisionBatch:
            del args, kwargs
            return OWLRAQICDecisionBatch(
                probabilities=authoritative.copy(),
                readout=readout,
                records={
                    "action": readout,
                    "readout": readout.astype(np.int32),
                    "confidence": np.full((h, w), 0.5, dtype=np.float64),
                    "trace_error": np.zeros((h, w), dtype=np.float64),
                    "min_eigenvalue": np.zeros((h, w), dtype=np.float64),
                    "backend_code": np.zeros((h, w), dtype=np.int32),
                },
                scores=np.zeros_like(authoritative),
                phases=np.zeros_like(authoritative),
                audit={},
            )

    import owl.raqic.gpu_engine as gpu_engine

    monkeypatch.setattr(gpu_engine, "OWLRAQICGPUEngine", FakeEngine)
    authority = np.ones((h, w, actions), dtype=bool)
    apply_raqic_decisions(
        state,
        cfg,
        authority,
        np.random.default_rng(cfg.world.seed),
        utilities=np.zeros((h, w, actions), dtype=np.float64),
    )

    assert state.raqic_probabilities is not None
    assert state.raqic_probabilities.dtype == np.float64
    np.testing.assert_array_equal(state.raqic_probabilities, authoritative)
    assert state.possibility.dtype == np.float32
    assert not np.shares_memory(state.possibility, state.raqic_probabilities)
    before = state.raqic_probabilities.copy()
    state.possibility[..., int(Action.REST)] = 0.0
    np.testing.assert_array_equal(state.raqic_probabilities, before)
