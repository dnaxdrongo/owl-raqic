from __future__ import annotations

import numpy as np

from owl.core.actions import Action
from owl.core.config import SimulationConfig
from owl.core.state import WorldState, field_shape


def assert_raqic_invariants(state: WorldState, cfg: SimulationConfig) -> None:
    if not getattr(cfg.raqic, "enabled", False):
        return
    h, w = field_shape(state)
    actions = len(Action)
    alive = (state.health > 0.0) & (~state.obstacle)
    for name in ("raqic_probabilities", "raqic_parent_intention"):
        arr = getattr(state, name, None)
        assert isinstance(arr, np.ndarray), f"{name} must exist"
        assert arr.shape == (h, w, actions), f"{name} shape mismatch"
        assert np.all(np.isfinite(arr)), f"{name} must be finite"
        assert np.nanmin(arr) >= -1e-6, f"{name} must be nonnegative"
        if np.any(alive):
            assert np.allclose(np.sum(arr[alive], axis=-1), 1.0, atol=1e-4), (
                f"{name} must sum to one"
            )
    assert state.raqic_probabilities is not None
    assert state.raqic_readout is not None
    dead = ~alive
    if np.any(dead):
        assert np.all(state.raqic_probabilities[dead, int(Action.REST)] >= 1 - 1e-6), (
            "dead cells must REST in RAQIC probs"
        )
        assert np.all(state.raqic_readout[dead] == int(Action.REST)), (
            "dead cells must REST in RAQIC readout"
        )
    if getattr(cfg.raqic, "assert_recovery_gates", False):
        tr = getattr(state, "raqic_trace_error", None)
        me = getattr(state, "raqic_min_eigenvalue", None)
        if isinstance(tr, np.ndarray):
            assert np.nanmax(tr) < 1e-4, "RAQIC trace residual too large"
        if isinstance(me, np.ndarray):
            assert np.nanmin(me) >= -1e-5, "RAQIC positivity gate failed"
