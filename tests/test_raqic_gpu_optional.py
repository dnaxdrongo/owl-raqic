from __future__ import annotations

import numpy as np
import pytest

from owl_raqic.gpu.backend import detect_cupy
from owl_raqic.gpu.decision_engine import RAQICDenseDecisionEngine, RAQICDenseExecutionConfig
from owl_raqic.gpu.dense_types import RAQICDenseBatch


def _batch():
    pytest.importorskip("cupy")
    return RAQICDenseBatch(
        ow_id=np.arange(8, dtype=np.int64),
        yx=np.column_stack([np.arange(8), np.zeros(8, dtype=int)]).astype(np.int32),
        features=np.random.default_rng(1).random((8, 11)),
        feature_bins=np.random.default_rng(2).integers(0, 255, size=(8, 11), dtype=np.int32),
        adelic_codes=np.zeros((8, 11), dtype=np.int32),
        authority_mask=np.ones((8, 5), dtype=bool),
        parent_intention=np.ones((8, 5), dtype=float) / 5,
        alive_mask=np.ones(8, dtype=bool),
        scale_id=np.zeros(8, dtype=np.int32),
        tick=2,
        feature_names=(
            "resource",
            "risk",
            "memory",
            "coherence",
            "phase",
            "boundary",
            "signal",
            "prediction_error",
            "parent_context",
            "food",
            "toxin",
        ),
        action_names=("REST", "SENSE", "MOVE_N", "FEED", "REPAIR"),
        active_primes=(2, 3, 5),
    )


def test_cupy_runtime_detection_optional():
    pytest.importorskip("cupy")
    info = detect_cupy()
    assert isinstance(info.available, bool)
    if info.available:
        assert info.float64_test_passed


def test_cupy_dense_matches_numpy_optional():
    pytest.importorskip("cupy")
    batch = _batch()
    cfg_np = RAQICDenseExecutionConfig(
        backend="numpy", precision="audit64", prime_weights={2: 0.25, 3: 0.15, 5: 0.10}
    )
    cfg_gpu = RAQICDenseExecutionConfig(
        backend="cupy", precision="audit64", prime_weights={2: 0.25, 3: 0.15, 5: 0.10}
    )
    np_res = RAQICDenseDecisionEngine(cfg_np).decide_batch(batch).to_numpy()
    gpu_res = RAQICDenseDecisionEngine(cfg_gpu).decide_batch(batch).to_numpy()
    assert np.allclose(gpu_res.probabilities, np_res.probabilities, atol=1e-10)
    assert np.array_equal(gpu_res.readout, np_res.readout)
