from __future__ import annotations

import numpy as np
import pytest

from owl_raqic.gpu.actualization_extensions import ActualizationExtensionConfig
from owl_raqic.gpu.dense_types import RAQICDenseBatch
from owl_raqic.gpu.math_gpu import decide_dense, repair_authority_mask

ACTIONS = (
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
    "EXPEL",
    "SPLIT",
    "MERGE",
    "FLEE",
    "PURSUE",
)


def make_batch() -> RAQICDenseBatch:
    rng = np.random.default_rng(9303)
    n = 20
    features = rng.random((n, 11))
    bins = np.floor(features * 255).astype(np.int32)
    mask = rng.random((n, 22)) > 0.25
    mask[0] = False
    parent = rng.random((n, 22))
    return RAQICDenseBatch(
        ow_id=np.arange(100, 100 + n, dtype=np.int64),
        yx=np.column_stack((np.arange(n), np.zeros(n))).astype(np.int32),
        features=features,
        feature_bins=bins,
        adelic_codes=bins.copy(),
        authority_mask=mask,
        parent_intention=parent,
        alive_mask=np.ones(n, dtype=bool),
        scale_id=np.zeros(n, dtype=np.int32),
        tick=4,
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
        action_names=ACTIONS,
        active_primes=(2, 3, 5),
        action_utilities=rng.normal(size=(n, 22)),
        parent_action_phase=rng.uniform(-np.pi, np.pi, size=(n, 22)),
        parent_action_coherence=rng.random((n, 22)),
    )


def run(batch: RAQICDenseBatch, config: ActualizationExtensionConfig | None):
    return decide_dense(
        batch,
        seed=9303,
        beta_intention=1.0,
        temperature=1.0,
        epsilon_adelic=1.0,
        prime_weights={2: 0.25, 3: 0.15, 5: 0.1},
        precision="audit64",
        xp=np,
        phase_mode="canonical_device",
        actualization_config=config,
    )


def test_stable_config_is_bitwise_identical_to_unextended_call() -> None:
    batch = make_batch()
    left = run(batch, None)
    right = run(batch, ActualizationExtensionConfig())
    for a, b in zip(left, right, strict=True):
        np.testing.assert_array_equal(a, b)


def test_no_legal_authority_row_is_repaired_to_rest() -> None:
    batch = make_batch()
    repaired = repair_authority_mask(batch.authority_mask, xp=np)
    assert repaired[0, 0]
    assert int(np.sum(repaired[0])) == 1
    result = run(batch, None)
    assert result[2][0, 0] == 1.0
    assert result[3][0] == 0


def test_numpy_cupy_enabled_variant_equivalence_when_cupy_available() -> None:
    cp = pytest.importorskip("cupy")
    batch = make_batch()
    config = ActualizationExtensionConfig(
        variant="phase_interference",
        utility_coupling=0.12,
        phase_resonance_coupling=0.08,
        interference_mixer_strength=0.05,
        interference_trotter_steps=2,
    )
    np_result = run(batch, config)
    cp_result = decide_dense(
        batch,
        seed=9303,
        beta_intention=1.0,
        temperature=1.0,
        epsilon_adelic=1.0,
        prime_weights={2: 0.25, 3: 0.15, 5: 0.1},
        precision="audit64",
        xp=cp,
        phase_mode="canonical_device",
        actualization_config=config,
    )
    for left, right in zip(np_result, cp_result, strict=True):
        np.testing.assert_allclose(left, cp.asnumpy(right), atol=1e-10, rtol=1e-10)
