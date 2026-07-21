from __future__ import annotations

from typing import Any, cast

import numpy as np

from owl_raqic.gpu.actualization_extensions import ActualizationExtensionConfig
from owl_raqic.gpu.dense_types import RAQICDenseBatch
from owl_raqic.gpu.math_gpu import decide_dense

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


def _batch() -> RAQICDenseBatch:
    rng = np.random.default_rng(90210)
    n = 12
    features = rng.random((n, 11))
    bins = np.floor(features * 255.0).astype(np.int32)
    authority = rng.random((n, 22)) > 0.2
    authority[:, 0] = True
    parent = rng.random((n, 22))
    parent /= parent.sum(axis=1, keepdims=True)
    return RAQICDenseBatch(
        ow_id=np.arange(n, dtype=np.int64),
        yx=np.column_stack((np.arange(n), np.zeros(n))).astype(np.int32),
        features=features,
        feature_bins=bins,
        adelic_codes=bins.copy(),
        authority_mask=authority,
        parent_intention=parent,
        alive_mask=np.ones(n, dtype=bool),
        scale_id=np.zeros(n, dtype=np.int32),
        tick=7,
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


def _run(batch: RAQICDenseBatch, config: ActualizationExtensionConfig | None) -> tuple[Any, ...]:
    return cast(
        tuple[Any, ...],
        decide_dense(
            batch,
            seed=9303,
            beta_intention=1.0,
            temperature=1.0,
            epsilon_adelic=1.0,
            prime_weights={2: 0.25, 3: 0.15, 5: 0.1},
            precision="audit64",
            xp=np,
            phase_mode="canonical_device",
            compute_phase=True,
            actualization_config=config,
        ),
    )


def test_zero_extension_uses_exact_baseline_outputs() -> None:
    batch = _batch()
    legacy = _run(batch, None)
    configured = _run(batch, ActualizationExtensionConfig())
    for left, right in zip(legacy, configured, strict=True):
        np.testing.assert_array_equal(left, right)


def test_shadow_mode_keeps_baseline_authoritative() -> None:
    batch = _batch()
    baseline = _run(batch, None)
    configured = ActualizationExtensionConfig(
        variant="phase_interference",
        utility_coupling=0.2,
        phase_resonance_coupling=0.15,
        interference_mixer_strength=0.25,
        shadow_only=True,
    )
    result = decide_dense(
        batch,
        seed=9303,
        beta_intention=1.0,
        temperature=1.0,
        epsilon_adelic=1.0,
        prime_weights={2: 0.25, 3: 0.15, 5: 0.1},
        precision="audit64",
        xp=np,
        phase_mode="canonical_device",
        actualization_config=configured,
        return_extension_evidence=True,
    )
    for left, right in zip(baseline, result[:5], strict=True):
        np.testing.assert_array_equal(left, right)
    evidence = result[5]
    assert evidence["shadow_probabilities"] is not None
    assert not np.array_equal(evidence["shadow_probabilities"], baseline[2])
