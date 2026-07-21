from __future__ import annotations

import numpy as np

from owl_raqic.gpu.actualization_extensions import ActualizationExtensionConfig
from owl_raqic.gpu.dense_types import RAQICDenseBatch
from owl_raqic.gpu.math_gpu import decide_dense

_ACTIONS = (
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
    rng = np.random.default_rng(744)
    n = 10
    features = rng.random((n, 11))
    bins = np.floor(features * 255).astype(np.int32)
    authority = rng.random((n, 22)) > 0.25
    authority[:, 0] = True
    return RAQICDenseBatch(
        ow_id=np.arange(n, dtype=np.int64),
        yx=np.column_stack((np.arange(n), np.zeros(n))).astype(np.int32),
        features=features,
        feature_bins=bins,
        adelic_codes=bins.copy(),
        authority_mask=authority,
        parent_intention=rng.random((n, 22)),
        alive_mask=np.ones(n, dtype=bool),
        scale_id=np.zeros(n, dtype=np.int32),
        tick=5,
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
        action_names=_ACTIONS,
        active_primes=(2, 3, 5),
        action_utilities=rng.normal(size=(n, 22)),
        parent_action_phase=rng.uniform(-np.pi, np.pi, size=(n, 22)),
        parent_action_coherence=rng.random((n, 22)),
    )


def _run(batch: RAQICDenseBatch, config: ActualizationExtensionConfig | None):
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


def test_shadow_uses_same_draw_and_does_not_mutate_inputs_or_authoritative_outputs() -> None:
    batch = _batch()
    snapshots = {
        "utilities": np.array(batch.action_utilities, copy=True),
        "parent": np.array(batch.parent_intention, copy=True),
        "phase": np.array(batch.parent_action_phase, copy=True),
        "coherence": np.array(batch.parent_action_coherence, copy=True),
    }
    baseline = _run(batch, None)
    config = ActualizationExtensionConfig(
        variant="phase_interference",
        utility_coupling=0.12,
        phase_resonance_coupling=0.08,
        interference_mixer_strength=0.05,
        interference_trotter_steps=2,
        shadow_only=True,
    )
    result = _run(batch, config)
    for expected, actual in zip(baseline, result, strict=True):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(batch.action_utilities, snapshots["utilities"])
    np.testing.assert_array_equal(batch.parent_intention, snapshots["parent"])
    np.testing.assert_array_equal(batch.parent_action_phase, snapshots["phase"])
    np.testing.assert_array_equal(batch.parent_action_coherence, snapshots["coherence"])


def test_shadow_evidence_arrays_do_not_alias_authoritative_arrays() -> None:
    batch = _batch()
    config = ActualizationExtensionConfig(
        variant="utility_innovation",
        utility_coupling=0.1,
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
        actualization_config=config,
        return_extension_evidence=True,
    )
    evidence = result[5]
    assert evidence["shadow_probabilities"] is not None
    assert not np.shares_memory(result[2], evidence["shadow_probabilities"])
    assert not np.shares_memory(result[3], evidence["shadow_readout"])
