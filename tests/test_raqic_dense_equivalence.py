from __future__ import annotations

import numpy as np
import pytest

from owl_raqic.algorithms.feature_pipeline import compute_action_phases, compute_scores
from owl_raqic.algorithms.raqic_driver import RAQICDecisionEngine
from owl_raqic.config import ActivePlaceConfig, RAQICAlgorithmConfig, RAQICRegisterConfig
from owl_raqic.gpu.decision_engine import RAQICDenseDecisionEngine, RAQICDenseExecutionConfig
from owl_raqic.gpu.dense_types import RAQICDenseBatch
from owl_raqic.gpu.math_gpu import compute_phases_dense, compute_scores_dense, masked_softmax
from owl_raqic.types import RAQICActionSet, RAQICFeaturePacket

ACTION_NAMES = ("REST", "SENSE", "MOVE_N", "MOVE_S", "MOVE_E", "MOVE_W", "FEED", "REPAIR")
FEATURE_NAMES = (
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
)


def _config():
    return RAQICAlgorithmConfig(
        seed=123,
        beta_intention=1.2,
        action_temperature=0.75,
        epsilon_adelic=0.9,
        active_places=ActivePlaceConfig(
            primes=(2, 3, 5), prime_weights={2: 0.25, 3: 0.15, 5: 0.10}
        ),
        registers=RAQICRegisterConfig(n_actions=len(ACTION_NAMES), n_features=len(FEATURE_NAMES)),
    )


def _packets(n=6):
    rng = np.random.default_rng(7)
    packets = []
    for i in range(n):
        vals = rng.random(len(FEATURE_NAMES))
        features = {k: float(v) for k, v in zip(FEATURE_NAMES, vals, strict=True)}
        bins = {k: int(np.floor(v * 255)) for k, v in features.items()}
        phase_code = max(bins["phase"], 1)
        for a in range(len(ACTION_NAMES)):
            bins[f"phase_num_{a}"] = int((phase_code + a + 1) % 251 + 1)
            bins[f"phase_den_{a}"] = int((a + 2) * 257)
        mask = rng.random(len(ACTION_NAMES)) > 0.15
        if not mask.any():
            mask[0] = True
        intention = rng.random(len(ACTION_NAMES))
        intention /= intention.sum()
        packets.append(
            RAQICFeaturePacket(i, 0, 3, features, bins, intention, mask, {"y": i, "x": 0})
        )
    return packets


def _dense_from_packets(packets):
    feats = np.array(
        [[float(p.feature_bins[k]) for k in FEATURE_NAMES] for p in packets], dtype=float
    )
    bins = np.floor(np.clip(feats, 0, 1) * 255).astype(np.int32)
    return RAQICDenseBatch(
        ow_id=np.array([p.ow_id for p in packets], dtype=np.int64),
        yx=np.array(
            [[int(p.metadata["y"]), int(p.metadata["x"])] for p in packets], dtype=np.int32
        ),
        features=feats,
        feature_bins=bins,
        adelic_codes=bins.copy(),
        authority_mask=np.array([p.authority_mask for p in packets], dtype=bool),
        parent_intention=np.array([p.parent_intention for p in packets], dtype=float),
        alive_mask=np.ones(len(packets), dtype=bool),
        scale_id=np.zeros(len(packets), dtype=np.int32),
        tick=3,
        feature_names=FEATURE_NAMES,
        action_names=ACTION_NAMES,
        active_primes=(2, 3, 5),
    )


def test_dense_scores_and_phases_match_scalar_packet_reference():
    cfg = _config()
    packets = _packets(5)
    batch = _dense_from_packets(packets)
    dense_scores = compute_scores_dense(
        batch.features,
        batch.feature_bins,
        ACTION_NAMES,
        cfg.active_places.primes,
        cfg.active_places.prime_weights,
        cfg.epsilon_adelic,
    )
    dense_phases = compute_phases_dense(batch.feature_bins, ACTION_NAMES, cfg.active_places.primes)
    for i, packet in enumerate(packets):
        assert np.allclose(dense_scores[i], compute_scores(packet, cfg, ACTION_NAMES), atol=1e-12)
        assert np.allclose(dense_phases[i], compute_action_phases(packet, cfg), atol=1e-12)


def test_dense_decision_matches_scalar_probabilities():
    cfg = _config()
    packets = _packets(6)
    batch = _dense_from_packets(packets)
    dense_engine = RAQICDenseDecisionEngine(
        RAQICDenseExecutionConfig(
            seed=cfg.seed,
            beta_intention=cfg.beta_intention,
            temperature=cfg.action_temperature,
            epsilon_adelic=cfg.epsilon_adelic,
            prime_weights=cfg.active_places.prime_weights,
            backend="numpy",
            precision="audit64",
        )
    )
    dense = dense_engine.decide_batch(batch).to_numpy()
    scalar_engine = RAQICDecisionEngine(cfg, RAQICActionSet(ACTION_NAMES))
    for i, packet in enumerate(packets):
        got = dense.probabilities[i]
        exp = scalar_engine.decide(packet, sample=False).action_probabilities
        assert np.allclose(got, exp, atol=1e-12)


def test_masked_softmax_extreme_logits_and_all_illegal_rest_repair():
    scores = np.array([[1000.0, -1000.0, 0.0], [5.0, 6.0, 7.0]])
    mask = np.array([[True, False, True], [False, False, False]])
    p = masked_softmax(scores, mask)
    assert np.all(np.isfinite(p))
    assert np.allclose(p.sum(axis=1), 1.0)
    assert p[0, 1] == 0.0
    assert p[1, 0] == pytest.approx(1.0)
