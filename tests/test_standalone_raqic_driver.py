import copy

import numpy as np

from owl_raqic import RAQICDecisionEngine, RAQICFeaturePacket
from owl_raqic.algorithms.batch_scheduler import group_by_signature
from owl_raqic.algorithms.recursive_loop import run_recursive_packets
from owl_raqic.config import RAQICAlgorithmConfig


def packet(**kwargs):
    defaults = {
        "ow_id": 1,
        "scale_id": 0,
        "tick": 0,
        "feature_bins": {
            "resource": 0.5,
            "risk": 0.2,
            "memory": 0.3,
            "coherence": 0.8,
            "boundary": 0.9,
            "signal": 0.1,
            "prediction_error": 0.2,
            "food": 0.7,
        },
        "parent_intention": np.ones(10) / 10,
    }
    defaults.update(kwargs)
    return RAQICFeaturePacket(**defaults)


def test_decide_one_probabilities():
    engine = RAQICDecisionEngine(RAQICAlgorithmConfig(rounds=2))
    result = engine.decide(packet())
    assert result.action_probabilities.shape == (10,)
    assert np.allclose(result.action_probabilities.sum(), 1)
    assert result.recovery_checks["kraus"]["complete"]
    assert result.recovery_checks["recursive_trace_preserved"]


def test_decide_sampled_action_name():
    engine = RAQICDecisionEngine(RAQICAlgorithmConfig(mode="dynamic", rounds=2, seed=10))
    result = engine.decide(packet(), sample=True)
    assert result.sampled_action is not None
    assert result.sampled_action_name is not None


def test_authority_mask_respected():
    mask = np.array([True] + [False] * 9)
    engine = RAQICDecisionEngine()
    result = engine.decide(packet(authority_mask=mask))
    assert np.allclose(result.action_probabilities, np.array([1] + [0] * 9))


def test_no_input_mutation():
    p = packet()
    before = copy.deepcopy(p)
    engine = RAQICDecisionEngine()
    _ = engine.decide(p)
    assert p == before


def test_decide_batch():
    packets = [packet(ow_id=i) for i in range(3)]
    engine = RAQICDecisionEngine()
    out = engine.decide_batch(packets)
    assert len(out) == 3


def test_group_by_signature():
    packets = [packet(ow_id=1), packet(ow_id=2), packet(ow_id=3, feature_bins={"resource": 0.1})]
    groups = group_by_signature(packets, 10)
    assert len(groups) == 2


def test_recursive_packets():
    packets = [packet(ow_id=1), packet(ow_id=2)]
    engine = RAQICDecisionEngine(RAQICAlgorithmConfig(rounds=1))
    traj = run_recursive_packets(engine, packets, rounds=2)
    assert len(traj) == 2
    assert len(traj[0]) == 2


def test_epsilon_zero_still_valid():
    engine = RAQICDecisionEngine(RAQICAlgorithmConfig(epsilon_adelic=0.0))
    result = engine.decide(packet())
    assert np.allclose(result.action_probabilities.sum(), 1)
