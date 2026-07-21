import numpy as np

from owl.gpu.policies import stable_masked_policy


def test_stable_masked_policy_extreme_logits():
    logits = np.array([[1000.0, -1000.0, 0.0], [-1000.0, -1000.0, -1000.0]])
    mask = np.array([[True, True, False], [False, False, False]])
    out = stable_masked_policy(logits, mask, xp=np, rest_index=0, precision="audit64")
    assert np.all(np.isfinite(out.probabilities))
    assert np.allclose(out.probabilities.sum(axis=1), 1.0)
    assert out.probabilities[0, 2] == 0.0
    assert out.probabilities[1, 0] == 1.0
    assert int(out.repair_count) == 1


def test_stable_masked_policy_one_legal_action():
    logits = np.array([[0.0, 3.0, 4.0]])
    mask = np.array([[False, True, False]])
    out = stable_masked_policy(logits, mask, xp=np, rest_index=0)
    assert np.allclose(out.probabilities, [[0.0, 1.0, 0.0]])
