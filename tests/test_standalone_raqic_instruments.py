import numpy as np

from owl_raqic.math.checks import (
    check_density_matrix,
    check_kraus_completeness,
    check_projector_partition,
    check_top_down_bias,
    check_unitarity,
)
from owl_raqic.math.instruments import (
    action_amplitudes,
    feedback_unitaries,
    preparation_kraus_from_amplitudes,
    recursive_channel,
    simulate_recursive_ensemble,
)
from owl_raqic.math.states import density_from_state, ket0


def test_action_amplitudes_normalize():
    amps, probs = action_amplitudes(np.array([0.0, 1.0, -0.5]))
    assert np.allclose(np.vdot(amps, amps), 1)
    assert np.allclose(probs.sum(), 1)


def test_action_mask_respected():
    amps, probs = action_amplitudes(np.array([0.0, 5.0, -0.5]), mask=np.array([True, False, True]))
    assert probs[1] == 0
    assert np.allclose(probs.sum(), 1)


def test_intention_bias_increases_target_not_forced():
    scores = np.zeros(4)
    target = np.array([0.0, 0.0, 1.0, 0.0])
    _, p0 = action_amplitudes(scores)
    _, pI = action_amplitudes(scores, intention=target, beta_intention=2.0)
    assert check_top_down_bias(p0, pI, 2)["passed"]


def test_kraus_completeness():
    amps, _ = action_amplitudes(np.array([0.1, 0.2, 0.3]))
    kraus, U, projs = preparation_kraus_from_amplitudes(amps)
    assert check_kraus_completeness(kraus)["complete"]
    assert check_unitarity(U)["unitary"]
    assert check_projector_partition(projs)["partition"]


def test_recursive_channel_trace_and_positive():
    amps, _ = action_amplitudes(np.array([0.1, 0.2, 0.3]))
    kraus, _, _ = preparation_kraus_from_amplitudes(amps)
    feedback = feedback_unitaries(len(amps))
    rho = density_from_state(ket0(len(amps)))
    out = recursive_channel(kraus, feedback, rho)
    chk = check_density_matrix(out)
    assert chk["passed"]


def test_recursive_ensemble_many_rounds():
    amps, _ = action_amplitudes(np.array([0.2, -0.1, 0.7, 0.3]))
    rec = simulate_recursive_ensemble(amps, rounds=5)
    assert np.allclose(rec["traces"], 1.0)
    assert np.all(rec["min_eigenvalues"] >= -1e-10)


def test_born_probabilities_match_amplitudes():
    scores = np.array([0.3, 0.1, -0.5])
    amps, probs = action_amplitudes(scores)
    kraus, _, _ = preparation_kraus_from_amplitudes(amps)
    rho = density_from_state(ket0(len(amps)))
    observed = np.array([np.trace(K @ rho @ K.conj().T).real for K in kraus])
    assert np.allclose(observed, probs)


def test_feedback_unitaries_unitary():
    for U in feedback_unitaries(3):
        assert check_unitarity(U)["unitary"]
