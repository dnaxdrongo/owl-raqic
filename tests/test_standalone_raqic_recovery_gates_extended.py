from fractions import Fraction

import numpy as np
import pytest

from owl_raqic.adelic.padic import padic_abs_rational, product_formula_rational
from owl_raqic.algorithms.backend_selector import select_backend
from owl_raqic.algorithms.sampling import counts_from_probabilities, sample_action
from owl_raqic.config import ActivePlaceConfig, RAQICAlgorithmConfig
from owl_raqic.math.checks import check_density_matrix, check_kraus_completeness, check_unitarity
from owl_raqic.math.instruments import (
    action_amplitudes,
    feedback_unitaries,
    householder_unitary_from_state,
    preparation_kraus_from_amplitudes,
    recursive_channel,
)
from owl_raqic.math.states import (
    ket0,
    maximally_mixed,
    pad_to_power_of_two,
    probabilities_from_state,
)


def test_active_place_rejects_nonprime():
    with pytest.raises(ValueError):
        ActivePlaceConfig(primes=(2, 4))


def test_algorithm_config_validates_temperature():
    with pytest.raises(ValueError):
        RAQICAlgorithmConfig(action_temperature=0)


def test_product_formula_multiple_rationals():
    for n, d in [(2, 3), (18, 35), (-45, 28), (121, 50)]:
        assert product_formula_rational(n, d) == Fraction(1, 1)


def test_padic_abs_known_values():
    assert padic_abs_rational(8, 9, 2) == Fraction(1, 8)
    assert padic_abs_rational(8, 9, 3) == Fraction(9, 1)


def test_householder_maps_zero_to_target():
    amps, _ = action_amplitudes(np.array([0.1, 0.4, -0.2, 0.0]))
    U = householder_unitary_from_state(amps)
    assert check_unitarity(U)["unitary"]
    assert np.allclose(U @ ket0(len(amps)), amps, atol=1e-8)


def test_kraus_probabilities_sum_for_maximally_mixed():
    amps, _ = action_amplitudes(np.array([0.1, 0.4, -0.2]))
    kraus, _, _ = preparation_kraus_from_amplitudes(amps)
    rho = maximally_mixed(len(amps))
    probs = np.array([np.trace(K @ rho @ K.conj().T).real for K in kraus])
    assert np.allclose(probs.sum(), 1)


def test_recursive_channel_preserves_maximally_mixed_trace():
    amps, _ = action_amplitudes(np.array([0.1, 0.4, -0.2]))
    kraus, _, _ = preparation_kraus_from_amplitudes(amps)
    feedback = feedback_unitaries(len(amps))
    out = recursive_channel(kraus, feedback, maximally_mixed(len(amps)))
    assert check_density_matrix(out)["passed"]


def test_probabilities_from_state():
    p = probabilities_from_state(np.array([1, 1j]))
    assert np.allclose(p, [0.5, 0.5])


def test_pad_to_power_of_two():
    out = pad_to_power_of_two(np.array([1, 1, 1], dtype=complex))
    assert len(out) == 4
    assert np.allclose(np.vdot(out, out), 1)


def test_sample_action_reproducible():
    p = np.array([0.2, 0.8])
    assert sample_action(p, seed=2) == sample_action(p, seed=2)


def test_counts_sum_to_shots():
    counts = counts_from_probabilities(np.array([0.25, 0.75]), 100, seed=1)
    assert sum(counts.values()) == 100


def test_backend_selector_cpu_audit():
    prof = select_backend("cpu_audit")
    assert prof.name == "cpu_audit"
    assert not prof.qiskit_required


def test_backend_selector_auto_returns_valid():
    prof = select_backend("auto")
    assert prof.name in ("cpu_audit", "cpu_statevector")


def test_forbidden_all_mask_returns_rest_probability():
    amps, probs = action_amplitudes(
        np.array([10.0, 9.0, 8.0]), mask=np.array([False, False, False])
    )
    assert np.allclose(probs, [1, 0, 0])


def test_action_temperature_changes_distribution():
    _, cold = action_amplitudes(np.array([0.0, 2.0]), temperature=0.5)
    _, hot = action_amplitudes(np.array([0.0, 2.0]), temperature=5.0)
    assert cold[1] > hot[1]


def test_feedback_unitaries_complete_shape():
    U = feedback_unitaries(5)
    assert len(U) == 5
    assert U[0].shape == (5, 5)


def test_kraus_completeness_for_complex_phases():
    amps, _ = action_amplitudes(np.array([0.0, 1.0, -0.5]), phases=np.array([0.1, 0.2, 0.3]))
    kraus, _, _ = preparation_kraus_from_amplitudes(amps)
    assert check_kraus_completeness(kraus)["complete"]


def test_density_check_rejects_bad_trace():
    rho = np.eye(2)
    assert not check_density_matrix(rho)["trace_one"]


def test_unitarity_rejects_nonunitary():
    assert not check_unitarity(np.array([[1, 1], [0, 1]], dtype=complex))["unitary"]


def test_config_epsilon_zero_allowed():
    cfg = RAQICAlgorithmConfig(epsilon_adelic=0.0)
    assert cfg.epsilon_adelic == 0.0
