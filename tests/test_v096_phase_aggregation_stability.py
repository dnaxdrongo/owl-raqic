from __future__ import annotations

import numpy as np
import pytest

from owl.engine.aggregation import _patch_phase_statistics
from owl.kernels.circular import weighted_patch_circular_statistics


def test_exact_cancellation_has_neutral_phase_and_synchrony() -> None:
    phase = np.array([[0.0, np.pi], [0.0, np.pi]], dtype=np.float32)
    weights = np.ones_like(phase)
    patch_phase, synchrony, resultant, supported = _patch_phase_statistics(
        phase,
        weights,
        2,
        resultant_support_epsilon=1e-7,
    )
    assert not bool(supported[0, 0])
    assert resultant[0, 0] <= 1e-7
    assert patch_phase[0, 0] == 0.0
    assert synchrony[0, 0] == 0.0


def test_coherent_patch_preserves_phase_and_squared_resultant() -> None:
    phase = np.full((2, 2), np.float32(1.25), dtype=np.float32)
    weights = np.ones_like(phase)
    patch_phase, synchrony, resultant, supported = _patch_phase_statistics(
        phase,
        weights,
        2,
        resultant_support_epsilon=1e-7,
    )
    assert bool(supported[0, 0])
    assert patch_phase[0, 0] == pytest.approx(1.25, abs=2e-7)
    assert resultant[0, 0] == pytest.approx(1.0, abs=1e-12)
    assert synchrony[0, 0] == pytest.approx(1.0, abs=1e-7)


def test_wraparound_mean_is_near_zero_not_pi() -> None:
    phase = np.array(
        [[2.0 * np.pi - 1e-4, 1e-4], [2.0 * np.pi - 2e-4, 2e-4]],
        dtype=np.float32,
    )
    weights = np.ones_like(phase)
    patch_phase, _, _, supported = _patch_phase_statistics(
        phase,
        weights,
        2,
        resultant_support_epsilon=1e-7,
    )
    assert bool(supported[0, 0])
    distance_to_zero = min(float(patch_phase[0, 0]), 2.0 * np.pi - float(patch_phase[0, 0]))
    assert distance_to_zero < 5e-7


def test_zero_weight_is_unsupported() -> None:
    phase = np.ones((4, 4), dtype=np.float32)
    weights = np.zeros_like(phase)
    patch_phase, synchrony, resultant, supported = _patch_phase_statistics(
        phase,
        weights,
        2,
        resultant_support_epsilon=1e-7,
    )
    assert not np.any(supported)
    np.testing.assert_array_equal(patch_phase, 0.0)
    np.testing.assert_array_equal(synchrony, 0.0)
    np.testing.assert_array_equal(resultant, 0.0)


def test_backend_neutral_numpy_contract_matches_cpu_wrapper() -> None:
    rng = np.random.default_rng(9601)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=(10, 10)).astype(np.float32)
    weights = rng.random((10, 10), dtype=np.float32)
    expected = _patch_phase_statistics(
        phase,
        weights,
        5,
        resultant_support_epsilon=1e-7,
    )
    actual = weighted_patch_circular_statistics(
        phase,
        weights,
        5,
        np,
        resultant_support_epsilon=1e-7,
    )
    for left, right in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(left, right)


def test_support_epsilon_is_validated() -> None:
    phase = np.zeros((2, 2), dtype=np.float32)
    weights = np.ones_like(phase)
    with pytest.raises(ValueError, match="resultant_support_epsilon"):
        weighted_patch_circular_statistics(
            phase,
            weights,
            2,
            np,
            resultant_support_epsilon=1.1,
        )
