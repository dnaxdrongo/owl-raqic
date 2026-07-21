from __future__ import annotations

import numpy as np

from owl_raqic.gpu.actualization_extensions import orthogonal_utility_innovation


def test_utility_innovation_is_bounded_masked_and_score_orthogonal() -> None:
    scores = np.asarray([[0.0, 1.0, 2.0, 8.0], [1.0, 1.0, 1.0, 1.0]])
    utilities = np.asarray([[2.0, 0.0, 1.0, -99.0], [3.0, -1.0, 0.5, 0.0]])
    mask = np.asarray([[True, True, True, False], [True, True, True, True]])
    innovation, diagnostics = orthogonal_utility_innovation(
        scores,
        utilities,
        mask,
        epsilon=1e-8,
        bound_floor=1.0,
        xp=np,
        dtype=np.float64,
    )
    assert innovation[0, 3] == 0.0
    assert np.max(np.abs(innovation)) <= 1.0 + 1e-12
    assert np.max(np.abs(diagnostics["orthogonality_residual"])) < 1e-12
    assert np.all(innovation[1] == 0.0)


def test_utility_innovation_handles_nonfinite_and_single_legal_action() -> None:
    scores = np.asarray([[np.nan, 2.0]])
    utilities = np.asarray([[np.inf, -np.inf]])
    mask = np.asarray([[True, False]])
    innovation, _ = orthogonal_utility_innovation(
        scores,
        utilities,
        mask,
        epsilon=1e-8,
        bound_floor=1.0,
        xp=np,
        dtype=np.float64,
    )
    np.testing.assert_array_equal(innovation, np.zeros_like(scores))
