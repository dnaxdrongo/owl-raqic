from __future__ import annotations

import numpy as np
import pytest

from owl.cadc.doubly_robust import doubly_robust_action_values


def test_doubly_robust_correction_changes_only_factual_action() -> None:
    policy = np.full((2, 22), 1.0 / 22.0)
    nuisance = np.zeros((2, 22))
    result = doubly_robust_action_values(
        observed_action=np.asarray([1, 21]),
        observed_outcome=np.asarray([2.0, -1.0]),
        behavior_probability=policy,
        crossfit_outcome_prediction=nuisance,
        legal_executable_mask=np.ones((2, 22), dtype=bool),
        propensity_floor=0.01,
    )
    assert result.estimates[0, 1] == 44.0
    assert result.estimates[1, 21] == -22.0
    assert np.count_nonzero(result.estimates) == 2


def test_doubly_robust_rejects_nonoverlap_and_bad_normalization() -> None:
    nuisance = np.zeros((1, 22))
    support = np.ones((1, 22), dtype=bool)
    policy = np.full((1, 22), 1.0 / 22.0)
    policy[0, 0] = 0.0
    with pytest.raises(ValueError, match="normalize"):
        doubly_robust_action_values(
            observed_action=np.asarray([0]),
            observed_outcome=np.asarray([1.0]),
            behavior_probability=policy,
            crossfit_outcome_prediction=nuisance,
            legal_executable_mask=support,
        )
