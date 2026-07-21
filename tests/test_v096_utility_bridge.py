from __future__ import annotations

import numpy as np
import pytest

from owl_raqic.gpu.actualization_extensions import (
    ActualizationExtensionConfig,
    apply_actualization_extensions,
    masked_standardize,
    orthogonal_utility_innovation,
)


def _innovation(scores: np.ndarray, utility: np.ndarray, mask: np.ndarray) -> np.ndarray:
    value, diagnostics = orthogonal_utility_innovation(
        scores,
        utility,
        mask,
        epsilon=1e-8,
        bound_floor=1.0,
        xp=np,
        dtype=np.float64,
    )
    assert np.max(np.abs(diagnostics["orthogonality_residual"])) <= 2e-14
    return value


def test_bridge_is_legal_bounded_and_score_orthogonal() -> None:
    scores = np.asarray([[0.0, 1.0, 2.0, 99.0]])
    utilities = np.asarray([[2.0, -1.0, 0.25, -999.0]])
    legal = np.asarray([[True, True, True, False]])
    innovation = _innovation(scores, utilities, legal)
    assert innovation[0, 3] == 0.0
    assert np.max(np.abs(innovation)) <= 1.0
    standardized_score = masked_standardize(scores, legal, epsilon=1e-8, xp=np, dtype=np.float64)
    assert abs(float(np.sum(standardized_score * innovation))) <= 2e-14


def test_parallel_utility_is_removed_and_orthogonal_pattern_is_retained() -> None:
    legal = np.ones((1, 3), dtype=bool)
    scores = np.asarray([[-1.0, 0.0, 1.0]])
    parallel = scores.copy()
    parallel_result = _innovation(scores, parallel, legal)
    np.testing.assert_allclose(parallel_result, 0.0, atol=2e-15)

    orthogonal = np.asarray([[1.0, -2.0, 1.0]])
    orthogonal_result = _innovation(scores, orthogonal, legal)
    assert np.linalg.norm(orthogonal_result) > 0.1


def test_degenerate_and_nonfinite_rows_return_zero_at_low_level() -> None:
    masks = [
        np.asarray([[True, False, False]]),
        np.asarray([[True, True, True]]),
    ]
    score_rows = [
        np.asarray([[1.0, 2.0, 3.0]]),
        np.asarray([[1.0, 1.0, 1.0]]),
    ]
    utility_rows = [
        np.asarray([[np.nan, 0.0, 0.0]]),
        np.asarray([[1.0, 2.0, 3.0]]),
    ]
    for scores, utility, mask in zip(score_rows, utility_rows, masks, strict=True):
        result = _innovation(scores, utility, mask)
        np.testing.assert_array_equal(result, np.zeros_like(result))


def test_full_extension_rejects_wrong_width_and_nonfinite_input() -> None:
    score = np.asarray([[0.0, 1.0]])
    phase = np.zeros_like(score)
    authority = np.ones_like(score, dtype=bool)
    parent = np.full_like(score, 0.5)
    cfg = ActualizationExtensionConfig(variant="utility_innovation", utility_coupling=0.1)
    with pytest.raises(ValueError, match="utilities must match"):
        apply_actualization_extensions(
            score,
            phase,
            authority,
            parent,
            np.ones((1, 3)),
            None,
            None,
            beta_intention=1.0,
            temperature=1.0,
            config=cfg,
            edges=((0, 1),),
            xp=np,
            dtype=np.float64,
        )
    with pytest.raises(ValueError, match="finite"):
        apply_actualization_extensions(
            score,
            phase,
            authority,
            parent,
            np.asarray([[np.nan, 1.0]]),
            None,
            None,
            beta_intention=1.0,
            temperature=1.0,
            config=cfg,
            edges=((0, 1),),
            xp=np,
            dtype=np.float64,
        )


def test_full_extension_rejects_phase_context_shape_mismatch() -> None:
    score = np.asarray([[0.0, 1.0]])
    phase = np.zeros_like(score)
    authority = np.ones_like(score, dtype=bool)
    parent = np.full_like(score, 0.5)
    cfg = ActualizationExtensionConfig(
        variant="fractal_resonance",
        utility_coupling=0.1,
        phase_resonance_coupling=0.2,
    )
    with pytest.raises(ValueError, match="parent_action_phase must match"):
        apply_actualization_extensions(
            score,
            phase,
            authority,
            parent,
            np.zeros_like(score),
            np.zeros((1, 3)),
            np.ones_like(score),
            beta_intention=1.0,
            temperature=1.0,
            config=cfg,
            edges=((0, 1),),
            xp=np,
            dtype=np.float64,
        )
    with pytest.raises(ValueError, match="parent_action_coherence must match"):
        apply_actualization_extensions(
            score,
            phase,
            authority,
            parent,
            np.zeros_like(score),
            np.zeros_like(score),
            np.ones((2, 2)),
            beta_intention=1.0,
            temperature=1.0,
            config=cfg,
            edges=((0, 1),),
            xp=np,
            dtype=np.float64,
        )
