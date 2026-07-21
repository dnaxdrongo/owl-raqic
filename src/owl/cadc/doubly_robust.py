"""Provide optional cross-fitted doubly robust factual augmentation.

Direct paired counterfactual branches remain the primary CADC oracle. These
utilities accept cross-fitted nuisance predictions and do not estimate a behavior
policy on evaluation rows.

The estimator follows the doubly robust off-policy construction described by
Jiang and Li (2016); see ``docs/REFERENCES.md`` [R24].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _xp(value: Any) -> Any:
    if type(value).__module__.split(".", maxsplit=1)[0] == "cupy":
        import cupy as cp

        return cp
    return np


@dataclass(frozen=True)
class DoublyRobustResult:
    """Cross-fit action-value estimates and their overlap support mask."""
    estimates: Any
    overlap_mask: Any
    clipped_propensity: Any
    effective_sample_size: Any


def doubly_robust_action_values(
    *,
    observed_action: Any,
    observed_outcome: Any,
    behavior_probability: Any,
    crossfit_outcome_prediction: Any,
    legal_executable_mask: Any,
    propensity_floor: float = 0.01,
) -> DoublyRobustResult:
    """Return per-row/action augmented inverse-propensity estimates.

    Shapes are ``[N]`` for factual actions/outcomes and ``[N,22]`` for policy,
    nuisance, and support arrays.  Invalid policy normalization, factual support,
    overlap, or nonfinite inputs fail closed.
    """
    if not 0.0 < propensity_floor < 0.5:
        raise ValueError("propensity floor must lie inside (0,0.5)")
    xp = _xp(crossfit_outcome_prediction)
    action = xp.asarray(observed_action, dtype=xp.int64)
    outcome = xp.asarray(observed_outcome, dtype=xp.float64)
    probability = xp.asarray(behavior_probability, dtype=xp.float64)
    prediction = xp.asarray(crossfit_outcome_prediction, dtype=xp.float64)
    support = xp.asarray(legal_executable_mask, dtype=bool)
    if prediction.ndim != 2 or prediction.shape[1] != 22:
        raise ValueError("DR nuisance predictions must use the immutable 22-action axis")
    if probability.shape != prediction.shape or support.shape != prediction.shape:
        raise ValueError("DR policy, nuisance, and support arrays must have equal shapes")
    if action.shape != outcome.shape or action.size != prediction.shape[0]:
        raise ValueError("DR factual action/outcome rows do not align")
    if not bool(xp.all(xp.isfinite(prediction))):
        raise ValueError("DR nuisance predictions contain nonfinite values")
    if not bool(xp.all(xp.isfinite(probability))) or bool(xp.any(probability < 0.0)):
        raise ValueError("DR behavior probabilities are invalid")
    probability = xp.where(support, probability, 0.0)
    normalization = probability.sum(axis=1)
    if not bool(xp.all(xp.abs(normalization - 1.0) <= 1e-6)):
        raise ValueError("DR behavior probabilities do not normalize over supported actions")
    row = xp.arange(action.size, dtype=xp.int64)
    if bool(xp.any((action < 0) | (action >= 22))) or not bool(
        xp.all(support[row, action])
    ):
        raise ValueError("DR factual action is outside legal/executable support")
    factual_probability = probability[row, action]
    overlap = factual_probability >= propensity_floor
    if not bool(xp.all(overlap)):
        raise ValueError("DR factual behavior policy has insufficient overlap")
    clipped = xp.maximum(factual_probability, propensity_floor)
    residual = outcome - prediction[row, action]
    one_hot = xp.zeros_like(prediction)
    one_hot[row, action] = 1.0
    estimates = prediction + one_hot * (residual / clipped)[:, None]
    weights = 1.0 / clipped
    effective_sample_size = (weights.sum() ** 2) / xp.maximum(
        (weights**2).sum(), 1e-30
    )
    return DoublyRobustResult(estimates, overlap, clipped, effective_sample_size)
