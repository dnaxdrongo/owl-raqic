"""Forecast, ranking, survival, subgroup, and negative-control evaluation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt


def evaluate_forecast(prediction: Any, target: Any, mask: Any | None = None) -> dict[str, float]:
    """Compute finite masked regression diagnostics in float64."""
    predicted = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    valid = np.isfinite(predicted) & np.isfinite(truth)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    if not valid.any():
        raise ValueError("forecast metric has no valid rows")
    error = predicted[valid] - truth[valid]
    return {
        "rows": float(valid.sum()),
        "mae": float(np.mean(np.abs(error), dtype=np.float64)),
        "rmse": float(np.sqrt(np.mean(error**2, dtype=np.float64))),
        "bias": float(np.mean(error, dtype=np.float64)),
    }


def evaluate_rankings(scores: Any, values: Any, mask: Any) -> dict[str, float]:
    """Compute masked fixed-action ranking, regret, and pair-order metrics."""
    predicted = np.asarray(scores, dtype=np.float64)
    truth = np.asarray(values, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    if predicted.shape != truth.shape or predicted.shape != valid.shape:
        raise ValueError("ranking inputs must have identical shapes")
    if predicted.ndim != 2 or predicted.shape[1] != 22:
        raise ValueError("ranking tensors must have shape [B,22]")
    if np.any(valid.sum(axis=1) < 2):
        raise ValueError("ranking evaluation needs at least two executable candidates")
    negative = np.finfo(np.float64).min
    predicted_best = np.argmax(np.where(valid, predicted, negative), axis=1)
    true_best = np.argmax(np.where(valid, truth, negative), axis=1)
    row = np.arange(predicted.shape[0])
    regret = truth[row, true_best] - truth[row, predicted_best]
    pair_left, pair_right = np.triu_indices(22, k=1)
    pair_mask = valid[:, pair_left] & valid[:, pair_right]
    predicted_delta = predicted[:, pair_left] - predicted[:, pair_right]
    true_delta = truth[:, pair_left] - truth[:, pair_right]
    predicted_order = predicted_delta > 0.0
    true_order = true_delta > 0.0
    non_tie = true_delta != 0.0
    evaluated_pairs = pair_mask & non_tie
    pair_correct = int(np.sum((predicted_order == true_order) & evaluated_pairs))
    pair_total = int(evaluated_pairs.sum())
    probability = 1.0 / (1.0 + np.exp(-np.clip(predicted_delta, -60.0, 60.0)))
    label = true_order.astype(np.float64)
    pair_losses = -(
        label * np.log(np.clip(probability, 1e-12, 1.0))
        + (1.0 - label) * np.log(np.clip(1.0 - probability, 1e-12, 1.0))
    )
    pair_log_loss = float(pair_losses[evaluated_pairs].sum())
    topk: dict[int, float] = {}
    ndcg: dict[int, float] = {}
    candidate_count = valid.sum(axis=1)
    prediction_order = np.argsort(
        np.where(valid, -predicted, np.inf), axis=1, kind="stable"
    )
    truth_order = np.argsort(
        np.where(valid, -truth, np.inf), axis=1, kind="stable"
    )
    floor = np.min(np.where(valid, truth, np.inf), axis=1)
    row_axis = np.arange(predicted.shape[0])[:, None]
    for cutoff in (1, 3, 5):
        position = np.arange(cutoff)[None, :]
        selected_width = np.minimum(candidate_count, cutoff)[:, None]
        position_valid = position < selected_width
        predicted_top = prediction_order[:, :cutoff]
        truth_top = truth_order[:, :cutoff]
        topk[cutoff] = float(
            np.sum(
                (predicted_top == true_best[:, None]) & position_valid,
                dtype=np.float64,
            )
        )
        relevance = truth[row_axis, predicted_top] - floor[:, None]
        ideal = truth[row_axis, truth_top] - floor[:, None]
        discount = 1.0 / np.log2(position.astype(np.float64) + 2.0)
        weighted_relevance = np.where(position_valid, relevance * discount, 0.0)
        weighted_ideal = np.where(position_valid, ideal * discount, 0.0)
        denominator = weighted_ideal.sum(axis=1, dtype=np.float64)
        ndcg_rows = np.divide(
            weighted_relevance.sum(axis=1, dtype=np.float64),
            denominator,
            out=np.ones_like(denominator),
            where=denominator > 1e-12,
        )
        ndcg[cutoff] = float(ndcg_rows.sum(dtype=np.float64))

    ascending_prediction = np.argsort(
        np.where(valid, predicted, np.inf), axis=1, kind="stable"
    )
    ascending_truth = np.argsort(
        np.where(valid, truth, np.inf), axis=1, kind="stable"
    )
    prediction_rank = np.empty_like(ascending_prediction)
    truth_rank = np.empty_like(ascending_truth)
    ordinal = np.broadcast_to(np.arange(22), ascending_prediction.shape)
    np.put_along_axis(prediction_rank, ascending_prediction, ordinal, axis=1)
    np.put_along_axis(truth_rank, ascending_truth, ordinal, axis=1)
    count = candidate_count.astype(np.float64)
    prediction_mean = (prediction_rank * valid).sum(axis=1) / count
    truth_mean = (truth_rank * valid).sum(axis=1) / count
    prediction_centered = (prediction_rank - prediction_mean[:, None]) * valid
    truth_centered = (truth_rank - truth_mean[:, None]) * valid
    covariance = (prediction_centered * truth_centered).sum(axis=1)
    scale = np.sqrt(
        (prediction_centered**2).sum(axis=1) * (truth_centered**2).sum(axis=1)
    )
    spearman_values = np.divide(
        covariance,
        scale,
        out=np.zeros_like(covariance, dtype=np.float64),
        where=scale > 0.0,
    )
    concordant = ((predicted_delta * true_delta) > 0.0) & pair_mask
    discordant = ((predicted_delta * true_delta) < 0.0) & pair_mask
    concordant_count = concordant.sum(axis=1)
    discordant_count = discordant.sum(axis=1)
    kendall_denominator = concordant_count + discordant_count
    kendall_values = np.divide(
        concordant_count - discordant_count,
        kendall_denominator,
        out=np.zeros_like(kendall_denominator, dtype=np.float64),
        where=kendall_denominator > 0,
    )
    kendall_defined = kendall_denominator > 0
    return {
        "decisions": float(predicted.shape[0]),
        "top1_accuracy": float(np.mean(predicted_best == true_best)),
        "mean_regret": float(np.mean(regret, dtype=np.float64)),
        "pairwise_accuracy": float(pair_correct / max(1, pair_total)),
        "pairwise_log_loss": float(pair_log_loss / max(1, pair_total)),
        "spearman": float(np.mean(spearman_values, dtype=np.float64)),
        "kendall_tau": (
            float(np.mean(kendall_values[kendall_defined], dtype=np.float64))
            if kendall_defined.any()
            else 0.0
        ),
        **{
            f"top{cutoff}_contains_best": float(value / predicted.shape[0])
            for cutoff, value in topk.items()
        },
        **{
            f"ndcg_at_{cutoff}": float(value / predicted.shape[0])
            for cutoff, value in ndcg.items()
        },
    }


def evaluate_survival(probability: Any, event: Any, mask: Any | None = None) -> dict[str, float]:
    """Compute masked binary survival probability diagnostics."""
    predicted = np.asarray(probability, dtype=np.float64)
    truth = np.asarray(event, dtype=np.float64)
    valid = np.isfinite(predicted) & np.isfinite(truth)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    if not valid.any():
        raise ValueError("survival metric has no valid rows")
    bounded = np.clip(predicted[valid], 0.0, 1.0)
    return {
        "rows": float(valid.sum()),
        "brier": float(np.mean((bounded - truth[valid]) ** 2, dtype=np.float64)),
        "mean_probability": float(np.mean(bounded, dtype=np.float64)),
        "event_rate": float(np.mean(truth[valid], dtype=np.float64)),
    }


@dataclass
class Phase4Evaluator:
    """Evaluate all rows and supported rows without deleting difficult examples."""

    support_mask: npt.NDArray[Any] | None = None

    def forecast(self, prediction: Any, target: Any) -> dict[str, dict[str, float]]:
        """Evaluate all rows and the optional frozen support subset."""
        result = {"all": evaluate_forecast(prediction, target)}
        if self.support_mask is not None:
            result["supported"] = evaluate_forecast(prediction, target, self.support_mask)
        return result


class NegativeControlRunner:
    """Deterministic safe-stratum label perturbations for required controls."""

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)

    def action_shuffle(self, actions: Any, strata: Any) -> npt.NDArray[Any]:
        """Permute action targets independently inside declared safe strata."""
        return self._within_strata(np.asarray(actions), np.asarray(strata), salt=11)

    def target_shuffle(self, targets: Any, compatible_strata: Any) -> npt.NDArray[Any]:
        """Permute targets only among compatible family/context/horizon rows."""
        return self._within_strata(
            np.asarray(targets), np.asarray(compatible_strata), salt=23
        )

    def repeat_mismatch(self, repeats: Any, source_ids: Any) -> npt.NDArray[Any]:
        """Break matched-repeat pairing within immutable source identities."""
        return self._within_strata(np.asarray(repeats), np.asarray(source_ids), salt=37)

    def temporal_break(
        self, outcomes: Any, source_ids: Any, strata: Any | None = None
    ) -> npt.NDArray[Any]:
        """Shift outcomes in deterministic temporal order within safe strata."""

        values = np.asarray(outcomes)
        sources = np.asarray(source_ids).astype(str)
        if values.shape[0] != sources.size:
            raise ValueError("temporal outcomes and source IDs do not align")
        labels = (
            np.zeros(sources.size, dtype=np.int8)
            if strata is None
            else np.asarray(strata).astype(str)
        )
        if labels.size != sources.size:
            raise ValueError("temporal strata and source IDs do not align")
        if values.shape[0] < 2:
            raise ValueError("temporal break needs at least two rows")
        result = values.copy()
        shifted_groups = 0
        for label in np.unique(labels):
            rows = np.flatnonzero(labels == label)
            if rows.size < 2:
                continue
            order = rows[np.argsort(sources[rows], kind="stable")]
            result[order] = np.roll(values[order], 1, axis=0)
            shifted_groups += 1
        if shifted_groups == 0:
            raise ValueError("temporal break has no stratum with at least two rows")
        return result

    def run_metric(
        self,
        metric: Callable[[Any, Any], Mapping[str, float]],
        prediction: Any,
        shuffled_target: Any,
    ) -> dict[str, float]:
        """Evaluate one supplied metric against a negative-control target."""
        return dict(metric(prediction, shuffled_target))

    def _within_strata(
        self,
        values: npt.NDArray[Any],
        strata: npt.NDArray[Any],
        *,
        salt: int,
    ) -> npt.NDArray[Any]:
        if values.shape[0] != strata.shape[0]:
            raise ValueError("negative-control values and strata do not align")
        result = values.copy()
        rng = np.random.default_rng(self.seed + salt)
        labels = strata.astype(str)
        for label in np.unique(labels):
            indices = np.flatnonzero(labels == label)
            if indices.size > 1:
                result[indices] = values[rng.permutation(indices)]
        return result
