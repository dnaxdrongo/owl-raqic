"""Score candidates and decisions with frozen rules and mandatory abstention."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from owl.cadc.schema import ACTION_FAMILY_REGISTRY, AbstentionReason
from owl.cadc.support import SupportCalibrator


@dataclass(frozen=True)
class CandidateScores:
    """Calibrated candidate values, intervals, support, and immutable ranks."""

    outcome_mean: npt.NDArray[Any]
    scalar_value: npt.NDArray[Any]
    lower: npt.NDArray[Any]
    upper: npt.NDArray[Any]
    support_status: npt.NDArray[Any]
    abstention_reason: npt.NDArray[Any]
    predicted_rank: npt.NDArray[Any]


class CADCScorer:
    """Apply a frozen model, calibration, support, and abstention contract."""

    def __init__(
        self,
        model: Any,
        *,
        support: SupportCalibrator,
        model_version: str,
    ) -> None:
        self.model = model
        self.support = support
        self.model_version = model_version

    def score_candidates(
        self,
        model_output: Mapping[str, Any],
        *,
        executable_mask: Any,
        lower: Any,
        upper: Any,
        action_support: Any,
        repeat_support: Any,
    ) -> CandidateScores:
        """Score a fixed 22-slot batch and apply mandatory support abstention."""
        values = np.asarray(model_output["scalar_value"], dtype=np.float64)
        outcomes = np.asarray(model_output["outcome_mean"])
        embeddings = np.asarray(model_output["embedding"])
        disagreement = np.asarray(model_output["epistemic_disagreement"])
        valid = np.asarray(executable_mask, dtype=bool)
        low = np.asarray(lower, dtype=np.float64)
        high = np.asarray(upper, dtype=np.float64)
        if values.shape != valid.shape or values.ndim != 2 or values.shape[1] != 22:
            raise ValueError("candidate score tensors must have shape [B,22]")
        statuses = np.full(values.shape, "nonexecutable", dtype=object)
        reasons = np.full(
            values.shape, AbstentionReason.LOW_ACTION_SUPPORT.value, dtype=object
        )
        if valid.any():
            support = self.support.decide_batch(
                embeddings[valid],
                action_support=np.asarray(action_support)[valid],
                repeat_support=np.asarray(repeat_support)[valid],
                disagreement=disagreement[valid],
                interval_width=(high - low)[valid],
            )
            statuses[valid] = support["support_status"]
            reasons[valid] = support["abstention_reason"]
        ranks = np.full(values.shape, -1, dtype=np.int16)
        order = np.argsort(
            -np.where(valid, values, -np.inf), axis=1, kind="stable"
        )
        ordinal = np.broadcast_to(
            np.arange(1, values.shape[1] + 1, dtype=np.int16), values.shape
        )
        np.put_along_axis(ranks, order, ordinal, axis=1)
        ranks[~valid] = -1
        return CandidateScores(outcomes, values, low, high, statuses, reasons, ranks)


def score_candidates(scorer: CADCScorer, *args: Any, **kwargs: Any) -> CandidateScores:
    """Call the scorer through the stable module-level inference entry point."""
    return scorer.score_candidates(*args, **kwargs)


def score_decisions(
    scores: CandidateScores, selected_action: Any, factual_value: Any | None = None
) -> dict[str, npt.NDArray[Any]]:
    """Reduce candidate scores to selected/best action decision diagnostics."""
    selected = np.asarray(selected_action, dtype=np.int64)
    rows = np.arange(selected.size)
    best = np.argmax(scores.scalar_value, axis=1)
    output = {
        "selected_action": selected,
        "selected_rank": scores.predicted_rank[rows, selected],
        "predicted_best_action": best,
        "predicted_regret": scores.scalar_value[rows, best]
        - scores.scalar_value[rows, selected],
        "support_status": scores.support_status[rows, selected],
        "abstention_reason": scores.abstention_reason[rows, selected],
        "selected_action_family": np.asarray(
            [ACTION_FAMILY_REGISTRY[value].primary_family.value for value in selected]
        ),
    }
    if factual_value is not None:
        output["factual_value"] = np.asarray(factual_value)
    return output
