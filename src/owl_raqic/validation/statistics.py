from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

import numpy as np


def total_variation(p: Any, q: Any) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    return float(0.5 * np.abs(p - q).sum())


def kl_divergence(p: Any, q: Any, eps: float = 1e-15) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / max(float(p.sum()), eps)
    q = q / max(float(q.sum()), eps)
    return float(np.sum(p * np.log(np.maximum(p, eps) / np.maximum(q, eps))))


@dataclass(frozen=True)
class ShotValidation:
    passed: bool
    total_variation: float
    allowance: float
    shots: int
    alpha: float


def shot_validation_pass(
    expected: Any,
    counts_or_probabilities: Any,
    shots: int,
    *,
    alpha: float = 0.01,
    max_tv: float = 0.02,
) -> ShotValidation:
    expected = np.asarray(expected, dtype=np.float64)
    observed = np.asarray(counts_or_probabilities, dtype=np.float64)
    if observed.sum() > 1.0 + 1e-9:
        observed = observed / float(shots)
    observed /= max(float(observed.sum()), 1e-15)
    expected /= max(float(expected.sum()), 1e-15)
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    # Conservative sum of marginal normal approximations.
    allowance = float(
        0.5 * z * np.sqrt(np.maximum(expected * (1.0 - expected), 0.0) / max(1, shots)).sum()
    )
    tv = total_variation(expected, observed)
    return ShotValidation(
        passed=bool(tv <= max_tv + allowance),
        total_variation=tv,
        allowance=allowance,
        shots=int(shots),
        alpha=float(alpha),
    )
