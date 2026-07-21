from __future__ import annotations

from typing import Any, cast

import numpy as np


def bottom_up_weights(
    boundary: np.ndarray,
    coherence: np.ndarray,
    resource: np.ndarray,
    distances: np.ndarray,
    lambdas: np.ndarray | None = None,
    eta: float = 1.0,
    eps: float = 1e-12,
) -> np.ndarray:
    B = np.asarray(boundary, dtype=float)
    C = np.asarray(coherence, dtype=float)
    R = np.asarray(resource, dtype=float)
    D = np.asarray(distances, dtype=float)
    if not (B.shape == C.shape == R.shape == D.shape):
        raise ValueError("boundary/coherence/resource/distances must have matching shapes")
    L = np.ones_like(B) if lambdas is None else np.asarray(lambdas, dtype=float)
    raw = (
        np.maximum(L, 0)
        * np.maximum(B, 0)
        * np.maximum(C, 0)
        * np.maximum(R, 0)
        * np.exp(-eta * np.maximum(D, 0))
    )
    s = raw.sum()
    if s <= eps or not np.isfinite(s):
        return cast(np.ndarray, np.ones_like(raw) / len(raw))
    return cast(np.ndarray, raw / s)


def aggregate_records(weights: np.ndarray, records: np.ndarray) -> np.ndarray:
    W = np.asarray(weights, dtype=float)
    X = np.asarray(records, dtype=float)
    if X.shape[0] != W.shape[0]:
        raise ValueError("records first dimension must match weights")
    return cast(np.ndarray, W @ X)


def tissue_over_cell_demo() -> dict[str, Any]:
    boundary = np.array([0.4, 0.95])
    coherence = np.array([0.25, 0.98])
    resource = np.array([0.5, 0.9])
    distances = np.array([3.0, 0.4])
    lambdas = np.array([0.2, 1.5])
    W = bottom_up_weights(boundary, coherence, resource, distances, lambdas, eta=1.0)
    return {"cell": float(W[0]), "tissue": float(W[1]), "passed": bool(W[1] > W[0])}
