from __future__ import annotations

from typing import cast

import numpy as np


def stable_softmax(
    scores: np.ndarray, temperature: float = 1.0, mask: np.ndarray | None = None
) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    z = scores / temperature
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != scores.shape:
            raise ValueError("mask shape must match scores")
        if not np.any(mask):
            out = np.zeros_like(scores, dtype=float)
            out[0] = 1.0
            return out
        z = np.where(mask, z, -1e12)
    z = z - np.max(z)
    e = np.exp(z)
    if mask is not None:
        e = np.where(mask, e, 0.0)
    s = e.sum()
    if s <= 0 or not np.isfinite(s):
        raise ValueError("softmax failed normalization")
    return cast(np.ndarray, e / s)


def normalize_intention(intention: np.ndarray | None, n_actions: int | None = None) -> np.ndarray:
    if intention is None:
        if n_actions is None:
            raise ValueError("n_actions is required when I is None")
        return np.ones(n_actions, dtype=float) / n_actions
    intention = np.asarray(intention, dtype=float)
    if np.any(intention < 0):
        intention = np.maximum(intention, 0.0)
    s = intention.sum()
    if s <= 0 or not np.isfinite(s):
        return np.ones_like(intention, dtype=float) / len(intention)
    return cast(np.ndarray, intention / s)


def update_parent_intention(
    prev_I: np.ndarray, aggregate_scores: np.ndarray, eta: float = 0.25, temperature: float = 1.0
) -> np.ndarray:
    if not (0 <= eta <= 1):
        raise ValueError("eta must be in [0,1]")
    prev = normalize_intention(prev_I)
    target = stable_softmax(aggregate_scores, temperature=temperature)
    return normalize_intention((1.0 - eta) * prev + eta * target)


def apply_top_down_bias(
    scores: np.ndarray, intention: np.ndarray | None, beta: float = 1.0
) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    intention = (
        normalize_intention(intention, len(scores))
        if intention is not None
        else np.zeros_like(scores)
    )
    if len(intention) != len(scores):
        J = np.ones_like(scores) / len(scores)
        m = min(len(intention), len(scores))
        J[:m] = intention[:m]
        intention = normalize_intention(J)
    return scores + beta * intention
