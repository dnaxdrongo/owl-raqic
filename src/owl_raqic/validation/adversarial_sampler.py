from __future__ import annotations

from typing import Any

import numpy as np


def adversarial_validation_indices(
    probabilities: Any,
    authority_mask: Any | None = None,
    *,
    limit: int = 32,
    parent_intention: Any | None = None,
) -> np.ndarray:
    """Select a deterministic mixture of difficult and representative rows."""
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or p.shape[0] == 0 or limit <= 0:
        return np.empty((0,), dtype=np.int64)
    n = p.shape[0]
    eps = 1e-15
    entropy = -np.sum(np.where(p > 0, p * np.log(np.maximum(p, eps)), 0.0), axis=1)
    confidence = np.max(p, axis=1)
    chosen: list[int] = []
    candidates = [
        int(np.argmax(entropy)),
        int(np.argmin(entropy)),
        int(np.argmax(confidence)),
        int(np.argmin(confidence)),
    ]
    if authority_mask is not None:
        legal = np.sum(np.asarray(authority_mask, dtype=bool), axis=1)
        one = np.flatnonzero(legal == 1)
        if one.size:
            candidates.append(int(one[0]))
        repaired = np.flatnonzero(legal == 0)
        if repaired.size:
            candidates.append(int(repaired[0]))
    if parent_intention is not None:
        concentration = np.max(np.asarray(parent_intention, dtype=float), axis=1)
        candidates.extend([int(np.argmax(concentration)), int(np.argmin(concentration))])
    # Include endpoints and evenly spaced rows for patch/world coverage.
    candidates.extend([0, n - 1])
    candidates.extend(np.linspace(0, n - 1, min(n, limit), dtype=int).tolist())
    for i in candidates:
        if i not in chosen:
            chosen.append(i)
        if len(chosen) >= limit:
            break
    return np.asarray(chosen, dtype=np.int64)
