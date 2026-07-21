from __future__ import annotations

import numpy as np


def sample_action(probabilities: np.ndarray, seed: int | None = None) -> int:
    rng = np.random.default_rng(seed)
    p = np.asarray(probabilities, dtype=float)
    p = p / p.sum()
    return int(rng.choice(len(p), p=p))


def counts_from_probabilities(
    probabilities: np.ndarray, shots: int, seed: int | None = None
) -> dict[str, int]:
    rng = np.random.default_rng(seed)
    p = np.asarray(probabilities, dtype=float)
    draws = rng.choice(len(p), size=shots, p=p / p.sum())
    return {str(i): int((draws == i).sum()) for i in range(len(p))}
