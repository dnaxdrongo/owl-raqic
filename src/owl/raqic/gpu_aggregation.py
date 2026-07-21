from __future__ import annotations

import numpy as np

from owl.core.actions import Action


def aggregate_records_to_patches_dense(
    probabilities: np.ndarray, weights: np.ndarray, patch_size: int, eps: float = 1e-12
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized patch aggregation reference for RAQIC probabilities.

    Works with NumPy arrays and is mirrored by the CuPy implementation through
    array namespace compatibility in future extensions.
    """
    h, w, actions = probabilities.shape
    if h % patch_size or w % patch_size:
        raise ValueError("grid shape must be divisible by patch_size")
    ph, pw = h // patch_size, w // patch_size
    p = probabilities.reshape(ph, patch_size, pw, patch_size, actions)
    wt = weights.reshape(ph, patch_size, pw, patch_size)
    denom = np.sum(wt, axis=(1, 3), keepdims=False)
    raw = np.sum(p * wt[..., None], axis=(1, 3))
    out = np.divide(raw, denom[..., None], out=np.zeros_like(raw), where=denom[..., None] > eps)
    bad = denom <= eps
    if np.any(bad):
        out[bad, :] = 0.0
        out[bad, int(Action.REST)] = 1.0
    sums = out.sum(axis=-1, keepdims=True)
    out = np.divide(out, sums, out=np.zeros_like(out), where=sums > eps)
    return out, denom
