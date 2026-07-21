"""Vectorized NumPy kernels for neighborhood, probability, and phase operations.

These kernels are intentionally independent of engine modules. They operate on
plain arrays so later environment, communication, phase, integration, utility,
and actualization code can share one audited numerical layer.
"""

from __future__ import annotations

from typing import cast

import numpy as np


def _require_spatial(field: np.ndarray, name: str = "field") -> np.ndarray:
    """Return ``field`` as an array after validating at least two spatial axes."""
    array = np.asarray(field)
    if array.ndim < 2:
        raise ValueError(f"{name} must have at least two spatial axes, got shape {array.shape}")
    return array


def laplacian_wrap(field: np.ndarray) -> np.ndarray:
    """Return the toroidal four-neighbor discrete Laplacian.

    Parameters
    ----------
    field:
        Cell-level or channel field with shape ``(height, width, ...)``.
        The first two axes are treated as spatial axes and any trailing axes
        are preserved.

    Returns
    -------
    np.ndarray
        Array with the same shape as ``field``. Positive values indicate a
        local deficit relative to cardinal neighbors; negative values indicate
        local excess. Boundary behavior is toroidal via ``np.roll``.
    """
    x = _require_spatial(field)
    return cast(
        np.ndarray,
        (
            np.roll(x, 1, axis=0)
            + np.roll(x, -1, axis=0)
            + np.roll(x, 1, axis=1)
            + np.roll(x, -1, axis=1)
            - 4.0 * x
        ),
    )


def neighbor_mean_wrap(field: np.ndarray) -> np.ndarray:
    """Return the toroidal Moore-neighborhood mean.

    Parameters
    ----------
    field:
        Cell-level or channel field with shape ``(height, width, ...)``.

    Returns
    -------
    np.ndarray
        Same shape as ``field``. Each spatial location is replaced by the
        average of its eight nearest toroidal neighbors. The center value is
        not included.
    """
    x = _require_spatial(field)
    total = (
        np.roll(x, 1, axis=0)
        + np.roll(x, -1, axis=0)
        + np.roll(x, 1, axis=1)
        + np.roll(x, -1, axis=1)
        + np.roll(np.roll(x, 1, axis=0), 1, axis=1)
        + np.roll(np.roll(x, 1, axis=0), -1, axis=1)
        + np.roll(np.roll(x, -1, axis=0), 1, axis=1)
        + np.roll(np.roll(x, -1, axis=0), -1, axis=1)
    )
    return cast(np.ndarray, total / 8.0)


def gradient_wrap(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return toroidal central-difference gradients along spatial axes.

    Parameters
    ----------
    field:
        Cell-level or channel field with shape ``(height, width, ...)``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(grad_y, grad_x)``, each with the same shape as ``field``.
        The derivative is approximated as half the forward-minus-backward
        toroidal difference.
    """
    x = _require_spatial(field)
    grad_y = 0.5 * (np.roll(x, -1, axis=0) - np.roll(x, 1, axis=0))
    grad_x = 0.5 * (np.roll(x, -1, axis=1) - np.roll(x, 1, axis=1))
    return grad_y, grad_x


def normalize_last_axis(values: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Normalize nonnegative values along the last axis.

    Negative entries are clipped to zero. Rows/slices whose clipped sum is too
    small are replaced with a uniform distribution so the result remains on the
    probability simplex.

    Parameters
    ----------
    values:
        Array with a nonempty last axis.
    epsilon:
        Numerical tolerance used to detect zero-sum slices.

    Returns
    -------
    np.ndarray
        Nonnegative array of the same shape as ``values`` that sums to one
        along the last axis.
    """
    x = np.asarray(values, dtype=np.float64)
    if x.ndim == 0:
        raise ValueError("values must have at least one axis")
    if x.shape[-1] == 0:
        raise ValueError("values last axis must be nonempty")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    y = np.maximum(x, 0.0)
    totals = np.sum(y, axis=-1, keepdims=True)
    normalized = np.divide(y, totals, out=np.zeros_like(y), where=totals > epsilon)

    zero_mask = totals <= epsilon
    if np.any(zero_mask):
        normalized = np.where(zero_mask, 1.0 / x.shape[-1], normalized)

    if np.issubdtype(np.asarray(values).dtype, np.floating):
        return cast(np.ndarray, normalized.astype(np.asarray(values).dtype, copy=False))
    return cast(np.ndarray, normalized.astype(np.float32))


def softmax_stable(logits: np.ndarray, axis: int = -1, epsilon: float = 1e-8) -> np.ndarray:
    """Return stable softmax probabilities along ``axis``.

    The implementation subtracts the maximum logit before exponentiation,
    preventing overflow for large positive logits. The result is normalized
    along ``axis`` and has the same shape as ``logits``.

    Parameters
    ----------
    logits:
        Finite action-score array.
    axis:
        Axis over which probabilities are normalized.
    epsilon:
        Positive tolerance protecting against degenerate denominators.

    Returns
    -------
    np.ndarray
        Probability array with values in ``[0, 1]`` and sums approximately one
        along ``axis``.
    """
    x = np.asarray(logits, dtype=np.float64)
    if x.ndim == 0:
        raise ValueError("logits must have at least one axis")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if not np.all(np.isfinite(x)):
        raise ValueError("logits must be finite")

    shifted = x - np.max(x, axis=axis, keepdims=True)
    expv = np.exp(shifted)
    denom = np.sum(expv, axis=axis, keepdims=True)
    probs = expv / np.maximum(denom, epsilon)

    if np.issubdtype(np.asarray(logits).dtype, np.floating):
        return cast(np.ndarray, probs.astype(np.asarray(logits).dtype, copy=False))
    return cast(np.ndarray, probs.astype(np.float32))


def sigmoid(values: np.ndarray | float) -> np.ndarray | float:
    """Return the logistic sigmoid of ``values`` in a numerically stable form.

    Parameters
    ----------
    values:
        Scalar or array-like input.

    Returns
    -------
    np.ndarray | float
        Values in ``[0, 1]`` with the same broadcast shape as the input. Scalar
        input returns a Python ``float``.
    """
    x = np.asarray(values, dtype=np.float64)
    out = np.empty_like(x, dtype=np.float64)

    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)

    if np.isscalar(values):
        return float(out)
    return out.astype(
        np.asarray(values).dtype
        if np.issubdtype(np.asarray(values).dtype, np.floating)
        else np.float32
    )


def circular_mean(phase: np.ndarray, axis: int | tuple[int, ...] | None = None) -> np.ndarray:
    """Return the circular mean of phase angles.

    Parameters
    ----------
    phase:
        Array of angles in radians.
    axis:
        Axis or axes over which to average. ``None`` averages all entries.

    Returns
    -------
    np.ndarray
        Angle(s) in radians in the range ``[-pi, pi]``. This is the standard
        complex-phase mean used later for synchrony and patch phase aggregation.
    """
    theta = np.asarray(phase)
    if theta.size == 0:
        raise ValueError("phase array must be nonempty")

    sin_mean = np.mean(np.sin(theta), axis=axis)
    cos_mean = np.mean(np.cos(theta), axis=axis)
    return cast(np.ndarray, np.arctan2(sin_mean, cos_mean))
