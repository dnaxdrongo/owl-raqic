"""Backend-neutral weighted circular aggregation primitives.

Circular phase is only meaningful when the weighted unit-vector resultant has
non-negligible magnitude.  These helpers calculate the phase and synchrony from
one shared float64 reduction contract and canonicalize unsupported phase to
zero.  The returned observables remain float32 because patch phase is physical
OWL state; only the reduction is promoted for numerical stability.
"""

from __future__ import annotations

from typing import Any

_TWO_PI = 2.0 * 3.141592653589793


def weighted_patch_circular_statistics(
    phase: Any,
    weights: Any,
    patch_size: int,
    xp: Any,
    *,
    resultant_support_epsilon: float,
) -> tuple[Any, Any, Any, Any]:
    """Return patch phase, synchrony, resultant, and support mask.

    Inputs are promoted to float64 before trigonometric evaluation and
    reduction.  This sharply reduces CPU/CUDA drift in low-coherence patches
    while preserving the established float32 physical-state contract.
    Unsupported circular means (zero weight or negligible resultant) use the
    neutral phase/synchrony value zero.
    """
    patch = int(patch_size)
    if patch <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size!r}")
    if phase.shape != weights.shape or len(phase.shape) != 2:
        raise ValueError("phase and weights must be matching two-dimensional fields")
    h, w = phase.shape
    if h % patch or w % patch:
        raise ValueError(f"field shape {(h, w)} must be exactly divisible by patch_size={patch}")
    support_epsilon = float(resultant_support_epsilon)
    if not 0.0 <= support_epsilon <= 1.0:
        raise ValueError("resultant_support_epsilon must be in [0, 1]")

    ph, pw = h // patch, w // patch
    phase64 = phase.astype(xp.float64, copy=False)
    weight64 = weights.astype(xp.float64, copy=False)
    phase_blocks = phase64.reshape(ph, patch, pw, patch).swapaxes(1, 2)
    weight_blocks = weight64.reshape(ph, patch, pw, patch).swapaxes(1, 2)

    denominator = xp.sum(weight_blocks, axis=(2, 3), dtype=xp.float64)
    sin_sum = xp.sum(
        xp.sin(phase_blocks) * weight_blocks,
        axis=(2, 3),
        dtype=xp.float64,
    )
    cos_sum = xp.sum(
        xp.cos(phase_blocks) * weight_blocks,
        axis=(2, 3),
        dtype=xp.float64,
    )
    safe_denominator = xp.maximum(denominator, xp.float64(1.0))
    sin_mean = xp.where(denominator > 0.0, sin_sum / safe_denominator, 0.0)
    cos_mean = xp.where(denominator > 0.0, cos_sum / safe_denominator, 0.0)
    resultant = xp.hypot(sin_mean, cos_mean)
    supported = (denominator > 0.0) & (resultant > xp.float64(support_epsilon))

    raw_phase = xp.mod(xp.arctan2(sin_mean, cos_mean), xp.float64(_TWO_PI))
    patch_phase = xp.where(supported, raw_phase, 0.0).astype(xp.float32)
    synchrony = xp.where(
        supported,
        xp.clip(resultant * resultant, 0.0, 1.0),
        0.0,
    ).astype(xp.float32)
    return patch_phase, synchrony, resultant, supported
