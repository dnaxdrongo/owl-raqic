"""SciPy ndimage wrappers for non-toroidal boundaries and local filters.

These wrappers are intentionally small and validated. They isolate SciPy-specific
boundary behavior from engine modules so later code can choose NumPy toroidal
kernels or SciPy finite-boundary kernels through one layer.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from scipy import ndimage

_ALLOWED_MODES = {"reflect", "constant", "nearest", "mirror", "wrap"}


def _validate_mode(mode: str) -> str:
    if mode not in _ALLOWED_MODES:
        raise ValueError(
            f"unsupported ndimage mode {mode!r}; expected one of {sorted(_ALLOWED_MODES)}"
        )
    return mode


def _require_2d(field: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(field)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2-D field, got shape {array.shape}")
    return array


def convolve_field(field: np.ndarray, kernel: np.ndarray, mode: str = "reflect") -> np.ndarray:
    """Convolve a 2-D field with a kernel and configurable boundary mode.

    Parameters
    ----------
    field:
        Cell-level field with shape ``(height, width)``.
    kernel:
        2-D convolution kernel.
    mode:
        SciPy ndimage extension mode: ``reflect``, ``constant``, ``nearest``,
        ``mirror``, or ``wrap``.

    Returns
    -------
    np.ndarray
        Convolved field with the same shape and dtype as ``field`` when
        possible.
    """
    x = _require_2d(field, "field")
    k = _require_2d(kernel, "kernel")
    selected_mode = _validate_mode(mode)
    return cast(np.ndarray, ndimage.convolve(x, k, mode=selected_mode).astype(x.dtype, copy=False))


def diffuse_with_obstacles(
    field: np.ndarray,
    obstacle: np.ndarray,
    rate: float,
    mode: str = "reflect",
) -> np.ndarray:
    """Diffuse a 2-D field while freezing obstacle cells.

    Parameters
    ----------
    field:
        Cell-level scalar field with shape ``(height, width)``.
    obstacle:
        Boolean mask with the same shape. ``True`` cells retain their original
        values after diffusion.
    rate:
        Nonnegative diffusion coefficient.
    mode:
        SciPy ndimage boundary mode.

    Returns
    -------
    np.ndarray
        Diffused copy of ``field``. The input is not mutated.

    Notes
    -----
    This is an baseline obstacle model: obstacle cells are frozen, and non-obstacle
    cells diffuse through the surrounding finite-difference Laplacian. Later
    passes can refine blocked-flux behavior if needed.
    """
    x = _require_2d(field, "field")
    mask = np.asarray(obstacle, dtype=np.bool_)
    if mask.shape != x.shape:
        raise ValueError(f"obstacle shape {mask.shape} must match field shape {x.shape}")
    if rate < 0 or not np.isfinite(rate):
        raise ValueError("rate must be finite and nonnegative")

    laplace_kernel = np.array(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float64
    )
    lap = ndimage.convolve(x.astype(np.float64), laplace_kernel, mode=_validate_mode(mode))
    out = x.astype(np.float64) + rate * lap
    out[mask] = x[mask]
    return cast(np.ndarray, out.astype(x.dtype, copy=False))


def local_mean(field: np.ndarray, radius: int, mode: str = "reflect") -> np.ndarray:
    """Compute a square-window local mean over a 2-D field.

    Parameters
    ----------
    field:
        Cell-level field with shape ``(height, width)``.
    radius:
        Neighborhood radius. ``0`` returns a copy of the input.
    mode:
        SciPy ndimage boundary mode.

    Returns
    -------
    np.ndarray
        Local mean field with the same shape as ``field``.
    """
    x = _require_2d(field, "field")
    if radius < 0:
        raise ValueError("radius must be nonnegative")
    if radius == 0:
        return x.copy()

    width = 2 * radius + 1
    kernel = np.full((width, width), 1.0 / (width * width), dtype=np.float64)
    out = ndimage.convolve(x.astype(np.float64), kernel, mode=_validate_mode(mode))
    return cast(np.ndarray, out.astype(x.dtype, copy=False))
