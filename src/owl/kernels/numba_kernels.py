"""Numba-accelerated kernels for branch-heavy hot loops.

The public functions in this module are thin validation wrappers around compiled
kernels. They do not import engine modules or ``WorldState`` so the numerical
layer stays reusable and acyclic.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar, cast

import numpy as np
from numba import njit

P = ParamSpec("P")
R = TypeVar("R")


def typed_njit(*args: Any, **kwargs: Any) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Return Numba's decorator while preserving the Python signature for mypy."""
    return cast(Callable[[Callable[P, R]], Callable[P, R]], njit(*args, **kwargs))


@typed_njit(cache=True)
def _sample_categorical_grid_numba(
    probabilities: np.ndarray, random_values: np.ndarray
) -> np.ndarray:
    """Compiled categorical sampler for an ``H x W x K`` probability cube."""
    height, width, actions = probabilities.shape
    out = np.empty((height, width), dtype=np.int16)
    for y in range(height):
        for x in range(width):
            r = random_values[y, x]
            if r < 0.0:
                r = 0.0
            elif r > 1.0:
                r = 1.0

            acc = 0.0
            chosen = actions - 1
            for action in range(actions):
                acc += probabilities[y, x, action]
                if r <= acc:
                    chosen = action
                    break
            out[y, x] = chosen
    return out


def sample_categorical_grid(probabilities: np.ndarray, random_values: np.ndarray) -> np.ndarray:
    """Sample one action per grid cell from a probability cube.

    Parameters
    ----------
    probabilities:
        Array with shape ``(height, width, num_actions)``. Values are expected
        to be nonnegative and normalized along the last axis.
    random_values:
        Array with shape ``(height, width)`` containing values in ``[0, 1]``.

    Returns
    -------
    np.ndarray
        Integer readout field with shape ``(height, width)`` and dtype
        ``np.int16``.

    Notes
    -----
    This is the low-level stochastic actualization kernel. It does not repair
    invalid probabilities; callers should normalize before sampling.
    """
    probs = np.asarray(probabilities)
    rand = np.asarray(random_values)
    if probs.ndim != 3:
        raise ValueError(
            f"probabilities must have shape (height, width, actions), got {probs.shape}"
        )
    if rand.shape != probs.shape[:2]:
        raise ValueError(
            f"random_values shape {rand.shape} must match probabilities "
            f"spatial shape {probs.shape[:2]}"
        )
    if probs.shape[-1] <= 0:
        raise ValueError("probabilities must have a nonempty action axis")
    if not np.all(np.isfinite(probs)):
        raise ValueError("probabilities must be finite")
    if not np.all(np.isfinite(rand)):
        raise ValueError("random_values must be finite")
    return _sample_categorical_grid_numba(probs.astype(np.float64), rand.astype(np.float64))


@typed_njit(cache=True)
def _move_cells_2d(
    field: np.ndarray,
    source_y: np.ndarray,
    source_x: np.ndarray,
    target_y: np.ndarray,
    target_x: np.ndarray,
    clear_value: float,
) -> np.ndarray:
    out = field.copy()
    for i in range(source_y.size):
        sy = source_y[i]
        sx = source_x[i]
        ty = target_y[i]
        tx = target_x[i]
        out[ty, tx] = out[sy, sx]
        out[sy, sx] = clear_value
    return out


@typed_njit(cache=True)
def _move_cells_3d(
    field: np.ndarray,
    source_y: np.ndarray,
    source_x: np.ndarray,
    target_y: np.ndarray,
    target_x: np.ndarray,
    clear_value: float,
) -> np.ndarray:
    out = field.copy()
    channels = field.shape[2]
    for i in range(source_y.size):
        sy = source_y[i]
        sx = source_x[i]
        ty = target_y[i]
        tx = target_x[i]
        for c in range(channels):
            out[ty, tx, c] = out[sy, sx, c]
            out[sy, sx, c] = clear_value
    return out


def _validate_coordinate_vectors(*coords: np.ndarray) -> tuple[np.ndarray, ...]:
    arrays = tuple(np.asarray(c, dtype=np.int64).reshape(-1) for c in coords)
    if not arrays:
        raise ValueError("at least one coordinate vector is required")
    size = arrays[0].size
    if any(a.size != size for a in arrays):
        raise ValueError("all coordinate vectors must have the same length")
    return arrays


def _validate_in_bounds(y: np.ndarray, x: np.ndarray, height: int, width: int, label: str) -> None:
    if np.any(y < 0) or np.any(y >= height) or np.any(x < 0) or np.any(x >= width):
        raise ValueError(f"{label} coordinates must be inside field shape {(height, width)}")


def move_cells_kernel(
    field: np.ndarray,
    source_y: np.ndarray,
    source_x: np.ndarray,
    target_y: np.ndarray,
    target_x: np.ndarray,
    clear_value: float = 0.0,
) -> np.ndarray:
    """Move values in a 2-D or 3-D field according to source/target coordinates.

    Parameters
    ----------
    field:
        Array with shape ``(height, width)`` or ``(height, width, channels)``.
        The function returns a moved copy and does not mutate the input.
    source_y, source_x, target_y, target_x:
        One-dimensional or broadcastable coordinate arrays. Each coordinate
        pair defines one move operation. Coordinates are assumed already
        conflict-resolved by higher-level movement code.
    clear_value:
        Value written to the source location after a move.

    Returns
    -------
    np.ndarray
        Copy of ``field`` with moves applied. Dtype matches the input field.

    Notes
    -----
    This primitive is deliberately field-oriented rather than object-oriented.
    Later movement code can call it for each cell-owned array that must move
    with the OW identity.
    """
    arr = np.asarray(field)
    if arr.ndim not in (2, 3):
        raise ValueError(f"field must be 2-D or 3-D, got shape {arr.shape}")

    sy, sx, ty, tx = _validate_coordinate_vectors(source_y, source_x, target_y, target_x)
    height, width = arr.shape[:2]
    _validate_in_bounds(sy, sx, height, width, "source")
    _validate_in_bounds(ty, tx, height, width, "target")

    if arr.ndim == 2:
        return _move_cells_2d(arr, sy, sx, ty, tx, clear_value).astype(arr.dtype, copy=False)
    return _move_cells_3d(arr, sy, sx, ty, tx, clear_value).astype(arr.dtype, copy=False)


@typed_njit(cache=True)
def _collision_scan(target_y: np.ndarray, target_x: np.ndarray, occupied: np.ndarray) -> np.ndarray:
    out = np.empty(target_y.size, dtype=np.bool_)
    for i in range(target_y.size):
        out[i] = occupied[target_y[i], target_x[i]]
    return out


def collision_scan_kernel(
    target_y: np.ndarray, target_x: np.ndarray, occupied: np.ndarray
) -> np.ndarray:
    """Return a boolean mask identifying movement targets that are occupied.

    Parameters
    ----------
    target_y, target_x:
        Coordinate vectors for proposed target cells.
    occupied:
        Boolean cell-level array with shape ``(height, width)``.

    Returns
    -------
    np.ndarray
        Boolean vector where ``True`` means the proposed target cell is
        occupied and should be routed to collision handling.
    """
    occ = np.asarray(occupied, dtype=np.bool_)
    if occ.ndim != 2:
        raise ValueError(f"occupied must be a 2-D boolean field, got shape {occ.shape}")
    ty, tx = _validate_coordinate_vectors(target_y, target_x)
    _validate_in_bounds(ty, tx, occ.shape[0], occ.shape[1], "target")
    return _collision_scan(ty, tx, occ)


@typed_njit(cache=True)
def _ingestion_attempts(
    predation: np.ndarray,
    integration: np.ndarray,
    resource: np.ndarray,
    aggression: np.ndarray,
    health: np.ndarray,
    boundary: np.ndarray,
    predator_y: np.ndarray,
    predator_x: np.ndarray,
    target_y: np.ndarray,
    target_x: np.ndarray,
    offset: float,
) -> np.ndarray:
    out = np.empty(predator_y.size, dtype=np.float64)
    for i in range(predator_y.size):
        py = predator_y[i]
        px = predator_x[i]
        ty = target_y[i]
        tx = target_x[i]

        predator_score = (
            1.5 * predation[py, px]
            + 0.8 * integration[py, px]
            + 0.5 * resource[py, px]
            + 0.3 * aggression[py, px]
        )
        target_resistance = (
            0.8 * health[ty, tx] + 0.8 * boundary[ty, tx] + 0.4 * integration[ty, tx]
        )
        z = predator_score - target_resistance - offset

        if z >= 0.0:
            out[i] = 1.0 / (1.0 + np.exp(-z))
        else:
            ez = np.exp(z)
            out[i] = ez / (1.0 + ez)
    return out


def ingestion_attempt_kernel(
    predation: np.ndarray,
    integration: np.ndarray,
    resource: np.ndarray,
    aggression: np.ndarray,
    health: np.ndarray,
    boundary: np.ndarray,
    predator_y: np.ndarray,
    predator_x: np.ndarray,
    target_y: np.ndarray,
    target_x: np.ndarray,
    offset: float = 0.3,
) -> np.ndarray:
    """Compute predatory ingestion success probabilities for coordinate pairs.

    Parameters
    ----------
    predation, integration, resource, aggression, health, boundary:
        Cell-level fields with identical shape ``(height, width)``.
    predator_y, predator_x, target_y, target_x:
        Coordinate vectors. Each index defines one predator-target pair.
    offset:
        Baseline difficulty subtracted from predator advantage.

    Returns
    -------
    np.ndarray
        Float vector with one probability per pair, bounded in ``[0, 1]``.

    Notes
    -----
    The kernel computes probabilities only. Later collision code will compare
    these probabilities against random values and apply state mutations.
    """
    fields = [
        np.asarray(f, dtype=np.float64)
        for f in (predation, integration, resource, aggression, health, boundary)
    ]
    shape = fields[0].shape
    if len(shape) != 2:
        raise ValueError(f"ingestion fields must be 2-D, got shape {shape}")
    if any(f.shape != shape for f in fields):
        raise ValueError("all ingestion fields must share the same shape")

    py, px, ty, tx = _validate_coordinate_vectors(predator_y, predator_x, target_y, target_x)
    _validate_in_bounds(py, px, shape[0], shape[1], "predator")
    _validate_in_bounds(ty, tx, shape[0], shape[1], "target")
    if not np.isfinite(offset):
        raise ValueError("offset must be finite")

    return _ingestion_attempts(
        fields[0],
        fields[1],
        fields[2],
        fields[3],
        fields[4],
        fields[5],
        py,
        px,
        ty,
        tx,
        float(offset),
    ).astype(np.float32)
