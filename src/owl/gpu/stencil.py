from __future__ import annotations

from typing import Any


def _boundary_roll(xp: Any, arr: Any, shift: int, axis: int, mode: str) -> Any:
    if mode == "toroidal":
        return xp.roll(arr, shift, axis=axis)
    # For reflective/absorbing/obstacle first implementation uses edge-padded
    # slicing for non-wrap behavior. Absorbing pads with zero, reflective pads edge.
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (1, 1)
    if mode == "reflective":
        padded = xp.pad(arr, pad_width, mode="edge")
    else:
        padded = xp.pad(arr, pad_width, mode="constant", constant_values=0)
    sl = [slice(None)] * arr.ndim
    if shift == 1:
        sl[axis] = slice(0, -2)
    elif shift == -1:
        sl[axis] = slice(2, None)
    else:
        raise ValueError("only +/-1 shifts are supported")
    return padded[tuple(sl)]


def neighbor_sum_4(field: Any, xp: Any, mode: str = "toroidal") -> Any:
    return (
        _boundary_roll(xp, field, 1, 0, mode)
        + _boundary_roll(xp, field, -1, 0, mode)
        + _boundary_roll(xp, field, 1, 1, mode)
        + _boundary_roll(xp, field, -1, 1, mode)
    )


def neighbor_sum_8(field: Any, xp: Any, mode: str = "toroidal") -> Any:
    n4 = neighbor_sum_4(field, xp, mode)
    up = _boundary_roll(xp, field, 1, 0, mode)
    down = _boundary_roll(xp, field, -1, 0, mode)
    return (
        n4
        + _boundary_roll(xp, up, 1, 1, mode)
        + _boundary_roll(xp, up, -1, 1, mode)
        + _boundary_roll(xp, down, 1, 1, mode)
        + _boundary_roll(xp, down, -1, 1, mode)
    )


def laplacian_4(field: Any, xp: Any, mode: str = "toroidal") -> Any:
    return neighbor_sum_4(field, xp, mode) - 4.0 * field


def local_mean_3x3(field: Any, xp: Any, mode: str = "toroidal") -> Any:
    return (field + neighbor_sum_8(field, xp, mode)) / 9.0


def phase_neighbor_sincos(phase: Any, xp: Any, mode: str = "toroidal") -> Any:
    s = xp.sin(phase)
    c = xp.cos(phase)
    return local_mean_3x3(s, xp, mode), local_mean_3x3(c, xp, mode)


def central_gradient(field: Any, xp: Any, mode: str = "toroidal") -> tuple[Any, Any]:
    """Centered finite-difference gradient with the declared boundary policy."""
    gradient_y = 0.5 * (
        _boundary_roll(xp, field, -1, 0, mode) - _boundary_roll(xp, field, 1, 0, mode)
    )
    gradient_x = 0.5 * (
        _boundary_roll(xp, field, -1, 1, mode) - _boundary_roll(xp, field, 1, 1, mode)
    )
    return gradient_y, gradient_x


def shift_2d(field: Any, xp: Any, dy: int, dx: int, mode: str = "toroidal") -> Any:
    """Shift a spatial field by a signed two-dimensional offset."""
    shifted = field
    if dy:
        if abs(int(dy)) != 1:
            raise ValueError("shift_2d currently supports unit y offsets")
        shifted = _boundary_roll(xp, shifted, int(dy), 0, mode)
    if dx:
        if abs(int(dx)) != 1:
            raise ValueError("shift_2d currently supports unit x offsets")
        shifted = _boundary_roll(xp, shifted, int(dx), 1, mode)
    return shifted


def categorical_neighbor_agreement(
    field: Any,
    xp: Any,
    mode: str = "toroidal",
) -> Any:
    """Count matching values across the eight neighboring coordinates."""
    same = xp.zeros(field.shape[:2], dtype=xp.float32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                same += (shift_2d(field, xp, dy, dx, mode) == field).astype(xp.float32)
    return same
