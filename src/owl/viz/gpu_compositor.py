from __future__ import annotations

from typing import Any, cast

import numpy as np

from owl.core.actions import Action
from owl.viz.sprites import SPRITE_SPECS
from owl.viz.themes import get_theme


def action_color_lut(theme_name: str = "owl_dark_neon") -> np.ndarray:
    del theme_name
    lut = np.zeros((len(Action), 4), dtype=np.uint8)
    for action, spec in SPRITE_SPECS.items():
        lut[int(action)] = np.asarray(spec.color, dtype=np.uint8)
    return lut


def _normalize_cpu(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values)
    lo = float(np.min(values[finite]))
    hi = float(np.max(values[finite]))
    out = np.zeros_like(values)
    if hi > lo:
        out[finite] = (values[finite] - lo) / (hi - lo)
    return cast(np.ndarray, np.clip(out, 0.0, 1.0))


def _readout_and_probabilities(state: Any) -> Any:
    readout = getattr(state, "raqic_readout", None)
    if readout is None:
        readout = state.readout
    probs = getattr(state, "raqic_probabilities", None)
    if probs is None:
        probs = getattr(state, "possibility", None)
    return readout, probs


def compose_frame_cpu(
    state: Any,
    *,
    field: str = "health",
    theme_name: str = "owl_dark_neon",
    show_actions: bool = True,
    show_uncertainty: bool = True,
    alpha: float = 1.0,
) -> np.ndarray:
    """Deterministic, non-mutating scientific scalar underlay."""

    theme = get_theme(theme_name)
    values = np.asarray(getattr(state, field), dtype=np.float32)
    v = _normalize_cpu(values)
    good = np.asarray(theme.health_good, dtype=np.float32)
    bad = np.asarray(theme.health_bad, dtype=np.float32)
    img = bad[None, None, :] * (1 - v[..., None]) + good[None, None, :] * v[..., None]
    readout, probs = _readout_and_probabilities(state)
    live = (np.asarray(state.health) > 0) & (~np.asarray(getattr(state, "obstacle", False)))
    obstacle = np.asarray(getattr(state, "obstacle", np.zeros(live.shape)), dtype=bool)
    if show_actions:
        readout = np.asarray(readout, dtype=np.int32)
        valid = (readout >= 0) & (readout < len(Action))
        lut = action_color_lut(theme_name)
        colors = lut[np.clip(readout, 0, len(lut) - 1)].astype(np.float32)
        img = np.where(live[..., None], 0.55 * img + 0.45 * colors, img)
        img[(~valid) & live] = np.asarray([255, 0, 255, 255], dtype=np.float32)
    if show_uncertainty and probs is not None:
        p = np.asarray(probs, dtype=np.float64)
        entropy = -np.sum(np.where(p > 0, p * np.log(np.maximum(p, 1e-12)), 0.0), axis=-1)
        max_entropy = np.log(max(2, p.shape[-1]))
        uncertainty = np.clip(entropy / max_entropy, 0.0, 1.0)
        haze = np.asarray([120, 160, 255, 255], dtype=np.float32)
        img = np.where(
            live[..., None],
            img * (1.0 - 0.18 * uncertainty[..., None]) + haze * 0.18 * uncertainty[..., None],
            img,
        )
    empty = np.asarray(theme.empty_space, dtype=np.float32)
    obstacle_color = np.asarray(theme.obstacle, dtype=np.float32)
    img = np.where(live[..., None], img, empty)
    img = np.where(obstacle[..., None], obstacle_color, img)
    img[..., 3] = np.where(live | obstacle, np.clip(float(alpha), 0.0, 1.0) * 255.0, 0.0)
    return cast(np.ndarray, np.clip(img, 0, 255).astype(np.uint8))


def compose_frame_device(
    ds: Any,
    *,
    field: str = "health",
    theme_name: str = "owl_dark_neon",
    show_actions: bool = True,
    show_uncertainty: bool = True,
    alpha: float = 1.0,
) -> Any:
    """GPU/NumPy compositor returning a device RGBA underlay."""

    xp = ds.xp
    values = xp.asarray(ds.arrays[field], dtype=xp.float32)
    finite = xp.isfinite(values)
    safe = xp.where(finite, values, 0.0)
    lo = xp.min(xp.where(finite, values, xp.inf))
    hi = xp.max(xp.where(finite, values, -xp.inf))
    denom = hi - lo
    v = xp.where(finite & (denom > 0), (safe - lo) / xp.maximum(denom, 1e-20), 0.0)
    theme = get_theme(theme_name)
    good = xp.asarray(theme.health_good, dtype=xp.float32)
    bad = xp.asarray(theme.health_bad, dtype=xp.float32)
    img = bad[None, None, :] * (1 - v[..., None]) + good[None, None, :] * v[..., None]
    obstacle = ds.arrays.get("obstacle", xp.zeros_like(ds.health, dtype=bool))
    live = (ds.health > 0) & (~obstacle)
    if show_actions and "readout" in ds.arrays:
        lut = xp.asarray(action_color_lut(theme_name), dtype=xp.float32)
        readout = ds.readout.astype(xp.int32)
        valid = (readout >= 0) & (readout < int(lut.shape[0]))
        colors = lut[xp.clip(readout, 0, int(lut.shape[0]) - 1)]
        img = xp.where(live[..., None], 0.55 * img + 0.45 * colors, img)
        magenta = xp.asarray([255, 0, 255, 255], dtype=xp.float32)
        img = xp.where(((~valid) & live)[..., None], magenta, img)
    if show_uncertainty and "raqic_probabilities" in ds.arrays:
        p = ds.raqic_probabilities.astype(xp.float64)
        entropy = -xp.sum(xp.where(p > 0, p * xp.log(xp.maximum(p, 1e-12)), 0.0), axis=-1)
        uncertainty = xp.clip(entropy / xp.log(float(max(2, p.shape[-1]))), 0.0, 1.0)
        haze = xp.asarray([120, 160, 255, 255], dtype=xp.float32)
        img = xp.where(
            live[..., None],
            img * (1.0 - 0.18 * uncertainty[..., None]) + haze * 0.18 * uncertainty[..., None],
            img,
        )
    empty = xp.asarray(theme.empty_space, dtype=xp.float32)
    obstacle_color = xp.asarray(theme.obstacle, dtype=xp.float32)
    img = xp.where(live[..., None], img, empty)
    img = xp.where(obstacle[..., None], obstacle_color, img)
    img[..., 3] = xp.where(
        live | obstacle,
        xp.asarray(np.clip(float(alpha), 0.0, 1.0) * 255.0, dtype=xp.float32),
        xp.asarray(0.0, dtype=xp.float32),
    )
    return xp.clip(img, 0, 255).astype(xp.uint8)


def save_frame_png(path: str, frame: np.ndarray) -> None:
    arr = np.asarray(frame, dtype=np.uint8)
    try:
        from PIL import Image

        Image.fromarray(arr, mode="RGBA" if arr.shape[-1] == 4 else "RGB").save(path)
    except ImportError:
        from matplotlib import pyplot as plt

        plt.imsave(path, arr)
