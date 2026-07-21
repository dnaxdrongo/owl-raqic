from __future__ import annotations

from typing import Any

import numpy as np


def gpu_prepare_rgb_frame(state: Any, field: str = "health") -> np.ndarray:
    """Prepare an RGB frame from a state field.

    In a GPU run the field may already have been colorized on device; this
    portable entry point returns a CPU uint8 frame for Pygame/VisPy upload.
    """
    values = np.asarray(getattr(state, field), dtype=np.float32)
    v = values - np.nanmin(values)
    denom = np.nanmax(v)
    if denom > 0:
        v = v / denom
    img = np.zeros(values.shape + (3,), dtype=np.uint8)
    img[..., 0] = np.clip(255 * v, 0, 255).astype(np.uint8)
    img[..., 1] = np.clip(255 * np.asarray(state.resource), 0, 255).astype(np.uint8)
    if getattr(state, "raqic_readout", None) is not None:
        img[..., 2] = (np.asarray(state.raqic_readout) * 13 % 255).astype(np.uint8)
    return img


def main() -> None:
    print(
        "gpu_viewer module installed. Use gpu_prepare_rgb_frame(state) from a running simulation."
    )
