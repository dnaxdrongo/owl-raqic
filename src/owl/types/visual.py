"""Visualization contracts."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt

RGBAFrame = npt.NDArray[np.uint8]


class Renderer(Protocol):
    def render(self, frame: RGBAFrame) -> None: ...
    def close(self) -> None: ...
