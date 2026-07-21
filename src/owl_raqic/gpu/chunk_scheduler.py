from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Chunk:
    start: int
    stop: int
    index: int
    total: int

    @property
    def slice(self) -> slice:
        return slice(self.start, self.stop)


def chunk_slices(n_items: int, chunk_size: int | None) -> Any:
    if n_items < 0:
        raise ValueError("n_items must be nonnegative")
    if chunk_size is None or chunk_size <= 0 or chunk_size >= n_items:
        if n_items:
            yield Chunk(0, n_items, 0, 1)
        return
    total = int(np.ceil(n_items / chunk_size))
    for i, start in enumerate(range(0, n_items, chunk_size)):
        yield Chunk(start, min(n_items, start + chunk_size), i, total)
