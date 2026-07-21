from __future__ import annotations

from typing import Any

import numpy as np

from owl_raqic.random_contract import RNGStream, uniform01


def deterministic_uniforms_numpy(
    seed: int, tick: int, ow_id: np.ndarray, stream_id: int = 0
) -> np.ndarray:
    stream = RNGStream.RAQIC_READOUT if int(stream_id) == 0 else int(stream_id)
    return np.asarray(
        uniform01(seed, tick, np.asarray(ow_id, dtype=np.uint64), stream, 0, xp=np),
        dtype=np.float64,
    )


def deterministic_uniforms_backend(
    seed: int, tick: int, ow_id: int, xp: Any, stream_id: int = 0
) -> Any:
    stream = RNGStream.RAQIC_READOUT if int(stream_id) == 0 else int(stream_id)
    return uniform01(
        seed, tick, xp.asarray(ow_id, dtype=xp.uint64), stream, 0, xp=xp, dtype=xp.float64
    )


def deterministic_uniforms_to_backend(
    seed: int, tick: int, ow_id: int, xp: Any, stream_id: int = 0
) -> Any:
    return deterministic_uniforms_backend(seed, tick, ow_id, xp, stream_id)
