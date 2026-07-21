from __future__ import annotations

from typing import Any

from owl_raqic.gpu.random import (
    deterministic_uniforms_backend,
    deterministic_uniforms_numpy,
)


def counter_uniform(seed: int, tick: int, ids: Any, xp: Any, stream_id: int = 0) -> Any:
    """Rank-invariant OW random stream keyed by global OW id."""
    return deterministic_uniforms_backend(
        seed,
        tick,
        ids,
        xp,
        stream_id=stream_id,
    )


__all__ = [
    "counter_uniform",
    "deterministic_uniforms_backend",
    "deterministic_uniforms_numpy",
]
