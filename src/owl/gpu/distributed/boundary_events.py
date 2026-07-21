from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BoundaryCandidate:
    source_global_id: int
    target_global_index: int
    priority_rank: int
    source_rank: int
    event_type: int
    payload: tuple[float, ...] = ()

    @property
    def ordering_key(self) -> Any:
        return (
            int(self.target_global_index),
            -int(self.priority_rank),
            int(self.source_global_id),
            int(self.source_rank),
            int(self.event_type),
        )


def resolve_boundary_candidates(candidates: Any) -> Any:
    """Choose one deterministic candidate per (event type, target)."""
    ordered = sorted(candidates, key=lambda item: item.ordering_key)
    winners = []
    seen = set()
    for candidate in ordered:
        key = (int(candidate.event_type), int(candidate.target_global_index))
        if key in seen:
            continue
        seen.add(key)
        winners.append(candidate)
    return tuple(winners)


def pack_candidates(candidates: Any) -> np.ndarray:
    out = np.zeros((len(candidates), 6), dtype=np.int64)
    for row, item in enumerate(candidates):
        out[row] = (
            item.source_global_id,
            item.target_global_index,
            item.priority_rank,
            item.source_rank,
            item.event_type,
            len(item.payload),
        )
    return out
