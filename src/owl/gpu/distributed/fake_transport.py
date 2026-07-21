from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class FakeCollectiveGroup:
    """Deterministic in-process transport used for distributed contract tests."""

    world_size: int
    mailboxes: dict[tuple[int, int, int], np.ndarray] = field(default_factory=dict)
    sequence: dict[int, int] = field(default_factory=dict)
    reductions: dict[int, dict[int, np.ndarray]] = field(default_factory=dict)

    def endpoint(self, rank: int) -> Any:
        return FakeTransport(self, int(rank))


class FakeTransport:
    def __init__(self, group: FakeCollectiveGroup, rank: int) -> None:
        self.group = group
        self.rank = rank
        self.world_size = group.world_size
        self._sequence = 0
        self.ledger: list[tuple[Any, ...]] = []

    def _next(self) -> Any:
        seq = self._sequence
        self._sequence += 1
        return seq

    def send(
        self,
        array: Any,
        *,
        peer: int,
        stream: Any | None = None,
        tick: int = -1,
        phase: str = "unspecified",
        field_group: Any = "unspecified",
    ) -> Any:
        seq = self._next()
        self.group.mailboxes[(self.rank, int(peer), seq)] = np.array(array, copy=True)
        self.ledger.append(
            ("send", seq, int(peer), np.asarray(array).shape, tick, phase, field_group)
        )

    def recv(
        self,
        array: Any,
        *,
        peer: int,
        stream: Any | None = None,
        tick: int = -1,
        phase: str = "unspecified",
        field_group: Any = "unspecified",
    ) -> Any:
        seq = self._next()
        key = (int(peer), self.rank, seq)
        if key not in self.group.mailboxes:
            raise RuntimeError(f"fake receive before matching send: {key}")
        np.copyto(array, self.group.mailboxes.pop(key))
        self.ledger.append(
            ("recv", seq, int(peer), np.asarray(array).shape, tick, phase, field_group)
        )

    def group_start(self) -> Any:
        return None

    def group_end(self) -> Any:
        return None

    def close(self) -> Any:
        return None
