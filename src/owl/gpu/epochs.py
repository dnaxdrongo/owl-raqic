from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FieldEpochs:
    """Small mutation-epoch tracker for scratch-cache validity."""

    epochs: dict[str, int] = field(default_factory=dict)

    def tick(self, *names: str) -> None:
        for name in names:
            self.epochs[name] = int(self.epochs.get(name, 0)) + 1

    def get(self, name: str) -> int:
        return int(self.epochs.get(name, 0))

    def snapshot(self, names: tuple[str, ...] | list[str]) -> dict[str, int]:
        return {name: self.get(name) for name in names}

    def matches(self, snapshot: dict[str, int]) -> bool:
        return all(self.get(name) == value for name, value in snapshot.items())
