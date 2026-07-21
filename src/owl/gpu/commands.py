from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class CommandKind(StrEnum):
    PAUSE = "pause"
    RESUME = "resume"
    CHECKPOINT = "checkpoint"
    REQUEST_VALIDATION = "request_validation"
    VISUAL_SETTING = "visual_setting"
    INJECT_FOOD = "inject_food"
    INJECT_TOXIN = "inject_toxin"


@dataclass(frozen=True)
class GPUCommand:
    kind: CommandKind
    payload: dict[str, Any] = field(default_factory=dict)
    state_mutating: bool = False


class GPUCommandQueue:
    def __init__(self, capacity: int = 1024):
        self.capacity = int(capacity)
        self._items: deque[GPUCommand] = deque()
        self.overflow_count = 0

    def put(self, command: GPUCommand, *, strict: bool = True) -> None:
        if len(self._items) >= self.capacity:
            self.overflow_count += 1
            if strict:
                raise OverflowError(f"GPU command queue capacity exceeded: {self.capacity}")
            return
        self._items.append(command)

    def drain(self) -> list[GPUCommand]:
        out = list(self._items)
        self._items.clear()
        return out

    def __len__(self) -> int:
        return len(self._items)
