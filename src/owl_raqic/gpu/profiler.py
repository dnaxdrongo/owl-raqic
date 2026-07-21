from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GPUProfile:
    stages: dict[str, float] = field(default_factory=dict)
    counters: dict[str, float | int | str | bool] = field(default_factory=dict)

    @contextmanager
    def stage(self, name: str) -> Any:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = self.stages.get(name, 0.0) + (time.perf_counter() - t0)

    def to_dict(self) -> dict[str, Any]:
        return {"stages": dict(self.stages), "counters": dict(self.counters)}
