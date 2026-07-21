from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StageRecord:
    name: str
    wall_seconds: float
    gpu_milliseconds: float | None = None
    _start_event: Any = None
    _end_event: Any = None


@dataclass
class GPUFullProfiler:
    backend: Any = None
    stream: Any = None
    records: list[StageRecord] = field(default_factory=list)
    enabled: bool = True

    @contextmanager
    def stage(self, name: str) -> Any:
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        start = end = None
        if self.backend is not None and getattr(self.backend, "is_gpu", False):
            cp = self.backend.xp
            start, end = cp.cuda.Event(), cp.cuda.Event()
            start.record(self.stream)
        try:
            yield
        finally:
            if end is not None:
                end.record(self.stream)
            self.records.append(StageRecord(name, time.perf_counter() - t0, None, start, end))

    def resolve_gpu_times(self) -> None:
        if self.backend is None or not getattr(self.backend, "is_gpu", False):
            return
        cp = self.backend.xp
        for rec in self.records:
            if rec.gpu_milliseconds is None and rec._end_event is not None:
                rec._end_event.synchronize()
                rec.gpu_milliseconds = float(
                    cp.cuda.get_elapsed_time(rec._start_event, rec._end_event)
                )

    def reset(self) -> None:
        self.records.clear()

    def to_dict(self, *, resolve_gpu: bool = False) -> dict[str, Any]:
        if resolve_gpu:
            self.resolve_gpu_times()
        return {
            "stages": [
                {
                    "name": r.name,
                    "wall_seconds": r.wall_seconds,
                    "gpu_milliseconds": r.gpu_milliseconds,
                }
                for r in self.records
            ],
            "total_wall_seconds": sum(r.wall_seconds for r in self.records),
            "total_gpu_milliseconds": sum(r.gpu_milliseconds or 0.0 for r in self.records)
            if any(r.gpu_milliseconds is not None for r in self.records)
            else None,
        }
