from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class BenchmarkRecord:
    label: str
    ticks: int
    cells: int
    seconds: float
    backend: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ticks_per_sec(self) -> float:
        return self.ticks / self.seconds if self.seconds > 0 else 0.0

    @property
    def cells_per_sec(self) -> float:
        return (self.ticks * self.cells) / self.seconds if self.seconds > 0 else 0.0


def write_benchmark_report(records: list[BenchmarkRecord], path: str | Path) -> None:
    data = []
    for r in records:
        row = {
            "label": r.label,
            "ticks": r.ticks,
            "cells": r.cells,
            "seconds": r.seconds,
            "ticks_per_sec": r.ticks_per_sec,
            "cells_per_sec": r.cells_per_sec,
            "backend": r.backend,
            **r.metadata,
        }
        data.append(row)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
