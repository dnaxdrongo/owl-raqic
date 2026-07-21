from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

BenchmarkRow = Mapping[str, Any]


def write_benchmark_csv(
    path: str | Path,
    rows: Iterable[BenchmarkRow],
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    materialized = [dict(row) for row in rows]
    fieldnames = list(dict.fromkeys(key for row in materialized for key in row))
    with output.open("w", encoding="utf-8", newline="") as handle:
        if fieldnames:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
            writer.writeheader()
            writer.writerows(materialized)
    return output
