from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class KernelRooflineEstimate:
    name: str
    cells: int
    bytes_read: int
    bytes_written: int
    flops: int
    elapsed_seconds: float
    peak_bandwidth_bytes_s: float | None = None
    peak_flops_s: float | None = None

    @property
    def total_bytes(self) -> int:
        return self.bytes_read + self.bytes_written

    @property
    def arithmetic_intensity(self) -> float:
        return self.flops / max(self.total_bytes, 1)

    @property
    def achieved_bandwidth_bytes_s(self) -> float:
        return self.total_bytes / max(self.elapsed_seconds, 1e-30)

    @property
    def achieved_flops_s(self) -> float:
        return self.flops / max(self.elapsed_seconds, 1e-30)

    @property
    def bandwidth_efficiency(self) -> float | None:
        if not self.peak_bandwidth_bytes_s:
            return None
        return self.achieved_bandwidth_bytes_s / self.peak_bandwidth_bytes_s

    @property
    def compute_efficiency(self) -> float | None:
        if not self.peak_flops_s:
            return None
        return self.achieved_flops_s / self.peak_flops_s

    def to_dict(self) -> Any:
        out = asdict(self)
        out.update(
            {
                "total_bytes": self.total_bytes,
                "arithmetic_intensity": self.arithmetic_intensity,
                "achieved_bandwidth_bytes_s": self.achieved_bandwidth_bytes_s,
                "achieved_flops_s": self.achieved_flops_s,
                "bandwidth_efficiency": self.bandwidth_efficiency,
                "compute_efficiency": self.compute_efficiency,
            }
        )
        return out


def write_roofline_report(estimates: list[KernelRooflineEstimate], path: str | Path) -> Any:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"kernels": [x.to_dict() for x in estimates]}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
