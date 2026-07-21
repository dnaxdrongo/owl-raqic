from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class NumericalLedger:
    run_class: str
    precision: str
    tolerances: dict[str, float]
    max_probability_row_error: float = 0.0
    min_probability: float = 1.0
    mask_violation_count: int = 0
    nan_inf_count: int = 0
    clipping_count: int = 0
    all_illegal_repair_count: int = 0
    topology_overflow_count: int = 0
    visual_overflow_count: int = 0
    graph_invalidation_count: int = 0
    fallback_count: int = 0
    threshold_ties: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @classmethod
    def from_config(cls, cfg: Any) -> NumericalLedger:
        precision = str(getattr(cfg.raqic, "full_gpu_precision", "audit64"))
        if precision == "audit64":
            probs, norm = 1e-10, 1e-10
        elif precision == "mixed":
            probs, norm = 1e-6, 1e-6
        else:
            probs, norm = 5e-6, 1e-6
        return cls(
            run_class=str(getattr(cfg.raqic, "full_gpu_run_class", "validation")),
            precision=precision,
            tolerances={
                "probability_max_abs": float(
                    getattr(cfg.raqic, "gpu_probability_tolerance", probs)
                ),
                "probability_row_sum": norm,
                "trace_residual": 1e-10 if precision == "audit64" else 1e-6,
                "min_eigenvalue": -1e-10 if precision == "audit64" else -1e-6,
            },
        )

    def update_metrics(self, metrics: dict[str, Any]) -> None:
        self.max_probability_row_error = max(
            self.max_probability_row_error,
            float(metrics.get("raqic_max_row_error", 0.0)),
        )
        self.topology_overflow_count = max(
            self.topology_overflow_count,
            int(metrics.get("topology_overflow", 0)),
        )
        self.visual_overflow_count = max(
            self.visual_overflow_count,
            int(metrics.get("visual_event_overflow", 0)),
        )
        self.fallback_count = max(self.fallback_count, int(metrics.get("fallback_count", 0)))

    def record_threshold_tie(self, name: str, count: int = 1) -> None:
        self.threshold_ties[name] = self.threshold_ties.get(name, 0) + int(count)

    def validate(self) -> dict[str, Any]:
        limit = self.tolerances["probability_row_sum"]
        failures = []
        if (
            not math.isfinite(self.max_probability_row_error)
            or self.max_probability_row_error > limit
        ):
            failures.append(
                f"probability row error {self.max_probability_row_error:.3e} exceeds {limit:.3e}"
            )
        if self.mask_violation_count:
            failures.append(f"mask violations: {self.mask_violation_count}")
        if self.nan_inf_count:
            failures.append(f"NaN/Inf values: {self.nan_inf_count}")
        if self.topology_overflow_count:
            failures.append(f"topology overflow: {self.topology_overflow_count}")
        return {"passed": not failures, "failures": failures}

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["validation"] = self.validate()
        return out

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return path
