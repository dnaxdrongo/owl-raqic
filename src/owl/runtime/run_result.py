from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .execution_plan import ExecutionPlan


@dataclass
class RunResult:
    state: Any
    metrics: list[dict[str, Any]]
    execution_plan: ExecutionPlan
    execution_metadata: dict[str, Any] = field(default_factory=dict)
    artifacts: tuple[Path, ...] = ()
    success: bool = True

    def summary(self) -> Mapping[str, Any]:
        return {
            "success": self.success,
            "final_tick": int(getattr(self.state, "tick", 0)),
            "metric_rows": len(self.metrics),
            "execution_plan": self.execution_plan.to_dict(),
            "execution_metadata": self.execution_metadata,
            "artifacts": [str(path) for path in self.artifacts],
        }
