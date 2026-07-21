from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from .json_types import json_native


@dataclass(frozen=True)
class RunPaths:
    root: Path
    run_id: str

    @property
    def base(self) -> Path:
        return self.root / self.run_id

    @property
    def reports(self) -> Path:
        return self.base / "reports"

    @property
    def checkpoints(self) -> Path:
        return self.base / "checkpoints"

    @property
    def frames(self) -> Path:
        return self.base / "frames"

    @property
    def traces(self) -> Path:
        return self.base / "traces"

    def create(self) -> RunPaths:
        for path in (self.reports, self.checkpoints, self.frames, self.traces):
            path.mkdir(parents=True, exist_ok=True)
        return self


def derive_run_paths(
    *,
    cfg: Any,
    plan: Any,
    root: str | Path = "runs",
    environment: dict[str, Any] | None = None,
) -> RunPaths:
    config = cfg.model_dump(mode="json") if hasattr(cfg, "model_dump") else cfg
    plan_data = plan.to_dict() if hasattr(plan, "to_dict") else plan
    payload = {
        "config": json_native(config),
        "plan": json_native(plan_data),
        "environment": json_native(environment or {}),
        "scientific_contract": getattr(plan, "scientific_contract_version", None),
    }
    material = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    run_id = sha256(material.encode("utf-8")).hexdigest()[:24]
    return RunPaths(Path(root), run_id).create()
