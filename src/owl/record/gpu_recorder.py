from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from owl.record.gpu_metrics import collect_gpu_summary_from_state


class GPUJSONLRecorder:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("w", encoding="utf-8")

    def record(self, state: Any, cfg: Any) -> None:
        self.handle.write(
            json.dumps(collect_gpu_summary_from_state(state, cfg), sort_keys=True) + "\n"
        )

    def close(self) -> None:
        self.handle.close()
