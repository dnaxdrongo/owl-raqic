from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def append_journal(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    row = {"timestamp": datetime.now(UTC).isoformat(), **payload}
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


class ProgressJournal:
    def __init__(self, run_root: str | Path) -> None:
        self.root = Path(run_root)
        self.status_path = self.root / "run_progress.json"
        self.journal_path = self.root / "run_progress.jsonl"

    def update(self, *, state: str, phase: str, **values: Any) -> None:
        payload = {"state": state, "phase": phase, **values}
        atomic_write_json(self.status_path, payload)
        append_journal(self.journal_path, payload)
