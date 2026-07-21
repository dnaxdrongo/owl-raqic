from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def write_audit_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    def default(value: object) -> object:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, complex):
            return {"real": value.real, "imag": value.imag}
        try:
            return str(value)
        except Exception:
            return None

    output.write_text(
        json.dumps(payload, indent=2, default=default),
        encoding="utf-8",
    )
    return output
