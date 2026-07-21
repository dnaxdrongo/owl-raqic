from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any


def json_native(value: Any) -> Any:
    """Convert supported scientific/runtime objects to strict JSON-native values.

    Unsupported objects raise ``TypeError`` rather than silently becoming strings;
    this prevents truthy strings such as ``"False"`` from entering certificates.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return json_native(value.value)
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return json_native(asdict(value))
    if isinstance(value, dict):
        return {str(k): json_native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_native(v) for v in value]

    # NumPy is optional at import time for CPU-light environments.
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return json_native(value.item())
        if isinstance(value, np.ndarray):
            return json_native(value.tolist())
    except Exception:
        pass

    # CuPy arrays/scalars are converted only when explicitly encountered by a
    # report producer outside hot/captured paths.
    module = type(value).__module__.split(".", 1)[0]
    if module == "cupy":
        try:
            import cupy as cp

            return json_native(cp.asnumpy(value))
        except Exception as exc:  # pragma: no cover - target GPU only
            raise TypeError(f"unable to convert CuPy value {type(value)!r}") from exc

    raise TypeError(f"unsupported certificate value type: {type(value)!r}")
