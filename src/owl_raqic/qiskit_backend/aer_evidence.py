from __future__ import annotations

from typing import Any


def parse_aer_gpu_evidence(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return positive, version-tolerant evidence that Aer used a GPU.

    A requested ``device='GPU'`` option is deliberately not accepted as
    execution evidence.  Only result metadata emitted by Aer is inspected.
    The complete metadata is retained for later version-specific review.
    """
    indicators: list[dict[str, Any]] = []
    stack: list[tuple[str, Any]] = [("", dict(metadata or {}))]
    count_keys = {
        "gpu_parallel_shots_",
        "gpu_parallel_shots",
        "batched_shots_optimization_parallel_gpus",
        "chunk_parallel_gpus",
        "num_gpus",
        "gpu_count",
    }
    device_keys = {"device", "execution_device", "backend_device", "target"}
    while stack:
        prefix, value = stack.pop()
        if isinstance(value, dict):
            for key, child in value.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                normalized = str(key).lower()
                if normalized in device_keys and any(
                    token in str(child).upper()
                    for token in ("GPU", "CUDA", "CUSTATEVEC", "CUQUANTUM")
                ):
                    indicators.append({"key": path, "value": child})
                if normalized in count_keys:
                    try:
                        if int(child) > 0:
                            indicators.append({"key": path, "value": child})
                    except (TypeError, ValueError):
                        pass
                stack.append((path, child))
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                stack.append((f"{prefix}[{index}]", child))
    return {
        "verified": bool(indicators),
        "positive_indicators": indicators,
        "raw_metadata": dict(metadata or {}),
    }
