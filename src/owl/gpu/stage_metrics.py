from __future__ import annotations

from typing import Any


def metric_value(ds: Any, value: Any, cast: Any) -> Any:
    """Return a device scalar in deferred mode, otherwise a host scalar."""
    if bool(ds.metadata.get("defer_host_metrics", False)):
        return value
    return cast(ds.backend.asnumpy(value))


def metric_int(ds: Any, value: Any) -> Any:
    return metric_value(ds, value, int)


def metric_float(ds: Any, value: Any) -> Any:
    return metric_value(ds, value, float)
