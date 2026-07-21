from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from owl.core.actions import Action


def _scalar(value: Any, ds: Any) -> float:
    """Compatibility scalar extraction for stage-once and focused tests only."""
    array = ds.backend.asnumpy(value)
    try:
        return float(array)
    except (TypeError, ValueError):
        return float(array.item())


def invariant_summary(ds: Any, cfg: Any) -> dict[str, int | float]:
    """Compatibility invariant summary.

    Persistent production uses :class:`DeviceMetricSlab` and transfers one
    compact record at metric cadence.  This function remains for stage-once
    compatibility and focused invariant tests, where a host boundary is
    explicit and graph capture is not active.
    """

    xp = ds.xp
    live = (ds.health > 0) & (~ds.obstacle)
    summary: dict[str, int | float] = {
        "live_cells": int(_scalar(xp.sum(live), ds)),
        "nan_count_health": int(_scalar(xp.sum(~xp.isfinite(ds.health)), ds)),
        "nan_count_resource": int(_scalar(xp.sum(~xp.isfinite(ds.resource)), ds)),
    }
    if "raqic_probabilities" in ds.arrays:
        probabilities = ds.raqic_probabilities
        sums = xp.sum(probabilities, axis=-1)
        row_error = xp.where(live, xp.abs(sums - 1.0), 0.0)
        summary["probability_max_row_error"] = float(_scalar(xp.max(row_error), ds))
        summary["dead_nonrest_count"] = int(
            _scalar(
                xp.sum((~live) & (ds.raqic_readout != int(Action.REST)))
                if "raqic_readout" in ds.arrays
                else xp.asarray(0),
                ds,
            )
        )
    return summary


def invariant_summary_from_metric(metric: Mapping[str, Any]) -> dict[str, int | float]:
    """Extract invariant fields from one decoded device metric slab."""

    return {
        "live_cells": int(metric.get("alive_count", 0)),
        "nan_count_health": int(metric.get("nan_count_health", 0)),
        "nan_count_resource": int(metric.get("nan_count_resource", 0)),
        "probability_max_row_error": float(metric.get("raqic_max_row_error", 0.0)),
        "dead_nonrest_count": int(metric.get("dead_nonrest_count", 0)),
    }


def assert_invariant_summary(summary: Mapping[str, int | float], cfg: Any) -> None:
    if summary["nan_count_health"] or summary["nan_count_resource"]:
        raise AssertionError(f"non-finite life field in GPU full run: {dict(summary)}")
    if (
        "probability_max_row_error" in summary
        and float(summary["probability_max_row_error"])
        > float(getattr(cfg.raqic, "gpu_probability_tolerance", 1e-8)) * 10
    ):
        raise AssertionError(f"probability normalization failed in GPU full run: {dict(summary)}")
    if int(summary.get("dead_nonrest_count", 0)):
        raise AssertionError(f"dead cells acted in GPU full run: {dict(summary)}")


def assert_gpu_full_invariants(ds: Any, cfg: Any) -> None:
    assert_invariant_summary(invariant_summary(ds, cfg), cfg)
