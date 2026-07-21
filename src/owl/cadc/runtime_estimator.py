"""Produce measured runtime and cost forecasts without authorizing execution."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any


def _positive(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def estimate_phase4_runtime(
    plan: Mapping[str, Any],
    status: Mapping[str, Any],
    inventories: Sequence[Mapping[str, Any]],
    *,
    hourly_cost_usd: float,
    remaining_budget_usd: float,
    post_corpus_reserve_seconds: float,
    corpus_target_seconds: float = 0.0,
    total_target_seconds: float = 0.0,
    gpu_telemetry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Forecast the selected plan from a completed representative pilot.

    Budget comparisons are advisory.  This function never changes a gate or
    converts a cost estimate into a scientific failure.
    """

    price = _positive(hourly_cost_usd, "hourly_cost_usd")
    budget = _positive(remaining_budget_usd, "remaining_budget_usd")
    reserve = float(post_corpus_reserve_seconds)
    if not math.isfinite(reserve) or reserve < 0.0:
        raise ValueError("post_corpus_reserve_seconds must be finite and nonnegative")
    if status.get("plan_id") != plan.get("plan_id"):
        raise ValueError("runtime pilot status belongs to a different corpus plan")
    units = list(plan.get("units", ()))
    if not units:
        raise ValueError("corpus plan has no units")
    sessions = [
        value
        for value in status.get("execution_sessions", ())
        if value.get("passed") is True
        and value.get("completed_unit_ids")
        and float(value.get("wall_seconds", 0.0)) > 0.0
    ]
    if not sessions:
        raise ValueError("no successful measured runtime session is available")
    session = sessions[-1]
    measured_ids = tuple(str(value) for value in session["completed_unit_ids"])
    if len(measured_ids) < 2:
        raise ValueError("runtime calibration requires at least two completed units")
    wall_seconds = _positive(session["wall_seconds"], "pilot wall_seconds")
    throughput = len(measured_ids) / wall_seconds
    unit_status = status.get("units", {})
    runtime_samples = [
        _positive(unit_status[unit_id]["runtime_seconds"], f"runtime {unit_id}")
        for unit_id in measured_ids
    ]
    mean_runtime = statistics.fmean(runtime_samples)
    deviation = statistics.stdev(runtime_samples) if len(runtime_samples) > 1 else 0.0
    coefficient = deviation / mean_runtime if mean_runtime else 0.0
    # A 15% floor covers launch/I/O variation that a small pilot cannot see;
    # the measured coefficient widens the interval for heterogeneous contexts.
    relative_margin = min(
        0.75,
        max(0.15, 0.10 + 1.96 * coefficient / math.sqrt(len(runtime_samples))),
    )
    successful_ids = {
        str(unit_id)
        for unit_id, value in unit_status.items()
        if value.get("exit_code") == 0
    }
    total_units = len(units)
    completed_units = len(successful_ids & {str(value["unit_id"]) for value in units})
    remaining_units = max(0, total_units - completed_units)
    total_point = total_units / throughput
    remaining_point = remaining_units / throughput

    def interval(point: float) -> dict[str, float]:
        return {
            "low": max(0.0, point * (1.0 - relative_margin)),
            "point": point,
            "high": point * (1.0 + relative_margin),
        }

    corpus_total = interval(total_point)
    corpus_remaining = interval(remaining_point)
    total_remaining = {
        name: seconds + reserve for name, seconds in corpus_remaining.items()
    }
    costs = {
        name: seconds * price / 3600.0 for name, seconds in total_remaining.items()
    }
    if costs["high"] <= budget:
        budget_signal = "LIKELY_WITHIN_REMAINING_BUDGET"
    elif costs["point"] <= budget:
        budget_signal = "POINT_ESTIMATE_WITHIN_BUT_UPPER_RANGE_EXCEEDS_BUDGET"
    else:
        budget_signal = "LIKELY_EXCEEDS_REMAINING_BUDGET"

    memory_estimates = []
    for inventory in inventories:
        memory = inventory.get("counterfactual_memory_plan") or {}
        workers = int(inventory.get("branch_worker_count", 1))
        per_branch = int(memory.get("per_branch_bytes", 0))
        fixed = sum(
            int(memory.get(name, 0))
            for name in (
                "factual_fixed_bytes",
                "source_snapshot_bytes",
                "hash_workspace_bytes",
                "pinned_packet_bytes",
                "library_reserve_bytes",
                "transient_reserve_bytes",
            )
        )
        if fixed or per_branch:
            memory_estimates.append(fixed + workers * per_branch)
    concurrent_units = int(session.get("max_concurrent_units", 1))
    aggregate_device_budget = int(
        session.get(
            "aggregate_device_budget_bytes",
            status.get("aggregate_device_budget_bytes", 0),
        )
        or 0
    )
    projected_device_bytes = (
        max(memory_estimates) * concurrent_units if memory_estimates else None
    )
    utilization = dict(gpu_telemetry or {})
    mean_gpu_utilization = float(utilization.get("mean_gpu_utilization_percent", 0.0))
    peak_memory_fraction = float(utilization.get("peak_memory_fraction", 0.0))
    if not utilization:
        tuning_signal = "GPU_TELEMETRY_UNAVAILABLE"
    elif mean_gpu_utilization < 60.0 and peak_memory_fraction < 0.70:
        tuning_signal = "GPU_HEADROOM_AVAILABLE_CONSIDER_HIGHER_CONCURRENCY"
    elif mean_gpu_utilization >= 90.0 or peak_memory_fraction >= 0.85:
        tuning_signal = "GPU_NEAR_COMPUTE_OR_MEMORY_SATURATION"
    else:
        tuning_signal = "GPU_CONCURRENCY_BALANCED"
    return {
        "schema_version": "owl.cadc.phase4-runtime-estimate.v1",
        "classification": "USER_RUNTIME_DECISION_REQUIRED",
        "passed": True,
        "automatic_failure": False,
        "decision_required": True,
        "plan_id": str(plan["plan_id"]),
        "calibration": {
            "measured_unit_ids": list(measured_ids),
            "measured_units": len(measured_ids),
            "wall_seconds": wall_seconds,
            "units_per_second": throughput,
            "mean_overlapped_unit_seconds": mean_runtime,
            "unit_runtime_standard_deviation": deviation,
            "coefficient_of_variation": coefficient,
            "relative_uncertainty_margin": relative_margin,
            "max_concurrent_units": concurrent_units,
            "branch_transfer_mode": session.get("branch_transfer_mode"),
        },
        "selected_profile": {
            "total_units": total_units,
            "completed_units": completed_units,
            "remaining_units": remaining_units,
        },
        "forecast_seconds": {
            "corpus_total": corpus_total,
            "corpus_remaining": corpus_remaining,
            "post_corpus_reserve": reserve,
            "total_remaining": total_remaining,
        },
        "forecast_cost_usd": costs,
        "hourly_cost_usd": price,
        "remaining_budget_usd": budget,
        "budget_signal": budget_signal,
        "targets": {
            "corpus_seconds": float(corpus_target_seconds),
            "total_seconds": float(total_target_seconds),
            "corpus_point_within_target": (
                not corpus_target_seconds or corpus_total["point"] <= corpus_target_seconds
            ),
            "total_point_within_target": (
                not total_target_seconds or total_remaining["point"] <= total_target_seconds
            ),
        },
        "projected_concurrent_device_bytes": projected_device_bytes,
        "aggregate_device_budget_bytes": aggregate_device_budget or None,
        "projected_device_memory_within_budget": (
            projected_device_bytes <= aggregate_device_budget
            if projected_device_bytes is not None and aggregate_device_budget
            else None
        ),
        "gpu_calibration_telemetry": utilization or None,
        "gpu_tuning_signal": tuning_signal,
        "decision_options": [
            "proceed_with_selected_profile",
            "choose_smaller_profile_and_recalibrate",
            "stop_without_running_remaining_corpus",
        ],
        "limitations": [
            "Corpus forecast is measured from the latest successful representative session.",
            "Post-corpus time is an explicit reserve until training/ETL stage measurements exist.",
            "RunPod price and remaining budget are caller-supplied and are not inferred.",
            "A budget signal is advisory and never overrides scientific or integrity gates.",
        ],
    }
