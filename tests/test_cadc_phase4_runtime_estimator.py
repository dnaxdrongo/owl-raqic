from __future__ import annotations

import json

import pytest
from scripts.estimate_cadc_phase4_runtime import _calibration_gpu_summary

from owl.cadc.runtime_estimator import estimate_phase4_runtime


def _evidence() -> tuple[dict, dict]:
    plan = {
        "plan_id": "plan-1",
        "units": [{"unit_id": f"u{index}"} for index in range(12)],
    }
    status = {
        "plan_id": "plan-1",
        "units": {
            "u0": {"exit_code": 0, "runtime_seconds": 55.0},
            "u1": {"exit_code": 0, "runtime_seconds": 65.0},
        },
        "execution_sessions": [
            {
                "passed": True,
                "completed_unit_ids": ["u0", "u1"],
                "wall_seconds": 70.0,
                "max_concurrent_units": 2,
                "branch_transfer_mode": "deferred_bounded",
            }
        ],
    }
    return plan, status


def test_runtime_estimate_is_advisory_even_when_budget_is_exceeded() -> None:
    plan, status = _evidence()
    result = estimate_phase4_runtime(
        plan,
        status,
        [],
        hourly_cost_usd=4.0,
        remaining_budget_usd=0.01,
        post_corpus_reserve_seconds=600.0,
        gpu_telemetry={
            "mean_gpu_utilization_percent": 45.0,
            "peak_memory_fraction": 0.5,
        },
    )
    assert result["passed"] is True
    assert result["automatic_failure"] is False
    assert result["decision_required"] is True
    assert result["budget_signal"] == "LIKELY_EXCEEDS_REMAINING_BUDGET"
    assert result["selected_profile"]["remaining_units"] == 10
    assert result["gpu_tuning_signal"] == (
        "GPU_HEADROOM_AVAILABLE_CONSIDER_HIGHER_CONCURRENCY"
    )


def test_runtime_estimate_rejects_cross_plan_evidence() -> None:
    plan, status = _evidence()
    status["plan_id"] = "different"
    with pytest.raises(ValueError, match="different corpus plan"):
        estimate_phase4_runtime(
            plan,
            status,
            [],
            hourly_cost_usd=1.0,
            remaining_budget_usd=10.0,
            post_corpus_reserve_seconds=0.0,
        )


def test_gpu_summary_uses_only_calibration_tail(tmp_path) -> None:
    telemetry = tmp_path / "gpu.csv"
    telemetry.write_text(
        "\n".join(
            (
                "2026/07/19 00:00:00, NVIDIA H100, 0000:00:00.0, 1, 0, 100, 80000, 60, P0",
                "2026/07/19 00:00:02, NVIDIA H100, 0000:00:00.0, 70, 10, 20000, 80000, 300, P0",
                "2026/07/19 00:00:04, NVIDIA H100, 0000:00:00.0, 90, 20, 40000, 80000, 500, P0",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    timings = tmp_path / "timings.json"
    timings.write_text(
        json.dumps(
            {
                "gates": {
                    "run_corpus_calibration": {"elapsed_seconds": 4.0}
                }
            }
        ),
        encoding="utf-8",
    )
    summary = _calibration_gpu_summary(telemetry, timings)
    assert summary is not None
    assert summary["sample_count"] == 2
    assert summary["mean_gpu_utilization_percent"] == 80.0
    assert summary["peak_memory_fraction"] == 0.5
