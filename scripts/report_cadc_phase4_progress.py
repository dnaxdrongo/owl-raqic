#!/usr/bin/env python3
"""Report measured CADC-MORE 2 progress without mutating the run."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import fmean
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _decision_time(decision: dict[str, Any], fallback: float) -> datetime:
    raw = decision.get("recorded_at")
    if isinstance(raw, str):
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(UTC)
        except ValueError:
            pass
    return datetime.fromtimestamp(fallback, tz=UTC)


def _gpu_status() -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,utilization.memory,memory.used,"
                "memory.total,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=3,
        )
        fields = [value.strip() for value in completed.stdout.splitlines()[0].split(",")]
        return {
            "available": True,
            "name": fields[0],
            "compute_utilization_percent": float(fields[1]),
            "memory_utilization_percent": float(fields[2]),
            "memory_used_mib": float(fields[3]),
            "memory_total_mib": float(fields[4]),
            "power_draw_watts": float(fields[5]),
        }
    except (FileNotFoundError, IndexError, subprocess.SubprocessError, ValueError):
        return {"available": False}


_STAGE_NAMES = {
    "certifier": "certification",
    "mypy_phase4": "final_quality",
    "ruff_phase4": "final_quality",
    "pytest_full": "final_quality",
    "hotpath_audit": "profiling",
    "profile": "profiling",
    "casebook": "analysis",
    "negative_controls": "analysis",
    "evaluate": "analysis",
    "score_artifacts": "scoring",
    "calibrate": "calibration",
    "train": "training",
    "repeat_pilot": "repeat_analysis",
    "build_dataset": "etl",
    "certify_corpus": "corpus_certification",
    "run_corpus": "corpus",
    "run_corpus_calibration": "runtime_pilot",
}


def _stage(
    status: dict[str, Any],
    corpus: dict[str, Any],
    training: dict[str, Any],
    active: dict[str, Any],
) -> str:
    if active.get("state") == "RUNNING":
        gate = str(active.get("gate", "unknown"))
        return _STAGE_NAMES.get(gate, gate)
    if training and training.get("passed") is not True:
        return "training"
    if corpus and corpus.get("passed") is not True:
        return "corpus"
    for gate, name in _STAGE_NAMES.items():
        if gate in status:
            return name if int(status[gate]) == 0 else f"failed:{gate}"
    return "preflight_or_runtime_pilot"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    root = Path(args.output_root).resolve()
    status = _load(root / "command_status.json")
    estimate = _load(root / "runtime_estimate.json")
    decision = _load(root / "runtime_decision.json")
    corpus_status = _load(root / "corpus" / "corpus_run_status.json")
    plan = _load(root / "corpus" / "corpus_plan.json")
    training = _load(root / "models" / "training_progress.json")
    certificate = _load(root / "phase4_certificate.json")
    active_gate = _load(root / "active_gate.json")
    now = datetime.now(UTC)

    failed_gates = {
        name: int(code)
        for name, code in status.items()
        if isinstance(code, (int, float, str)) and int(code) != 0
    }
    active_state = str(active_gate.get("state", ""))
    if active_state == "RUNNING":
        # A resume may legitimately coexist with a stale failed-closed
        # certificate from the interrupted attempt. The live gate is the
        # current execution authority until it completes.
        state = "RUNNING"
    elif active_state == "FAILED":
        state = "FAILED"
    elif certificate.get("passed") is True:
        state = "SUCCEEDED"
    elif failed_gates or certificate.get("classification") == "FAILED_CLOSED":
        state = "FAILED"
    elif not decision and estimate:
        state = "PAUSED_AWAITING_USER_RUNTIME_DECISION"
    else:
        state = "RUNNING"

    total_units = len(plan.get("units", ()))
    unit_records = corpus_status.get("units", {})
    completed_units = sum(
        int(isinstance(value, dict) and value.get("exit_code") == 0)
        for value in unit_records.values()
    )
    active_units = list(corpus_status.get("active_units", ()))
    worker_count = int(corpus_status.get("max_concurrent_units", 0) or 0)
    stage = _stage(status, corpus_status, training, active_gate)

    point = 0.0
    high = 0.0
    basis = "unavailable"
    confidence = "unavailable"
    forecast = estimate.get("forecast_seconds", {})
    reserve = float(forecast.get("post_corpus_reserve", 0.0) or 0.0)
    estimate_path = root / "runtime_estimate.json"
    fallback = estimate_path.stat().st_mtime if estimate_path.is_file() else now.timestamp()
    recorded_at = _decision_time(decision, fallback)

    successful_runtimes = [
        float(value.get("runtime_seconds", 0.0))
        for value in unit_records.values()
        if isinstance(value, dict)
        and value.get("exit_code") == 0
        and float(value.get("runtime_seconds", 0.0)) > 0.0
    ]
    if stage == "corpus" and total_units and successful_runtimes:
        mean_unit = fmean(successful_runtimes)
        effective_workers = max(1, worker_count)
        active_elapsed = corpus_status.get("active_unit_elapsed_seconds", {})
        active_remaining_samples = [
            max(0.0, mean_unit - float(active_elapsed.get(unit_id, 0.0)))
            for unit_id in active_units
        ]
        active_remaining = sum(active_remaining_samples)
        pending_units = int(
            corpus_status.get(
                "pending_units",
                max(0, total_units - completed_units - len(active_units)),
            )
        )
        # Active jobs occupy lanes now; queued jobs consume complete lane-time.
        # Dividing the combined lane-seconds by the measured worker count gives
        # a substantially less jittery ETA than treating active units as new.
        lane_seconds = active_remaining + max(0, pending_units) * mean_unit
        corpus_remaining = max(
            max(active_remaining_samples, default=0.0),
            lane_seconds / effective_workers,
        )
        variation = (
            max(successful_runtimes) / mean_unit - 1.0
            if len(successful_runtimes) > 1 and mean_unit > 0.0
            else 0.25
        )
        margin = min(0.75, max(0.15, variation))
        point = corpus_remaining + reserve
        high = corpus_remaining * (1.0 + margin) + reserve
        basis = "live_completed_unit_throughput_plus_post_corpus_reserve"
        confidence = (
            "measured_medium"
            if completed_units >= max(6, worker_count)
            else "measured_low"
        )
    elif training and training.get("passed") is not True:
        completed_trials = int(training.get("completed_member_trials", 0))
        total_trials = int(training.get("total_member_trials", 0))
        epoch = int(training.get("epoch_completed", 0))
        epochs = max(1, int(training.get("epochs_configured", 1)))
        mean_epoch = float(training.get("mean_epoch_seconds_current_member", 0.0) or 0.0)
        remaining_equivalent = max(0.0, total_trials - completed_trials - epoch / epochs)
        training_remaining = remaining_equivalent * epochs * mean_epoch
        downstream_reserve = reserve * 0.35
        point = training_remaining + downstream_reserve
        high = training_remaining * 1.30 + downstream_reserve
        basis = "live_training_epoch_throughput_plus_downstream_reserve"
        confidence = "measured_medium" if epoch >= 3 else "measured_low"
    elif estimate:
        total = forecast.get("total_remaining", {})
        elapsed = max(0.0, (now - recorded_at).total_seconds())
        point = max(0.0, float(total.get("point", 0.0)) - elapsed)
        high = max(point, float(total.get("high", point)) - elapsed)
        basis = "accepted_runtime_forecast_countdown"
        confidence = "estimated_low"

    if state in {"SUCCEEDED", "FAILED"}:
        point = high = 0.0
    point = point if math.isfinite(point) else 0.0
    high = high if math.isfinite(high) else point
    payload = {
        "schema_version": "owl.cadc.phase4-live-progress.v1",
        "state": state,
        "stage": stage,
        "failed_gates": failed_gates,
        "corpus": {
            "completed_units": completed_units,
            "total_units": total_units,
            "active_units": active_units,
            "workers": worker_count,
        },
        "training": training,
        "countdown": {
            "remaining_seconds_point": point,
            "remaining_seconds_high": high,
            "eta_point_utc": _iso(now + timedelta(seconds=point)),
            "eta_high_utc": _iso(now + timedelta(seconds=high)),
            "basis": basis,
            "confidence": confidence,
            "advisory_only": True,
        },
        "certificate": {
            "passed": certificate.get("passed"),
            "classification": certificate.get("classification"),
            "phase5_unlocked": certificate.get("phase5_unlocked", False),
        },
        "gpu": _gpu_status(),
    }
    rendered = (
        json.dumps(payload, sort_keys=True)
        if args.json
        else json.dumps(payload, indent=2, sort_keys=True)
    )
    print(rendered)
    return 1 if state == "FAILED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
