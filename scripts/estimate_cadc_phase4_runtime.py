#!/usr/bin/env python3
"""Emit a measured advisory runtime and cost estimate for CADC-MORE 2 analysis."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from owl.cadc.artifacts import atomic_json  # noqa: E402
from owl.cadc.runtime_estimator import estimate_phase4_runtime  # noqa: E402


def _calibration_gpu_summary(
    telemetry_path: Path,
    timings_path: Path,
) -> dict[str, object] | None:
    """Summarize only the tail corresponding to the measured calibration gate."""

    if not telemetry_path.is_file() or not timings_path.is_file():
        return None
    timings = json.loads(timings_path.read_text(encoding="utf-8"))
    gate = timings.get("gates", {}).get("run_corpus_calibration", {})
    elapsed = float(gate.get("elapsed_seconds", 0.0))
    if not math.isfinite(elapsed) or elapsed <= 0.0:
        return None
    rows = []
    with telemetry_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) < 9:
                continue
            try:
                rows.append(
                    {
                        "gpu": float(row[3]),
                        "memory_utilization": float(row[4]),
                        "memory_used": float(row[5]),
                        "memory_total": float(row[6]),
                        "power": float(row[7]),
                    }
                )
            except ValueError:
                continue
    sample_count = max(1, int(math.ceil(elapsed / 2.0)))
    selected = rows[-sample_count:]
    if not selected:
        return None

    def mean(name: str) -> float:
        return sum(value[name] for value in selected) / len(selected)

    peak_used = max(value["memory_used"] for value in selected)
    total = max(value["memory_total"] for value in selected)
    return {
        "sample_count": len(selected),
        "sample_interval_seconds": 2.0,
        "measured_gate_seconds": elapsed,
        "mean_gpu_utilization_percent": mean("gpu"),
        "maximum_gpu_utilization_percent": max(value["gpu"] for value in selected),
        "mean_memory_utilization_percent": mean("memory_utilization"),
        "peak_memory_used_mib": peak_used,
        "memory_total_mib": total,
        "peak_memory_fraction": peak_used / total if total else 0.0,
        "mean_power_watts": mean("power"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--hourly-cost-usd", required=True, type=float)
    parser.add_argument("--remaining-budget-usd", required=True, type=float)
    parser.add_argument("--post-corpus-reserve-minutes", required=True, type=float)
    parser.add_argument("--corpus-target-minutes", default=0.0, type=float)
    parser.add_argument("--total-target-minutes", default=0.0, type=float)
    parser.add_argument("--gpu-telemetry", default="")
    parser.add_argument("--gate-timings", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    status = json.loads(Path(args.status).read_text(encoding="utf-8"))
    inventories = []
    for value in status.get("units", {}).values():
        path = Path(str(value.get("inventory", "")))
        if value.get("exit_code") == 0 and path.is_file():
            inventories.append(json.loads(path.read_text(encoding="utf-8")))
    estimate = estimate_phase4_runtime(
        plan,
        status,
        inventories,
        hourly_cost_usd=args.hourly_cost_usd,
        remaining_budget_usd=args.remaining_budget_usd,
        post_corpus_reserve_seconds=args.post_corpus_reserve_minutes * 60.0,
        corpus_target_seconds=args.corpus_target_minutes * 60.0,
        total_target_seconds=args.total_target_minutes * 60.0,
        gpu_telemetry=(
            _calibration_gpu_summary(
                Path(args.gpu_telemetry),
                Path(args.gate_timings),
            )
            if args.gpu_telemetry and args.gate_timings
            else None
        ),
    )
    atomic_json(args.output, estimate)
    forecast = estimate["forecast_seconds"]["total_remaining"]
    cost = estimate["forecast_cost_usd"]
    print("Phase 4 runtime estimate (advisory; no automatic budget failure)")
    print(
        f"Remaining time: {forecast['point'] / 60:.1f} min "
        f"(range {forecast['low'] / 60:.1f}-{forecast['high'] / 60:.1f})"
    )
    print(
        f"Remaining cost: ${cost['point']:.2f} "
        f"(range ${cost['low']:.2f}-${cost['high']:.2f})"
    )
    print(f"Budget signal: {estimate['budget_signal']}")
    print(f"GPU tuning signal: {estimate['gpu_tuning_signal']}")
    print(f"Decision required: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
