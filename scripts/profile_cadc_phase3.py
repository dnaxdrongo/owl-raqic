#!/usr/bin/env python3
"""Normalize counterfactual performance, memory, transfer, and device evidence."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import traceback
from pathlib import Path
from typing import Any


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def nvidia_snapshot() -> list[dict[str, str]]:
    query = "name,uuid,driver_version,memory.total,memory.used,utilization.gpu,utilization.memory"
    try:
        process = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    names = query.split(",")
    return [
        dict(zip(names, (value.strip() for value in line.split(",")), strict=True))
        for line in process.stdout.splitlines()
        if line.strip()
    ]


def memory_samples(input_root: Path) -> dict[str, Any]:
    path = input_root / "gpu_memory_samples.csv"
    if not path.is_file():
        return {"sample_count": 0}
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {"sample_count": 0}
    used = [int(row["memory_used_mib"]) for row in rows]
    total = [int(row["memory_total_mib"]) for row in rows]
    rss = [int(row["process_rss_kib"]) for row in rows]
    return {
        "sample_count": len(rows),
        "baseline_used_mib": used[0],
        "peak_used_mib": max(used),
        "total_mib": min(total),
        "peak_delta_mib": max(used) - used[0],
        "peak_process_rss_kib": max(rss),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    input_root = Path(args.input).resolve() if args.input else output.parent
    manifest_path = input_root / "phase3_acceptance_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise TypeError("acceptance manifest must be a JSON object")
        performance = manifest.get("performance", {})
        transfer = manifest.get("transfer", {})
        memory = manifest.get("memory_plan", {})
        required_sections = ("device", "performance", "memory_plan", "transfer", "parquet")
        missing_sections = [name for name in required_sections if name not in manifest]
        upstream_passed = manifest.get("passed") is True
        failures = []
        if not upstream_passed:
            failures.append(
                "upstream_acceptance_failed: " + "; ".join(map(str, manifest.get("failures", [])))
            )
        if missing_sections:
            failures.append("missing_acceptance_sections: " + ",".join(missing_sections))
        if float(performance.get("branch_ticks_per_second", 0)) <= 0:
            failures.append("nonpositive_branch_ticks_per_second")
        if float(performance.get("branches_per_second", 0)) <= 0:
            failures.append("nonpositive_branches_per_second")
        if memory.get("passed") is not True:
            failures.append("memory_plan_not_passed")
        payload = {
            "schema_version": "owl.cadc.phase3-performance.v1",
            "passed": not failures,
            "classification": (
                "PERFORMANCE_VALIDATED" if not failures else "FAILED_CLOSED_UPSTREAM_OR_PROFILE"
            ),
            "failures": failures,
            "upstream_acceptance_passed": upstream_passed,
            "upstream_failure_stage": manifest.get("failure_stage"),
            "config": str(Path(args.config).resolve()),
            "input_manifest": str(manifest_path),
            "device": manifest.get("device", {}),
            "nvidia_smi": nvidia_snapshot(),
            "observed_memory": memory_samples(input_root),
            "performance": performance,
            "memory": memory,
            "transfer": transfer,
            "parquet": manifest.get("parquet", {}),
            "limitations": [
                "kernel count and stream overlap require an external Nsight trace",
                "nvidia-smi utilization is a post-run snapshot, not a time series",
            ],
        }
    except Exception as exc:
        payload = {
            "schema_version": "owl.cadc.phase3-performance.v1",
            "passed": False,
            "classification": "FAILED_CLOSED_PROFILE_EXCEPTION",
            "failures": [f"profile_exception: {type(exc).__name__}: {exc}"],
            "traceback": traceback.format_exc().splitlines(),
            "config": str(Path(args.config).resolve()),
            "input_manifest": str(manifest_path),
            "nvidia_smi": nvidia_snapshot(),
            "observed_memory": memory_samples(input_root),
            "device": {},
            "performance": {},
            "memory": {},
            "transfer": {},
            "parquet": {},
        }
    atomic_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
