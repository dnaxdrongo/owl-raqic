#!/usr/bin/env python3
"""Run or resume planned corpus units with the certified counterfactual engine."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from hashlib import sha256
from pathlib import Path
from typing import Any


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _inventory(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"corpus unit inventory must be a mapping: {path}")
    return payload


def _validated_success(output: Path, unit_id: str) -> dict[str, Any] | None:
    inventory = _inventory(output / "corpus_unit_inventory.json")
    if inventory is None or inventory.get("passed") is not True:
        return None
    if inventory.get("unit_id") != unit_id:
        raise RuntimeError(f"successful corpus inventory identity mismatch: {output}")
    for field, expected in (
        ("factual_root", output / "factual_bundle"),
        ("counterfactual_root", output / "counterfactual"),
    ):
        registered = Path(str(inventory.get(field, ""))).resolve()
        if registered != expected.resolve() or not registered.is_dir():
            raise RuntimeError(f"successful corpus inventory has invalid {field}: {output}")
        if not any(path.is_file() for path in registered.rglob("*")):
            raise RuntimeError(f"successful corpus inventory has empty {field}: {output}")
    for field in ("source_decisions", "branch_horizons", "candidate_pairs"):
        if int(inventory.get(field, 0)) <= 0:
            raise RuntimeError(
                f"successful corpus inventory has nonpositive {field}: {output}"
            )
    return inventory


def _next_attempt_path(output: Path, unit_id: str) -> Path:
    root = output.parent / "_failed_attempts" / unit_id
    root.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while (root / f"attempt-{attempt:04d}").exists():
        attempt += 1
    return root / f"attempt-{attempt:04d}"


def _quarantine_failed_output(output: Path, unit_id: str) -> dict[str, Any] | None:
    """Atomically retain a failed/partial unit before creating a clean retry root."""
    if not output.exists():
        return None
    if not output.is_dir():
        raise RuntimeError(f"corpus unit output is not a directory: {output}")
    successful = _validated_success(output, unit_id)
    if successful is not None:
        raise RuntimeError(f"refusing to quarantine a successful corpus unit: {unit_id}")
    destination = _next_attempt_path(output, unit_id)
    inventory_path = output / "corpus_unit_inventory.json"
    inventory_digest = (
        sha256(inventory_path.read_bytes()).hexdigest()
        if inventory_path.is_file()
        else None
    )
    output.rename(destination)
    return {
        "path": str(destination),
        "inventory_sha256": inventory_digest,
    }


def _terminate_active(active: dict[int, dict[str, Any]]) -> None:
    """Stop only corpus-unit children owned by this runner."""
    for job in active.values():
        process = job["process"]
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and any(
        job["process"].poll() is None for job in active.values()
    ):
        time.sleep(0.05)
    for job in active.values():
        process = job["process"]
        if process.poll() is None:
            process.kill()
        job["log"].close()


def _unit_command(args: Any, helper: Path, unit: dict[str, Any], output: Path) -> list[str]:
    command = [
        sys.executable,
        str(helper),
        "--engine-root",
        str(Path(args.engine_root).resolve()),
        "--config",
        unit["derived_config_path"],
        "--output",
        str(output),
        "--unit-id",
        str(unit["unit_id"]),
        "--context-family",
        unit["context_family"],
        "--source-tick",
        str(unit["source_tick"]),
        "--phase25-certificate",
        str(Path(args.phase25_certificate).resolve()),
        "--hardening-receipt",
        str(Path(args.hardening_receipt).resolve()),
        "--backend",
        args.backend,
        "--branch-transfer-mode",
        args.branch_transfer_mode,
    ]
    if args.aggregate_device_budget_bytes:
        command.extend(
            (
                "--worker-device-budget-bytes",
                str(
                    int(args.aggregate_device_budget_bytes)
                    // int(args.max_concurrent_units)
                ),
            )
        )
    return command


def _worker_environment() -> dict[str, str]:
    """Keep CPU orchestration bounded when several CUDA workers share a pod."""

    environment = dict(os.environ)
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        environment[name] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    return environment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--engine-root", required=True)
    parser.add_argument("--phase25-certificate", required=True)
    parser.add_argument("--hardening-receipt", required=True)
    parser.add_argument("--backend", choices=("numpy", "cupy"), default="cupy")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-units", type=int, default=0)
    parser.add_argument("--max-concurrent-units", type=int, default=1)
    parser.add_argument(
        "--aggregate-device-budget-bytes",
        type=int,
        default=0,
        help=(
            "total device-memory envelope shared by all concurrent corpus units; "
            "each child receives an equal fail-closed slice"
        ),
    )
    parser.add_argument(
        "--branch-transfer-mode",
        choices=("immediate_reference", "deferred_bounded"),
        default="immediate_reference",
    )
    args = parser.parse_args()
    if args.max_concurrent_units < 1:
        raise ValueError("max-concurrent-units must be positive")
    if args.aggregate_device_budget_bytes < 0:
        raise ValueError("aggregate-device-budget-bytes cannot be negative")
    if (
        args.aggregate_device_budget_bytes
        and args.aggregate_device_budget_bytes < args.max_concurrent_units
    ):
        raise ValueError("aggregate device budget cannot provide one byte per worker")
    if args.backend == "numpy" and args.max_concurrent_units != 1:
        raise ValueError("NumPy reference corpus execution must remain serial")
    if args.backend == "numpy" and args.branch_transfer_mode != "immediate_reference":
        raise ValueError("NumPy reference corpus requires immediate branch transfers")
    plan_path = Path(args.plan).resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    units = list(plan["units"])
    if args.max_units:
        units = units[: args.max_units]
    helper = Path(__file__).with_name("_run_cadc_phase4_corpus_unit.py")
    status_path = plan_path.parent / "corpus_run_status.json"
    status: dict[str, Any] = {
        "schema_version": "owl.cadc.phase4-corpus-run-status.v1",
        "plan_id": plan["plan_id"],
        "units": {},
        "passed": False,
        "max_concurrent_units": int(args.max_concurrent_units),
        "aggregate_device_budget_bytes": int(args.aggregate_device_budget_bytes),
        "worker_device_budget_bytes": (
            int(args.aggregate_device_budget_bytes) // int(args.max_concurrent_units)
            if args.aggregate_device_budget_bytes
            else None
        ),
        "branch_transfer_mode": args.branch_transfer_mode,
    }
    if args.resume and status_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("plan_id") != plan["plan_id"]:
            raise RuntimeError("corpus resume status belongs to a different frozen plan")
        if not isinstance(status.get("units"), dict):
            raise TypeError("corpus resume status has no unit mapping")
        if int(status.get("max_concurrent_units", args.max_concurrent_units)) != int(
            args.max_concurrent_units
        ):
            raise RuntimeError("corpus resume worker count differs from the frozen run")
        if int(
            status.get(
                "aggregate_device_budget_bytes",
                args.aggregate_device_budget_bytes,
            )
        ) != int(args.aggregate_device_budget_bytes):
            raise RuntimeError("corpus resume device budget differs from the frozen run")
        if (
            status.get("branch_transfer_mode", args.branch_transfer_mode)
            != args.branch_transfer_mode
        ):
            raise RuntimeError("corpus resume branch-transfer mode differs from the frozen run")
    status["aggregate_device_budget_bytes"] = int(args.aggregate_device_budget_bytes)
    status["worker_device_budget_bytes"] = (
        int(args.aggregate_device_budget_bytes) // int(args.max_concurrent_units)
        if args.aggregate_device_budget_bytes
        else None
    )
    jobs: list[dict[str, Any]] = []
    for unit in units:
        unit_id = str(unit["unit_id"])
        existing = status["units"].get(unit_id, {})
        if args.resume and existing.get("exit_code") == 0:
            if _validated_success(Path(unit["output_path"]), unit_id) is None:
                raise RuntimeError(
                    f"successful corpus status has no passing inventory: {unit_id}"
                )
            continue
        output = Path(unit["output_path"])
        prior_attempt = None
        if args.resume and output.exists():
            recovered = _validated_success(output, unit_id)
            if recovered is not None:
                status["units"][unit_id] = {
                    "exit_code": 0,
                    "inventory": str(output / "corpus_unit_inventory.json"),
                    "console_log": str(output / "corpus_unit_console.log"),
                    "recovered_from_inventory": True,
                    "prior_attempts": existing.get("prior_attempts", []),
                }
                atomic_json(status_path, status)
                continue
            prior_attempt = _quarantine_failed_output(output, unit_id)
        output.mkdir(parents=True, exist_ok=True)
        log_path = output / "corpus_unit_console.log"
        prior_attempts = list(existing.get("prior_attempts", []))
        if prior_attempt is not None:
            prior_attempts.append(prior_attempt)
        jobs.append(
            {
                "unit": unit,
                "unit_id": unit_id,
                "output": output,
                "command": _unit_command(args, helper, unit, output),
                "log_path": log_path,
                "prior_attempts": prior_attempts,
            }
        )

    started = time.monotonic()
    active: dict[int, dict[str, Any]] = {}
    next_job = 0
    peak_active = 0
    last_heartbeat = 0.0
    try:
        while next_job < len(jobs) or active:
            while next_job < len(jobs) and len(active) < args.max_concurrent_units:
                job = jobs[next_job]
                next_job += 1
                log = job["log_path"].open("w", encoding="utf-8")
                process = subprocess.Popen(
                    job["command"],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd=Path(__file__).resolve().parents[1],
                    env=_worker_environment(),
                )
                job["process"] = process
                job["log"] = log
                job["started_monotonic"] = time.monotonic()
                active[process.pid] = job
                peak_active = max(peak_active, len(active))
                status["active_units"] = sorted(
                    value["unit_id"] for value in active.values()
                )
                status["peak_active_units"] = peak_active
                atomic_json(status_path, status)

            completed_pids = [
                pid for pid, job in active.items() if job["process"].poll() is not None
            ]
            if not completed_pids:
                now = time.monotonic()
                if now - last_heartbeat >= 5.0:
                    status["active_unit_elapsed_seconds"] = {
                        value["unit_id"]: now - value["started_monotonic"]
                        for value in active.values()
                    }
                    status["pending_units"] = len(jobs) - next_job
                    status["runtime_seconds"] = now - started
                    atomic_json(status_path, status)
                    last_heartbeat = now
                time.sleep(0.10)
                continue
            for pid in completed_pids:
                job = active.pop(pid)
                process = job["process"]
                job["log"].close()
                unit_id = job["unit_id"]
                output = job["output"]
                inventory = _inventory(output / "corpus_unit_inventory.json")
                effective_code = int(process.returncode)
                if effective_code == 0 and _validated_success(output, unit_id) is None:
                    effective_code = 1
                status["units"][unit_id] = {
                    "exit_code": effective_code,
                    "inventory": str(output / "corpus_unit_inventory.json"),
                    "console_log": str(job["log_path"]),
                    "prior_attempts": job["prior_attempts"],
                    "runtime_seconds": time.monotonic() - job["started_monotonic"],
                }
                status["active_units"] = sorted(
                    value["unit_id"] for value in active.values()
                )
                status["active_unit_elapsed_seconds"] = {
                    value["unit_id"]: time.monotonic()
                    - value["started_monotonic"]
                    for value in active.values()
                }
                status["pending_units"] = len(jobs) - next_job
                status["completed_units"] = sum(
                    int(value.get("exit_code", 1) == 0)
                    for value in status["units"].values()
                )
                if effective_code != 0:
                    status["last_failure"] = {
                        "unit_id": unit_id,
                        "exit_code": effective_code,
                        "exception_type": (inventory or {}).get("exception_type"),
                        "message": (inventory or {}).get("message"),
                        "inventory": str(output / "corpus_unit_inventory.json"),
                        "console_log": str(job["log_path"]),
                    }
                    status["passed"] = False
                    status["runtime_seconds"] = time.monotonic() - started
                    status.setdefault("execution_sessions", []).append(
                        {
                            "passed": False,
                            "unit_ids": [value["unit_id"] for value in jobs],
                            "completed_unit_ids": [
                                value["unit_id"]
                                for value in jobs
                                if status["units"].get(value["unit_id"], {}).get(
                                    "exit_code"
                                )
                                == 0
                            ],
                            "wall_seconds": status["runtime_seconds"],
                            "max_concurrent_units": int(args.max_concurrent_units),
                            "aggregate_device_budget_bytes": int(
                                args.aggregate_device_budget_bytes
                            ),
                            "branch_transfer_mode": args.branch_transfer_mode,
                        }
                    )
                    atomic_json(status_path, status)
                    _terminate_active(active)
                    return effective_code
                status.pop("last_failure", None)
                status["runtime_seconds"] = time.monotonic() - started
                atomic_json(status_path, status)
    except BaseException:
        _terminate_active(active)
        status["passed"] = False
        status["runtime_seconds"] = time.monotonic() - started
        status["interrupted"] = True
        atomic_json(status_path, status)
        raise
    status["passed"] = all(
        status["units"].get(str(unit["unit_id"]), {}).get("exit_code") == 0
        for unit in units
    )
    status["active_units"] = []
    status["active_unit_elapsed_seconds"] = {}
    status["pending_units"] = 0
    status["runtime_seconds"] = time.monotonic() - started
    status["peak_active_units"] = peak_active
    status["selected_unit_count"] = len(units)
    status["total_plan_unit_count"] = len(plan["units"])
    status["partial_planning_run"] = len(units) < len(plan["units"])
    status.setdefault("execution_sessions", []).append(
        {
            "passed": bool(status["passed"]),
            "unit_ids": [value["unit_id"] for value in jobs],
            "completed_unit_ids": [
                value["unit_id"]
                for value in jobs
                if status["units"].get(value["unit_id"], {}).get("exit_code") == 0
            ],
            "wall_seconds": status["runtime_seconds"],
            "max_concurrent_units": int(args.max_concurrent_units),
            "aggregate_device_budget_bytes": int(
                args.aggregate_device_budget_bytes
            ),
            "branch_transfer_mode": args.branch_transfer_mode,
        }
    )
    atomic_json(status_path, status)
    return 0 if status["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
