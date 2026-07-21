"""Registered multi-condition OWL experiments with one authoritative replay condition."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from owl.core.config import load_config
from owl.experiments.manifest import ExperimentManifest
from owl.experiments.progress import ProgressJournal, atomic_write_json
from owl.replay.manifest import sha256_file

_ALLOWED_DIFFERENCE_PATHS = {
    "world.seed",
    "world.max_steps",
    "recording.enabled",
    "recording.metrics_path",
    "recording.zarr_path",
    "visualization.enabled",
    "visualization.backend",
    "raqic.actualization_variant",
    "raqic.utility_coupling",
    "raqic.utility_projection_epsilon",
    "raqic.utility_bound_floor",
    "raqic.phase_resonance_coupling",
    "raqic.phase_resonance_patch_weight",
    "raqic.phase_resonance_global_weight",
    "raqic.interference_mixer_strength",
    "raqic.interference_trotter_steps",
    "raqic.interference_action_graph",
    "raqic.experimental_shadow_only",
    "raqic.record_actualization_diagnostics",
    "raqic.qiskit_circuit_families",
    "raqic.qiskit_authoritative_family",
    "raqic.full_gpu_phase_policy",
}


def _condition_environment() -> dict[str, str]:
    """Prevent four CUDA-world orchestrators from multiplying BLAS threads."""

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


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            output.update(_flatten(item, path))
        return output
    return {prefix: value}


def _physical_payload(config_path: str | Path) -> dict[str, Any]:
    payload = load_config(config_path).model_dump(mode="json")
    flat = _flatten(payload)
    return {key: value for key, value in flat.items() if key not in _ALLOWED_DIFFERENCE_PATHS}


def validate_condition_compatibility(manifest: ExperimentManifest) -> dict[str, Any]:
    baseline = _physical_payload(manifest.conditions[0].config)
    differences: dict[str, list[str]] = {}
    for condition in manifest.conditions[1:]:
        candidate = _physical_payload(condition.config)
        changed = sorted(
            key for key in set(baseline) | set(candidate) if baseline.get(key) != candidate.get(key)
        )
        if changed:
            differences[condition.name] = changed
    if differences:
        raise ValueError(f"unregistered physical differences between conditions: {differences}")
    return {
        "passed": True,
        "condition_count": len(manifest.conditions),
        "allowed_difference_paths": sorted(_ALLOWED_DIFFERENCE_PATHS),
    }


def validate_registered_experiment(
    *,
    repo: Path,
    manifest_path: Path,
    run_root: Path,
    allow_cpu: bool = False,
) -> dict[str, Any]:
    from owl.experiments.controller import validate_preflight

    manifest = ExperimentManifest.load(manifest_path)
    compatibility = validate_condition_compatibility(manifest)
    run_root.mkdir(parents=True, exist_ok=True)
    full = next(condition for condition in manifest.conditions if condition.full_replay)
    full_root = _world_root(
        run_root,
        seed=manifest.seeds[0],
        condition=full.name,
        seed_count=len(manifest.seeds),
    )
    primary = validate_preflight(
        repo=repo,
        config_path=Path(full.config),
        run_root=full_root,
        allow_cpu=allow_cpu,
    )
    receipts: dict[str, Any] = {}
    for condition in manifest.conditions:
        cfg = load_config(condition.config)
        if str(cfg.visualization.backend) != "none":
            raise ValueError(f"condition {condition.name} must use visualization.backend=none")
        receipt = primary if condition.name == full.name else {
                **primary,
                "config": str(Path(condition.config).resolve()),
                "config_sha256": sha256_file(Path(condition.config)),
                "shared_runtime_binding_preflight_from": full.name,
            }
        receipts[condition.name] = receipt
        for seed in manifest.seeds:
            condition_root = _world_root(
                run_root,
                seed=seed,
                condition=condition.name,
                seed_count=len(manifest.seeds),
            )
            condition_root.mkdir(parents=True, exist_ok=True)
            atomic_write_json(condition_root / "preflight_receipt.json", receipt)
    payload = {
        "schema_version": "owl.experiment.registered-preflight.v1",
        "passed": True,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest.stable_hash(),
        "seeds": list(manifest.seeds),
        "world_count": len(manifest.seeds) * len(manifest.conditions),
        "compatibility": compatibility,
        "conditions": {
            condition.name: {
                "config": condition.config,
                "config_sha256": receipts[condition.name]["config_sha256"],
                "full_replay": condition.full_replay,
                "preflight_passed": True,
            }
            for condition in manifest.conditions
        },
        "qiskit_execution": primary["qiskit_execution"],
    }
    atomic_write_json(run_root / "registered_preflight.json", payload)
    return payload


def _read_last_metric(bundle: Path) -> dict[str, Any]:
    path = bundle / "analysis" / "tick_metrics.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else {}


def _world_root(
    run_root: Path,
    *,
    seed: int,
    condition: str,
    seed_count: int,
) -> Path:
    """Preserve the v1 single-seed layout and isolate multi-seed worlds."""

    if seed_count == 1:
        return run_root / "conditions" / condition
    return run_root / "seeds" / str(seed) / "conditions" / condition


def start_registered_experiment(
    *,
    repo: Path,
    manifest_path: Path,
    run_root: Path,
    hourly_cost: float,
    max_cost: float | None,
    max_runtime_hours: float | None,
    max_output_gib: float | None,
) -> int:
    manifest = ExperimentManifest.load(manifest_path)
    preflight_path = run_root / "registered_preflight.json"
    if not preflight_path.exists():
        raise RuntimeError("registered preflight is missing; run validate-manifest first")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("manifest_sha256") != manifest.stable_hash():
        raise RuntimeError("registered preflight is stale for this manifest")
    progress = ProgressJournal(run_root)
    progress.update(
        state="RUNNING",
        phase="registered_conditions",
        current_condition=None,
        completed_conditions=0,
        condition_count=len(manifest.conditions),
        seed_count=len(manifest.seeds),
        world_count=len(manifest.conditions) * len(manifest.seeds),
        completed_worlds=0,
    )
    summaries: dict[str, dict[str, Any]] = {}
    experiment_started = time.monotonic()
    pending = [
        (seed, condition)
        for seed in manifest.seeds
        for condition in manifest.conditions
    ]
    world_count = len(pending)
    active: dict[int, dict[str, Any]] = {}

    def terminate_active() -> None:
        for job in active.values():
            process = job["process"]
            if process.poll() is None:
                process.terminate()
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and any(
            job["process"].poll() is None for job in active.values()
        ):
            time.sleep(0.1)
        for job in active.values():
            process = job["process"]
            if process.poll() is None:
                process.kill()
            job["log"].close()

    while pending or active:
        elapsed_hours = (time.monotonic() - experiment_started) / 3600.0
        remaining_runtime = (
            None
            if max_runtime_hours is None
            else max(0.0, float(max_runtime_hours) - elapsed_hours)
        )
        spent_estimate = elapsed_hours * float(hourly_cost)
        remaining_cost = None if max_cost is None else max(0.0, float(max_cost) - spent_estimate)
        if remaining_runtime is not None and remaining_runtime <= 0:
            terminate_active()
            progress.update(
                state="INTERRUPTED_RESUMABLE",
                phase="registered_budget_guard",
                current_condition=None,
                completed_conditions=len(summaries),
                condition_count=len(manifest.conditions),
                completed_worlds=len(summaries),
                world_count=world_count,
                elapsed_hours=elapsed_hours,
                estimated_cost=spent_estimate,
                reason="maximum registered-experiment runtime reached",
            )
            return 2
        if remaining_cost is not None and remaining_cost <= 0:
            terminate_active()
            progress.update(
                state="INTERRUPTED_RESUMABLE",
                phase="registered_budget_guard",
                current_condition=None,
                completed_conditions=len(summaries),
                condition_count=len(manifest.conditions),
                completed_worlds=len(summaries),
                world_count=world_count,
                elapsed_hours=elapsed_hours,
                estimated_cost=spent_estimate,
                reason="maximum registered-experiment cost reached",
            )
            return 2
        while pending and len(active) < manifest.max_concurrent_conditions:
            seed, condition = pending.pop(0)
            job_id = f"{seed}:{condition.name}"
            condition_root = _world_root(
                run_root,
                seed=seed,
                condition=condition.name,
                seed_count=len(manifest.seeds),
            )
            full_replay = condition.full_replay and seed == manifest.seeds[0]
            command = [
                sys.executable,
                "-m",
                "owl.experiments.controller",
                "start",
                "--repo",
                str(repo),
                "--config",
                str(condition.config),
                "--run-root",
                str(condition_root),
                "--condition",
                condition.name,
                "--ticks",
                str(manifest.ticks),
                "--seed",
                str(seed),
                "--recording-tier",
                manifest.recording_tier if full_replay else "metrics_only",
                "--hourly-cost",
                str(float(hourly_cost)),
                "--progress-every",
                str(manifest.progress_every),
            ]
            if remaining_cost is not None:
                command.extend(("--max-cost", str(remaining_cost)))
            if remaining_runtime is not None:
                command.extend(("--max-runtime-hours", str(remaining_runtime)))
            if max_output_gib is not None:
                command.extend(("--max-output-gib", str(max_output_gib)))
            log_path = condition_root / "logs" / "registered_runner.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=repo,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=_condition_environment(),
            )
            active[process.pid] = {
                "process": process,
                "condition": condition,
                "seed": seed,
                "job_id": job_id,
                "full_replay": full_replay,
                "root": condition_root,
                "log": log,
                "log_path": log_path,
            }
        progress.update(
            state="RUNNING",
            phase="concurrent_conditions",
            current_condition=None,
            active_conditions=[job["job_id"] for job in active.values()],
            completed_conditions=len(summaries),
            condition_count=len(manifest.conditions),
            completed_worlds=len(summaries),
            world_count=world_count,
            max_concurrent_conditions=manifest.max_concurrent_conditions,
        )
        completed = [
            pid for pid, job in active.items() if job["process"].poll() is not None
        ]
        if not completed:
            time.sleep(0.2)
            continue
        for pid in completed:
            job = active.pop(pid)
            job["log"].close()
            condition = job["condition"]
            seed = int(job["seed"])
            job_id = str(job["job_id"])
            condition_root = job["root"]
            exit_code = int(job["process"].returncode)
            summaries[job_id] = {
                "seed": seed,
                "condition": condition.name,
                "full_replay": bool(job["full_replay"]),
                "exit_code": exit_code,
                "console_log": str(job["log_path"]),
                **_read_last_metric(condition_root / "bundle"),
            }
            if exit_code == 0:
                continue
            terminate_active()
            progress.update(
                state="FAILED_PARTIAL" if exit_code != 2 else "INTERRUPTED_RESUMABLE",
                phase="condition_failed",
                current_condition=condition.name,
                completed_conditions=len(summaries),
                condition_count=len(manifest.conditions),
                completed_worlds=len(summaries),
                world_count=world_count,
                exit_code=exit_code,
            )
            ordered_partial = [
                summaries[f"{seed_value}:{item.name}"]
                for seed_value in manifest.seeds
                for item in manifest.conditions
                if f"{seed_value}:{item.name}" in summaries
            ]
            atomic_write_json(
                run_root / "variant_summary.json",
                {"conditions": ordered_partial},
            )
            return int(exit_code)
    ordered = [
        summaries[f"{seed}:{item.name}"]
        for seed in manifest.seeds
        for item in manifest.conditions
    ]
    fields = sorted({key for row in ordered for key in row})
    with (run_root / "variant_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ordered)
    atomic_write_json(run_root / "variant_summary.json", {"conditions": ordered})
    progress.update(
        state="SUCCEEDED",
        phase="registered_complete",
        current_condition=None,
        completed_conditions=len(manifest.conditions),
        condition_count=len(manifest.conditions),
        completed_worlds=world_count,
        world_count=world_count,
    )
    return 0
