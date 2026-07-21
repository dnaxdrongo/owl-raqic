from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import zipfile
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from owl.core.actions import Action
from owl.core.config import load_config
from owl.experiments.process_control import (
    RunLock,
    process_alive,
    read_control_record,
    stop_process_group,
    write_control_record,
)
from owl.experiments.progress import ProgressJournal, atomic_write_json
from owl.record.replay_recorder import ReplayRecorder
from owl.replay.manifest import sha256_file
from owl.replay.zarr_source import ZarrReplayDataSource
from owl.viz.event_bus import events_from_state
from owl.viz.visual_snapshot import snapshot_from_device_state
from owl_raqic.qiskit_backend.per_ow_executor import PerOWQiskitExecutor
from owl_raqic.qiskit_backend.qiskit_policy import QiskitExecutionPolicy

TERMINAL_STATES = {
    "SUCCEEDED",
    "FAILED_PARTIAL",
    "INTERRUPTED_RESUMABLE",
    "CANCELLED",
    "INVALID_PREFLIGHT",
    "PACKAGED",
}


def _release_hash(repo: Path) -> str:
    """Hash the executable source surface used by a preflight receipt."""

    digest = hashlib.sha256()
    included_roots = (repo / "src", repo / "scripts", repo / "configs", repo / "experiments")
    allowed_suffixes = {".py", ".toml", ".yaml", ".yml", ".json"}
    paths: list[Path] = [repo / "pyproject.toml"]
    for root in included_roots:
        if root.exists():
            paths.extend(path for path in root.rglob("*") if path.is_file())
    for path in sorted(set(paths)):
        if not path.exists() or (
            path.suffix not in allowed_suffixes and path.name != "pyproject.toml"
        ):
            continue
        if any(
            part
            in {
                ".venv",
                ".git",
                "__pycache__",
                "runs",
                "reports",
                "quality",
                "build",
                "dist",
            }
            for part in path.parts
        ):
            continue
        digest.update(path.relative_to(repo).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _config_hash(path: Path) -> str:
    return sha256_file(path)


def _receipt_path(run_root: Path) -> Path:
    return run_root / "preflight_receipt.json"


def _hardware_probe(cfg: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    try:
        import cupy as cp

        count = int(cp.cuda.runtime.getDeviceCount())
        payload["cuda_device_count"] = count
        if count:
            props = cp.cuda.runtime.getDeviceProperties(0)
            name = props["name"]
            if isinstance(name, bytes):
                name = name.decode()
            payload.update(
                {
                    "gpu_name": str(name),
                    "compute_capability": f"{props['major']}.{props['minor']}",
                    "cupy": cp.__version__,
                }
            )
    except Exception as exc:
        payload["cuda_probe_error"] = repr(exc)
    payload["qiskit_device"] = str(cfg.raqic.qiskit_gpu_device)
    payload["qiskit_method"] = str(cfg.raqic.qiskit_gpu_method)
    return payload


def validate_preflight(
    *,
    repo: Path,
    config_path: Path,
    run_root: Path,
    allow_cpu: bool = False,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    source_hash = _release_hash(repo)
    config_hash = _config_hash(config_path)
    policy = QiskitExecutionPolicy.from_config(cfg)
    if not policy.runtime_parameter_binding:
        raise ValueError("flagship config must enable required native runtime binding")
    if policy.runtime_binding_policy != "required_native":
        raise ValueError("flagship config must use qiskit_runtime_binding_policy=required_native")
    if policy.automatic_execution_fallback:
        raise ValueError("automatic Qiskit execution fallback is prohibited")
    if str(cfg.visualization.backend) not in {"pygame", "none"}:
        raise ValueError("visualization.backend is invalid")
    if str(cfg.visualization.backend) != "none":
        raise ValueError("GPU replay-recording config must use visualization.backend=none")
    executor = PerOWQiskitExecutor(policy, seed=int(cfg.world.seed))
    if allow_cpu:
        from owl_raqic.qiskit_backend.native_state_preparation import (
            preflight_native_runtime_binding,
        )

        qiskit = preflight_native_runtime_binding(
            action_count=len(Action),
            method=str(policy.method),
            device="CPU",
            strict_gpu=False,
            tolerance=float(policy.runtime_binding_preflight_tolerance),
            batch_size=int(policy.runtime_binding_preflight_batch_size),
            seed=int(cfg.world.seed),
        )
    else:
        qiskit = executor.preflight_required_runtime_binding(len(Action))
    receipt = {
        "schema_version": "owl.experiment.preflight.v1",
        "passed": True,
        "created_at": datetime.now(UTC).isoformat(),
        "scientific_ticks_started": 0,
        "repo": str(repo),
        "config": str(config_path),
        "source_sha256": source_hash,
        "config_sha256": config_hash,
        "hardware": _hardware_probe(cfg),
        "qiskit_execution": qiskit,
        "visualization_backend": str(cfg.visualization.backend),
        "automatic_fallback_allowed": False,
    }
    run_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(_receipt_path(run_root), receipt)
    return receipt


def _load_valid_receipt(repo: Path, config: Path, run_root: Path) -> dict[str, Any]:
    path = _receipt_path(run_root)
    if not path.exists():
        raise RuntimeError("missing preflight receipt; run validate before start")
    value = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    if value.get("passed") is not True:
        raise RuntimeError("preflight receipt is not passing")
    if value.get("source_sha256") != _release_hash(repo):
        raise RuntimeError("preflight receipt source hash is stale")
    if value.get("config_sha256") != _config_hash(config):
        raise RuntimeError("preflight receipt config hash is stale")
    qiskit = value.get("qiskit_execution") or {}
    if not qiskit.get("passed") or not qiskit.get("runtime_binding_used"):
        raise RuntimeError("preflight receipt lacks required-native runtime binding proof")
    return value


def _snapshot_with_events(run: Any, *, recording_tier: str) -> Any:
    if recording_tier == "metrics_only":
        return snapshot_from_device_state(
            run.ds,
            field_names=(
                "health",
                "resource",
                "toxin",
                "food",
                "waste",
                "occupancy",
                "readout",
                "raqic_readout",
                "integration",
                "raqic_record_confidence",
                "raqic_utility_innovation_norm",
                "raqic_interference_delta_l1",
            ),
        )
    base = snapshot_from_device_state(run.ds)
    namespace = SimpleNamespace(tick=base.tick, **base.arrays)
    events = events_from_state(
        namespace,
        max_events=max(4096, int(base.world_shape[0] * base.world_shape[1])),
        strict=True,
    )
    return replace(base, events=tuple(events.events))


def _write_bundle_provenance(
    bundle: Path,
    *,
    cfg: Any,
    receipt: dict[str, Any],
    args: argparse.Namespace,
    run_root: Path,
) -> None:
    import yaml

    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "normalized_config.yaml").write_text(
        yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    (bundle / "experiment_manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "owl.experiment.condition.v1",
                "condition": str(args.condition),
                "seed": int(args.seed),
                "ticks": int(args.ticks),
                "recording_tier": str(args.recording_tier),
                "source_sha256": receipt.get("source_sha256"),
                "config_sha256": receipt.get("config_sha256"),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    atomic_write_json(bundle / "preflight_receipt.json", receipt)
    logs = bundle / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for name in ("run_progress.json", "run_progress.jsonl", "control.json"):
        source = run_root / name
        if source.exists():
            (logs / name).write_bytes(source.read_bytes())


def _save_portable_checkpoint(run: Any, run_root: Path) -> Path:
    from owl.record.snapshots import save_snapshot

    state = run.checkpoint(force=True)
    checkpoint_dir = run_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    target = checkpoint_dir / f"world_tick_{int(state.tick):08d}.npz"
    save_snapshot(state, str(target))
    atomic_write_json(
        checkpoint_dir / "latest.json",
        {
            "schema_version": "owl.experiment.checkpoint.v1",
            "tick": int(state.tick),
            "path": str(target),
            "sha256": sha256_file(target),
            "created_at": datetime.now(UTC).isoformat(),
        },
    )
    return target


def _run_foreground(args: argparse.Namespace) -> int:
    repo = Path(args.repo).resolve()
    config_path = Path(args.config).resolve()
    run_root = Path(args.run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    receipt = _load_valid_receipt(repo, config_path, run_root)
    progress = ProgressJournal(run_root)
    lock = RunLock(run_root / ".experiment.lock")
    recorder: ReplayRecorder | None = None
    run: Any | None = None
    interrupted = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    started = time.monotonic()
    with lock:
        control = write_control_record(run_root, sys.argv)
        progress.update(
            state="RUNNING",
            phase="initialize",
            pid=control.pid,
            pgid=control.pgid,
            current_tick=0,
            requested_ticks=int(args.ticks),
        )
        cfg = load_config(config_path).model_copy(deep=True)
        cfg.world.max_steps = int(args.ticks)
        cfg.world.seed = int(args.seed)
        cfg.visualization.enabled = False
        cfg.visualization.backend = "none"
        source_hash = str(receipt["source_sha256"])
        config_hash = str(receipt["config_sha256"])
        bundle = run_root / "bundle"
        resume_mode = bool(getattr(args, "resume", False))
        initial_state = None
        if resume_mode:
            from owl.record.snapshots import load_snapshot
            from owl.replay.manifest import ReplayManifest

            manifest = ReplayManifest.load(bundle)
            if manifest.source_sha256 != source_hash or manifest.config_sha256 != config_hash:
                raise RuntimeError("resume source/config hashes do not match the existing bundle")
            if manifest.requested_ticks != int(args.ticks):
                raise RuntimeError("resume tick target differs from the original run")
            latest = json.loads(
                (run_root / "checkpoints" / "latest.json").read_text(encoding="utf-8")
            )
            checkpoint_path = Path(str(latest["path"]))
            if not checkpoint_path.exists() or sha256_file(checkpoint_path) != latest["sha256"]:
                raise RuntimeError("latest portable checkpoint is missing or corrupt")
            initial_state = load_snapshot(str(checkpoint_path))
            if int(initial_state.tick) != int(latest["tick"]):
                raise RuntimeError("checkpoint tick metadata does not match checkpoint state")
            recorder = ReplayRecorder.resume(bundle)
        else:
            recorder = ReplayRecorder(
                bundle,
                run_id=run_root.name,
                condition=str(args.condition),
                seed=int(args.seed),
                requested_ticks=int(args.ticks),
                recording_tier=str(args.recording_tier),
                source_sha256=source_hash,
                config_sha256=config_hash,
                action_names=[action.name for action in Action],
                hardware=dict(receipt.get("hardware", {})),
                qiskit_execution=dict(receipt.get("qiskit_execution", {})),
                max_output_bytes=(
                    None if args.max_output_gib is None else int(args.max_output_gib * 1024**3)
                ),
                materialization_mode=str(getattr(args, "materialization_mode", "inline")),
                cadc_config=cfg.recording.cadc,
            )
        _write_bundle_provenance(bundle, cfg=cfg, receipt=receipt, args=args, run_root=run_root)
        from owl.gpu.run_context import PersistentOWLDeviceRun

        try:
            run = PersistentOWLDeviceRun.from_config(
                cfg,
                initial_state=initial_state,
                output_root=run_root / "scientific",
            )
            progress.update(
                state="RUNNING",
                phase="simulate_and_record",
                pid=control.pid,
                pgid=control.pgid,
                current_tick=0,
                requested_ticks=int(args.ticks),
            )
            remaining_ticks = max(0, int(args.ticks) - int(run.ds.tick))
            progress_every = int(getattr(args, "progress_every", 5) or 5)
            if progress_every < 1:
                raise ValueError("progress_every must be positive")
            for _ in range(remaining_ticks):
                if interrupted:
                    break
                diagnostics = run.step()
                snapshot = _snapshot_with_events(run, recording_tier=str(args.recording_tier))
                recorder.record_device(run.ds, snapshot, diagnostics=diagnostics)
                elapsed_hours = (time.monotonic() - started) / 3600.0
                estimated_cost = elapsed_hours * float(args.hourly_cost or 0.0)
                emit_progress = (
                    int(snapshot.tick) % progress_every == 0
                    or int(snapshot.tick) >= int(args.ticks)
                )
                if emit_progress:
                    progress.update(
                        state="RUNNING",
                        phase="simulate_record_async",
                        pid=control.pid,
                        pgid=control.pgid,
                        current_tick=int(snapshot.tick),
                        requested_ticks=int(args.ticks),
                        elapsed_hours=elapsed_hours,
                        estimated_cost=estimated_cost,
                        progress_every=progress_every,
                        last_checkpoint=str(
                            run.run_paths.checkpoints if run.run_paths else ""
                        ),
                    )
                    atomic_write_json(
                        run_root / "heartbeat.json",
                        {
                            "tick": int(snapshot.tick),
                            "elapsed_hours": elapsed_hours,
                            "estimated_cost": estimated_cost,
                            "progress_every": progress_every,
                            "updated_at": datetime.now(UTC).isoformat(),
                        },
                    )
                checkpoint_every = int(getattr(args, "checkpoint_every", 25) or 0)
                if checkpoint_every > 0 and int(snapshot.tick) % int(checkpoint_every) == 0:
                    checkpoint_path = _save_portable_checkpoint(run, run_root)
                    progress.update(
                        state="CHECKPOINTING",
                        phase="checkpoint_committed",
                        current_tick=int(snapshot.tick),
                        requested_ticks=int(args.ticks),
                        last_checkpoint=str(checkpoint_path),
                    )
                if args.max_runtime_hours is not None and elapsed_hours >= args.max_runtime_hours:
                    interrupted = True
                if args.max_cost is not None:
                    reserve = (
                        float(args.hourly_cost or 0.0)
                        * float(getattr(args, "stop_before_budget_exhaustion_minutes", 10.0) or 0.0)
                        / 60.0
                    )
                    if estimated_cost >= max(0.0, float(args.max_cost) - reserve):
                        interrupted = True
                if interrupted:
                    _save_portable_checkpoint(run, run_root)
                    break
            if interrupted:
                _write_bundle_provenance(
                    bundle, cfg=cfg, receipt=receipt, args=args, run_root=run_root
                )
                recorder.close(state="INTERRUPTED_RESUMABLE", failure="budget_or_signal_stop")
                progress.update(
                    state="INTERRUPTED_RESUMABLE",
                    phase="closed",
                    current_tick=int(run.ds.tick),
                    requested_ticks=int(args.ticks),
                )
                return 2
            _write_bundle_provenance(bundle, cfg=cfg, receipt=receipt, args=args, run_root=run_root)
            closed_manifest = recorder.close(state="SUCCEEDED")
            if closed_manifest.materialization_state == "pending":
                progress.update(
                    state="SCIENTIFIC_ARRAYS_COMPLETE_MATERIALIZATION_PENDING",
                    phase="materialization_pending",
                    current_tick=int(run.ds.tick),
                    requested_ticks=int(args.ticks),
                )
                return 3
            progress.update(
                state="SUCCEEDED",
                phase="validated_bundle",
                current_tick=int(run.ds.tick),
                requested_ticks=int(args.ticks),
            )
            return 0
        except Exception as exc:
            close_error: str | None = None
            if recorder is not None and recorder._ticks:
                try:
                    _write_bundle_provenance(
                        bundle, cfg=cfg, receipt=receipt, args=args, run_root=run_root
                    )
                    recorder.close(state="FAILED_PARTIAL", failure=repr(exc))
                except Exception as recorder_exc:  # preserve the original failure boundary
                    close_error = repr(recorder_exc)
            progress.update(
                state="FAILED_PARTIAL",
                phase="failed",
                error=repr(exc),
                recorder_close_error=close_error,
                current_tick=int(run.ds.tick) if run is not None else 0,
                requested_ticks=int(args.ticks),
            )
            raise
        finally:
            if run is not None:
                run.close(checkpoint=interrupted)


def _detach(args: argparse.Namespace) -> int:
    command = [sys.executable, "-m", "owl.experiments.controller", *sys.argv[1:]]
    filtered = [item for item in command if item != "--detach"]
    log = Path(args.run_root) / "logs" / "controller.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as handle:
        process = subprocess.Popen(
            filtered,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd=args.repo,
        )
    print(json.dumps({"launched": True, "pid": process.pid, "log": str(log)}, indent=2))
    return 0


def _status(run_root: Path) -> int:
    progress = run_root / "run_progress.json"
    payload = json.loads(progress.read_text(encoding="utf-8")) if progress.exists() else {}
    control_path = run_root / "control.json"
    if control_path.exists():
        record = read_control_record(run_root)
        payload["process_alive"] = process_alive(record.pid)
        payload["pid"] = record.pid
        payload["pgid"] = record.pgid
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _package(run_root: Path, *, allow_incomplete: bool = False) -> int:
    status_path = run_root / "run_progress.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    state = str(status.get("state", "UNKNOWN"))
    success = state == "SUCCEEDED"
    if (
        state in {"SCIENTIFIC_ARRAYS_COMPLETE_MATERIALIZATION_PENDING", "MATERIALIZING"}
        and not allow_incomplete
    ):
        raise RuntimeError(
            "refusing to package a bundle with pending action-table materialization; "
            "run owl-experiment materialize or pass --allow-incomplete for diagnostic evidence"
        )
    if success:
        bundle = run_root / "bundle"
        verification = ZarrReplayDataSource(bundle).verify(metadata_only=False)
        if verification.get("passed") is not True:
            raise RuntimeError(
                f"refusing success packaging because replay verification failed: {verification}"
            )
        status["bundle_verification"] = verification
        atomic_write_json(status_path, status)
    label = "results" if success else f"{state}_evidence"
    output = run_root / "packages" / f"{run_root.name}_{label}.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for path in sorted(run_root.rglob("*")):
            if not path.is_file() or output == path or "packages" in path.parts:
                continue
            archive.write(path, path.relative_to(run_root.parent))
    digest = sha256_file(output)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{digest}  {output.name}\n",
        encoding="utf-8",
    )
    if success:
        status["state"] = "PACKAGED"
        status["package"] = str(output)
        atomic_write_json(status_path, status)
    print(output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="owl-experiment")
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--repo", required=True)
    plan.add_argument("--config", required=True)
    plan.add_argument("--ticks", type=int, required=True)
    plan.add_argument("--recording-tier", default="analysis_full")
    plan.add_argument("--hourly-cost", type=float, default=0.0)
    validate = sub.add_parser("validate")
    validate.add_argument("--repo", required=True)
    validate.add_argument("--config", required=True)
    validate.add_argument("--run-root", required=True)
    validate.add_argument("--allow-cpu-preflight", action="store_true")
    start = sub.add_parser("start")
    start.add_argument("--repo", required=True)
    start.add_argument("--config", required=True)
    start.add_argument("--run-root", required=True)
    start.add_argument("--condition", default="all_on")
    start.add_argument("--ticks", type=int, required=True)
    start.add_argument("--seed", type=int, default=9303)
    start.add_argument("--recording-tier", default="analysis_full")
    start.add_argument("--materialization-mode", choices=("inline", "deferred"), default="inline")
    start.add_argument("--hourly-cost", type=float, default=0.0)
    start.add_argument("--max-cost", type=float)
    start.add_argument("--max-runtime-hours", type=float)
    start.add_argument("--max-output-gib", type=float)
    start.add_argument("--checkpoint-every", type=int, default=25)
    start.add_argument("--progress-every", type=int, default=5)
    start.add_argument("--stop-before-budget-exhaustion-minutes", type=float, default=10.0)
    start.add_argument("--detach", action="store_true")
    resume = sub.add_parser("resume")
    resume.add_argument("--repo", required=True)
    resume.add_argument("--config", required=True)
    resume.add_argument("--run-root", required=True)
    resume.add_argument("--hourly-cost", type=float, default=0.0)
    resume.add_argument("--max-cost", type=float)
    resume.add_argument("--max-runtime-hours", type=float)
    resume.add_argument("--max-output-gib", type=float)
    resume.add_argument("--checkpoint-every", type=int, default=25)
    resume.add_argument("--progress-every", type=int, default=5)
    resume.add_argument("--stop-before-budget-exhaustion-minutes", type=float, default=10.0)
    resume.add_argument("--detach", action="store_true")
    validate_manifest = sub.add_parser("validate-manifest")
    validate_manifest.add_argument("--repo", required=True)
    validate_manifest.add_argument("--manifest", required=True)
    validate_manifest.add_argument("--run-root", required=True)
    validate_manifest.add_argument("--allow-cpu-preflight", action="store_true")
    start_manifest = sub.add_parser("start-manifest")
    start_manifest.add_argument("--repo", required=True)
    start_manifest.add_argument("--manifest", required=True)
    start_manifest.add_argument("--run-root", required=True)
    start_manifest.add_argument("--hourly-cost", type=float, default=0.0)
    start_manifest.add_argument("--max-cost", type=float)
    start_manifest.add_argument("--max-runtime-hours", type=float)
    start_manifest.add_argument("--max-output-gib", type=float)
    start_manifest.add_argument("--detach", action="store_true")
    status = sub.add_parser("status")
    status.add_argument("--run-root", required=True)
    stop = sub.add_parser("stop")
    stop.add_argument("--run-root", required=True)
    package = sub.add_parser("package")
    package.add_argument("--run-root", required=True)
    package.add_argument("--allow-incomplete", action="store_true")
    materialize = sub.add_parser("materialize")
    materialize.add_argument("--bundle", required=True)
    materialize.add_argument("--max-batch-rows", type=int, default=131_072)
    materialize.add_argument("--max-batch-bytes", type=int, default=128 * 1024 * 1024)
    materialize.add_argument("--row-group-rows", type=int, default=131_072)
    materialize.add_argument("--compression", default="zstd")
    verify = sub.add_parser("verify")
    verify.add_argument("--bundle", required=True)
    replay = sub.add_parser("replay")
    replay.add_argument("--bundle", required=True)
    replay.add_argument("viewer_args", nargs=argparse.REMAINDER)
    export = sub.add_parser("export-csv")
    export.add_argument("--bundle", required=True)
    export.add_argument("--ow-id", type=int, required=True)
    export.add_argument("--start-tick", type=int, required=True)
    export.add_argument("--end-tick", type=int, required=True)
    export.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Dispatch is intentionally resolved before timestamps, run roots, locks, or subprocesses.
    if args.command == "plan":
        cfg = load_config(Path(args.config).resolve())
        height = int(cfg.world.height)
        width = int(cfg.world.width)
        action_count = len(Action)
        living_upper_bound = height * width
        action_rows = (
            living_upper_bound * action_count * int(args.ticks)
            if args.recording_tier in {"analysis_full", "debug_full"}
            else 0
        )
        # Conservative upper-bound estimates intentionally assume every cell is occupied.
        replay_scalar_fields = 18
        replay_action_fields = 10 if args.recording_tier in {"analysis_full", "debug_full"} else 0
        replay_raw_bytes = (
            int(args.ticks)
            * height
            * width
            * (replay_scalar_fields * 8 + replay_action_fields * action_count * 8)
        )
        table_row_bytes = 256
        table_raw_bytes = living_upper_bound * int(args.ticks) * table_row_bytes
        table_raw_bytes += action_rows * 192
        estimated_raw_bytes = replay_raw_bytes + table_raw_bytes
        estimated_compressed_bytes = int(estimated_raw_bytes * 0.35)
        payload = {
            "schema_version": "owl.experiment.plan.v1",
            "repo": str(Path(args.repo).resolve()),
            "config": str(Path(args.config).resolve()),
            "ticks": int(args.ticks),
            "world_shape": [height, width],
            "recording_tier": str(args.recording_tier),
            "maximum_ow_tick_rows": living_upper_bound * int(args.ticks),
            "maximum_action_math_rows": action_rows,
            "estimated_raw_bytes_upper_bound": estimated_raw_bytes,
            "estimated_compressed_bytes_upper_bound": estimated_compressed_bytes,
            "estimated_compressed_gib_upper_bound": estimated_compressed_bytes / 1024**3,
            "hourly_cost": float(args.hourly_cost),
            "scientific_ticks_started": 0,
            "estimate_note": (
                "conservative occupied-world upper bound; actual compression and population vary"
            ),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "status":
        return _status(Path(args.run_root))
    if args.command == "stop":
        print(json.dumps(stop_process_group(args.run_root), indent=2))
        return 0
    if args.command == "package":
        return _package(Path(args.run_root), allow_incomplete=bool(args.allow_incomplete))
    if args.command == "materialize":
        from owl.record.action_math_materializer import materialize_action_math

        payload = materialize_action_math(
            args.bundle,
            max_batch_rows=int(args.max_batch_rows),
            max_batch_bytes=int(args.max_batch_bytes),
            row_group_rows=int(args.row_group_rows),
            compression=str(args.compression),
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "verify":
        print(json.dumps(ZarrReplayDataSource(args.bundle).verify(metadata_only=False), indent=2))
        return 0
    if args.command == "replay":
        from owl.viz.replay_app import main as replay_main

        return int(replay_main([args.bundle, *args.viewer_args]))
    if args.command == "export-csv":
        source = ZarrReplayDataSource(args.bundle)
        print(
            source.export_selection_csv(
                args.output,
                ow_id=args.ow_id,
                start_tick=args.start_tick,
                end_tick=args.end_tick,
            )
        )
        return 0
    if args.command == "resume":
        from owl.replay.manifest import ReplayManifest

        bundle = Path(args.run_root) / "bundle"
        bundle_status = json.loads((bundle / "run_status.json").read_text(encoding="utf-8"))
        if bundle_status.get("state") in {
            "SCIENTIFIC_ARRAYS_COMPLETE_MATERIALIZATION_PENDING",
            "MATERIALIZING",
        }:
            from owl.record.action_math_materializer import materialize_action_math

            payload = materialize_action_math(bundle)
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        manifest = ReplayManifest.load(bundle)
        args.condition = manifest.condition
        args.ticks = manifest.requested_ticks
        args.seed = manifest.seed
        args.recording_tier = manifest.recording_tier
        args.resume = True
        return _detach(args) if args.detach else _run_foreground(args)
    if args.command == "validate-manifest":
        from owl.experiments.registered import validate_registered_experiment

        payload = validate_registered_experiment(
            repo=Path(args.repo).resolve(),
            manifest_path=Path(args.manifest).resolve(),
            run_root=Path(args.run_root).resolve(),
            allow_cpu=bool(args.allow_cpu_preflight),
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "start-manifest":
        if args.detach:
            return _detach(args)
        from owl.experiments.registered import start_registered_experiment

        return int(
            start_registered_experiment(
                repo=Path(args.repo).resolve(),
                manifest_path=Path(args.manifest).resolve(),
                run_root=Path(args.run_root).resolve(),
                hourly_cost=float(args.hourly_cost),
                max_cost=args.max_cost,
                max_runtime_hours=args.max_runtime_hours,
                max_output_gib=args.max_output_gib,
            )
        )
    if args.command == "validate":
        receipt = validate_preflight(
            repo=Path(args.repo).resolve(),
            config_path=Path(args.config).resolve(),
            run_root=Path(args.run_root).resolve(),
            allow_cpu=bool(args.allow_cpu_preflight),
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    if args.command == "start":
        args.resume = False
        return _detach(args) if args.detach else _run_foreground(args)
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
