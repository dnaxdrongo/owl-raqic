#!/usr/bin/env python3
"""Run or resume the gated corpus, model, evaluation, and certification workflow."""

from __future__ import annotations

import argparse
import atexit
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from owl.cadc.artifacts import atomic_json, sha256_file  # noqa: E402
from owl.cadc.config import load_phase4_config  # noqa: E402
from owl.experiments.controller import _release_hash  # noqa: E402


def _gate_environment() -> dict[str, str]:
    """Bound host thread pools while GPU gates own the numerical workload."""

    environment = {
        **os.environ,
        "PYTHONPATH": f"{SRC}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
    }
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


class _GpuTelemetry:
    """Low-overhead target telemetry owned by the acceptance parent process."""

    def __init__(self, output: Path) -> None:
        self.output = output
        self.process: subprocess.Popen[str] | None = None
        self.handle: Any | None = None

    def start(self) -> None:
        executable = shutil.which("nvidia-smi")
        receipt = self.output / "gpu_telemetry_status.json"
        if executable is None:
            atomic_json(
                receipt,
                {
                    "schema_version": "owl.cadc.phase4-gpu-telemetry.v1",
                    "captured": False,
                    "reason": "nvidia_smi_unavailable",
                },
            )
            return
        destination = self.output / "gpu_telemetry.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.handle = destination.open("a", encoding="utf-8")
        self.process = subprocess.Popen(
            [
                executable,
                "--query-gpu=timestamp,name,pci.bus_id,utilization.gpu,"
                "utilization.memory,memory.used,memory.total,power.draw,pstate",
                "--format=csv,noheader,nounits",
                "--loop-ms=2000",
            ],
            stdout=self.handle,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        atomic_json(
            receipt,
            {
                "schema_version": "owl.cadc.phase4-gpu-telemetry.v1",
                "captured": True,
                "interval_milliseconds": 2000,
                "path": str(destination),
                "pid": self.process.pid,
            },
        )

    def stop(self) -> None:
        process = self.process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)
        if self.handle is not None:
            self.handle.close()
        self.process = None
        self.handle = None


def _run(
    name: str,
    command: list[str],
    *,
    output: Path,
    status: dict[str, int],
    resume: bool,
    required_artifact: Path | None = None,
) -> None:
    if (
        resume
        and status.get(name) == 0
        and (required_artifact is None or required_artifact.is_file())
    ):
        return
    log_path = output / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    active_path = output / "active_gate.json"
    started_at = datetime.now(UTC)
    started_clock = perf_counter()
    atomic_json(
        active_path,
        {
            "schema_version": "owl.cadc.phase4-active-gate.v1",
            "state": "RUNNING",
            "gate": name,
            "started_at": started_at.isoformat(),
            "log": str(log_path),
        },
    )
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            env=_gate_environment(),
        )
    status[name] = completed.returncode
    finished_at = datetime.now(UTC)
    timings_path = output / "gate_timings.json"
    timings: dict[str, Any] = (
        json.loads(timings_path.read_text(encoding="utf-8"))
        if timings_path.is_file()
        else {"schema_version": "owl.cadc.phase4-gate-timings.v1", "gates": {}}
    )
    gates = timings.setdefault("gates", {})
    if not isinstance(gates, dict):
        raise TypeError("Phase 4 gate timing registry is not a mapping")
    gates[name] = {
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "elapsed_seconds": perf_counter() - started_clock,
        "exit_code": completed.returncode,
        "log": str(log_path),
    }
    atomic_json(timings_path, timings)
    atomic_json(output / "command_status.json", status)
    atomic_json(
        active_path,
        {
            "schema_version": "owl.cadc.phase4-active-gate.v1",
            "state": "COMPLETED" if completed.returncode == 0 else "FAILED",
            "gate": name,
            "exit_code": completed.returncode,
            "finished_at": finished_at.isoformat(),
            "log": str(log_path),
        },
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Phase 4 gate failed: {name}; see {log_path}")


def _runtime_decision(
    decision_path: Path,
    estimate_path: Path,
    plan_path: Path,
) -> bool:
    """Validate an explicit user decision; return whether work may continue."""

    estimate = json.loads(estimate_path.read_text(encoding="utf-8"))
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if estimate.get("plan_id") != plan.get("plan_id"):
        raise RuntimeError("runtime estimate belongs to a different corpus plan")
    if decision.get("plan_id") != plan.get("plan_id"):
        raise RuntimeError("runtime decision belongs to a different corpus plan")
    if decision.get("estimate_sha256") != sha256_file(estimate_path):
        raise RuntimeError("runtime estimate changed after the user decision")
    choice = str(decision.get("choice", ""))
    if choice not in {"proceed", "reduce", "stop"}:
        raise RuntimeError("runtime decision has an unknown choice")
    if decision.get("continue_selected_profile") != (choice == "proceed"):
        raise RuntimeError("runtime decision continuation flag is inconsistent")
    return choice == "proceed"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--runtime-decision", default="")
    parser.add_argument("--hourly-cost-usd", type=float, default=0.0)
    parser.add_argument("--remaining-budget-usd", type=float, default=0.0)
    parser.add_argument("--post-corpus-reserve-minutes", type=float, default=90.0)
    parser.add_argument(
        "--reuse-certified-data",
        default="",
        help="reuse a checksum-verified Phase 4 corpus/dataset run on H200/B200",
    )
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_phase4_config(config_path)
    if config.runtime.precision == "fp8":
        raise RuntimeError(
            "B200 FP8 profile is intentionally locked until a separate Transformer "
            "Engine eager/BF16 parity certificate exists; use B200 BF16"
        )
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_binding_path = output / "source_binding.json"
    current_source = _release_hash(ROOT)
    if args.resume and source_binding_path.is_file():
        source_binding = json.loads(source_binding_path.read_text(encoding="utf-8"))
        if source_binding.get("phase4_source_sha256") != current_source:
            raise RuntimeError(
                "Phase 4 source changed after acceptance began; existing gate evidence "
                "is stale and cannot be resumed"
            )
    else:
        atomic_json(
            source_binding_path,
            {
                "schema_version": "owl.cadc.phase4-source-binding.v1",
                "phase4_source_sha256": current_source,
                "bound_at": datetime.now(UTC).isoformat(),
            },
        )
    telemetry = _GpuTelemetry(output)
    telemetry.start()
    atexit.register(telemetry.stop)
    status_path = output / "command_status.json"
    status: dict[str, int] = (
        json.loads(status_path.read_text(encoding="utf-8"))
        if args.resume and status_path.is_file()
        else {}
    )
    python = sys.executable
    reuse_root = (
        Path(args.reuse_certified_data).resolve()
        if args.reuse_certified_data
        else None
    )
    corpus = (reuse_root / "corpus") if reuse_root is not None else output / "corpus"
    plan = corpus / "corpus_plan.json"
    corpus_certificate = corpus / "corpus_certificate.json"
    dataset = (reuse_root / "dataset") if reuse_root is not None else output / "dataset"
    repeat_pilot = (
        reuse_root / "repeat_pilot.json"
        if reuse_root is not None
        else output / "repeat_pilot.json"
    )
    models = output / "models"
    calibration = output / "calibration"
    scored = output / "scored_artifacts"
    evaluation = output / "evaluation"
    negative = output / "negative_controls.json"
    math = output / "math_verification.json"
    casebook = output / "casebook"
    environment = output / "environment.json"
    gpu_stack_smoke = output / "gpu_stack_smoke.json"
    performance = output / "performance.json"
    hotpath = output / "hotpath_audit.json"
    synthetic = output / "synthetic_scenarios.json"
    certificate = output / "phase4_certificate.json"
    runtime_estimate = output / "runtime_estimate.json"
    runtime_pause = output / "runtime_decision_required.json"
    phase4_tests = [
        str(path.relative_to(ROOT))
        for path in sorted((ROOT / "tests").glob("test_cadc_phase4_*.py"))
    ]
    phase4_scripts = [
        str(path.relative_to(ROOT))
        for path in sorted((ROOT / "scripts").glob("*cadc_phase4*.py"))
    ]
    if not phase4_tests:
        raise RuntimeError("Phase 4 acceptance found no Phase 4 tests")
    prior_status = output / "command_status.pre_certifier.json"
    cert_command = [
        python,
        "scripts/certify_cadc_phase4.py",
        "--config",
        str(config_path),
        "--corpus-certificate",
        str(corpus_certificate),
        "--dataset-receipt",
        str(dataset / "dataset_build_receipt.json"),
        "--repeat-pilot",
        str(repeat_pilot),
        "--training-receipt",
        str(models / "training_receipt.json"),
        "--calibration-receipt",
        str(calibration / "calibration_receipt.json"),
        "--score-receipt",
        str(scored / "scored_artifacts_receipt.json"),
        "--evaluation",
        str(evaluation / "evaluation.json"),
        "--negative-controls",
        str(negative),
        "--math-verification",
        str(math),
        "--casebook-manifest",
        str(casebook / "casebook_manifest.json"),
        "--environment-manifest",
        str(environment),
        "--gpu-stack-smoke",
        str(gpu_stack_smoke),
        "--performance",
        str(performance),
        "--hotpath-audit",
        str(hotpath),
        "--synthetic-scenarios",
        str(synthetic),
        "--command-status",
        str(prior_status),
        "--output",
        str(certificate),
    ]
    try:
        _run(
            "pip_check",
            [python, "-m", "pip", "check"],
            output=output,
            status=status,
            resume=args.resume,
        )
        _run(
            "environment",
            [
                python,
                "scripts/capture_cadc_phase4_environment.py",
                "--config",
                str(config_path),
                "--output",
                str(environment),
            ],
            output=output,
            status=status,
            resume=args.resume,
            required_artifact=environment,
        )
        _run(
            "gpu_stack_smoke",
            [
                python,
                "scripts/preflight_cadc_phase4_gpu_stack.py",
                "--config",
                str(config_path),
                "--output",
                str(gpu_stack_smoke),
            ],
            output=output,
            status=status,
            resume=args.resume,
            required_artifact=gpu_stack_smoke,
        )
        _run(
            "math_verification",
            [
                python,
                "scripts/verify_phase4_cadc_math.py",
                "--symbolic-script",
                "quality/phase4/verify_phase4_symbolic_reference.py",
                "--output",
                str(math),
            ],
            output=output,
            status=status,
            resume=args.resume,
            required_artifact=math,
        )
        _run(
            "synthetic_scenarios",
            [
                python,
                "scripts/verify_cadc_phase4_synthetic_scenarios.py",
                "--config",
                str(config_path),
                "--output",
                str(synthetic),
            ],
            output=output,
            status=status,
            resume=args.resume,
            required_artifact=synthetic,
        )
        # Run every source-only gate exactly once before paid corpus work.
        # Resume preserves a passing gate instead of repeating it.
        _run(
            "pytest_full",
            [python, "-m", "pytest", "-q", "-W", "error"],
            output=output,
            status=status,
            resume=args.resume,
        )
        _run(
            "ruff_phase4",
            [
                python,
                "-m",
                "ruff",
                "check",
                "src/owl/cadc",
                *phase4_scripts,
                *phase4_tests,
            ],
            output=output,
            status=status,
            resume=args.resume,
        )
        _run(
            "mypy_phase4",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--follow-imports=silent",
                "--no-sqlite-cache",
                f"--cache-dir={output / 'mypy_cache'}",
                "src/owl/cadc",
            ],
            output=output,
            status=status,
            resume=args.resume,
        )
        if reuse_root is not None:
            reuse_manifest_path = reuse_root / "REUSABLE_DATA_MANIFEST.json"
            reuse_manifest = json.loads(
                reuse_manifest_path.read_text(encoding="utf-8")
            )
            corpus_payload = json.loads(corpus_certificate.read_text(encoding="utf-8"))
            dataset_payload = json.loads(
                (dataset / "dataset_build_receipt.json").read_text(encoding="utf-8")
            )
            repeat_payload = json.loads(repeat_pilot.read_text(encoding="utf-8"))
            reusable = all(
                (
                    corpus_payload.get("passed") is True,
                    corpus_payload.get("corpus_contract_sha256")
                    == config.corpus_digest(),
                    dataset_payload.get("passed") is True,
                    dataset_payload.get("corpus_contract_sha256")
                    == config.corpus_digest(),
                    dataset_payload.get("model_spec_sha256")
                    == config.model_spec_digest(),
                    repeat_payload.get("passed") is True,
                    repeat_payload.get("model_spec_sha256")
                    == config.model_spec_digest(),
                    reuse_manifest.get("passed") is True,
                    reuse_manifest.get("corpus_contract_sha256")
                    == config.corpus_digest(),
                    reuse_manifest.get("model_spec_sha256")
                    == config.model_spec_digest(),
                    config.runtime.target.value
                    in reuse_manifest.get("compatible_targets", []),
                )
            )
            for item in reuse_manifest.get("files", []):
                registered = reuse_root / str(item.get("path", ""))
                if (
                    not registered.is_file()
                    or sha256_file(registered) != item.get("sha256")
                    or registered.stat().st_size != int(item.get("bytes", -1))
                ):
                    reusable = False
                    break
            for part in dataset_payload.get("parts", []):
                local_part = (
                    dataset / "canonical_data" / str(part["name"]) / "part-000000.parquet"
                )
                if (
                    not local_part.is_file()
                    or sha256_file(local_part) != part.get("sha256")
                    or local_part.stat().st_size != int(part.get("bytes", -1))
                ):
                    reusable = False
                    break
            if not reusable:
                raise RuntimeError("reused Phase 4 corpus/dataset scope failed closed")
            atomic_json(
                output / "reused_data_receipt.json",
                {
                    "schema_version": "owl.cadc.phase4-reused-data-receipt.v1",
                    "passed": True,
                    "source_manifest": str(reuse_manifest_path),
                    "source_manifest_sha256": sha256_file(reuse_manifest_path),
                    "corpus_contract_sha256": config.corpus_digest(),
                    "model_spec_sha256": config.model_spec_digest(),
                    "target": config.runtime.target.value,
                    "precision": config.runtime.precision,
                    "phase5_locked": True,
                },
            )
            status["reuse_certified_data"] = 0
            atomic_json(output / "command_status.json", status)
        else:
            _run(
                "plan_corpus",
                [
                    python,
                    "scripts/plan_cadc_phase4_corpus.py",
                    "--config",
                    str(config_path),
                    "--output",
                    str(corpus),
                ],
                output=output,
                status=status,
                resume=args.resume,
                required_artifact=plan,
            )
            if config.runtime.require_runtime_decision:
                if args.hourly_cost_usd <= 0.0 or args.remaining_budget_usd <= 0.0:
                    raise ValueError(
                        "runtime decision mode requires positive --hourly-cost-usd "
                        "and --remaining-budget-usd"
                    )
                if not runtime_estimate.is_file():
                    _run(
                        "run_corpus_calibration",
                        [
                            python,
                            "scripts/run_cadc_phase4_corpus.py",
                            "--plan",
                            str(plan),
                            "--engine-root",
                            config.phase3_input.immutable_engine_root,
                            "--phase25-certificate",
                            config.phase3_input.phase25_certificate,
                            "--hardening-receipt",
                            config.phase3_input.phase25_hardening_receipt,
                            "--backend",
                            config.runtime.backend,
                            "--max-concurrent-units",
                            str(config.runtime.corpus_workers),
                            "--aggregate-device-budget-bytes",
                            str(config.runtime.max_device_bytes),
                            "--branch-transfer-mode",
                            config.runtime.corpus_transfer_mode,
                            "--max-units",
                            str(config.runtime.runtime_calibration_units),
                            "--resume",
                        ],
                        output=output,
                        status=status,
                        resume=args.resume,
                        required_artifact=corpus / "corpus_run_status.json",
                    )
                    _run(
                        "estimate_runtime",
                        [
                            python,
                            "scripts/estimate_cadc_phase4_runtime.py",
                            "--plan",
                            str(plan),
                            "--status",
                            str(corpus / "corpus_run_status.json"),
                            "--hourly-cost-usd",
                            str(args.hourly_cost_usd),
                            "--remaining-budget-usd",
                            str(args.remaining_budget_usd),
                            "--post-corpus-reserve-minutes",
                            str(args.post_corpus_reserve_minutes),
                            "--corpus-target-minutes",
                            str(config.runtime.corpus_target_seconds / 60.0),
                            "--total-target-minutes",
                            str(config.runtime.total_target_seconds / 60.0),
                            "--gpu-telemetry",
                            str(output / "gpu_telemetry.csv"),
                            "--gate-timings",
                            str(output / "gate_timings.json"),
                            "--output",
                            str(runtime_estimate),
                        ],
                        output=output,
                        status=status,
                        resume=False,
                        required_artifact=runtime_estimate,
                    )
                decision_path = (
                    Path(args.runtime_decision).resolve()
                    if args.runtime_decision
                    else output / "runtime_decision.json"
                )
                if not decision_path.is_file():
                    atomic_json(
                        runtime_pause,
                        {
                            "schema_version": "owl.cadc.phase4-runtime-pause.v1",
                            "classification": "PAUSED_AWAITING_USER_RUNTIME_DECISION",
                            "passed": True,
                            "failed": False,
                            "estimate": str(runtime_estimate),
                            "estimate_sha256": sha256_file(runtime_estimate),
                            "plan": str(plan),
                            "decision_required": True,
                            "automatic_budget_failure": False,
                        },
                    )
                    print(f"Phase 4 paused for your runtime decision: {runtime_estimate}")
                    return 0
                if not _runtime_decision(decision_path, runtime_estimate, plan):
                    atomic_json(
                        runtime_pause,
                        {
                            "schema_version": "owl.cadc.phase4-runtime-pause.v1",
                            "classification": "PAUSED_BY_USER_RUNTIME_DECISION",
                            "passed": True,
                            "failed": False,
                            "decision": str(decision_path),
                            "decision_sha256": sha256_file(decision_path),
                            "decision_required": False,
                            "automatic_budget_failure": False,
                        },
                    )
                    print(f"Phase 4 paused by your decision: {decision_path}")
                    return 0
            _run(
                "run_corpus",
                [
                    python,
                    "scripts/run_cadc_phase4_corpus.py",
                    "--plan",
                    str(plan),
                    "--engine-root",
                    config.phase3_input.immutable_engine_root,
                    "--phase25-certificate",
                    config.phase3_input.phase25_certificate,
                    "--hardening-receipt",
                    config.phase3_input.phase25_hardening_receipt,
                    "--backend",
                    config.runtime.backend,
                    "--max-concurrent-units",
                    str(config.runtime.corpus_workers),
                    "--aggregate-device-budget-bytes",
                    str(config.runtime.max_device_bytes),
                    "--branch-transfer-mode",
                    config.runtime.corpus_transfer_mode,
                    "--resume",
                ],
                output=output,
                status=status,
                resume=args.resume,
                required_artifact=corpus / "corpus_run_status.json",
            )
            _run(
                "certify_corpus",
                [
                    python,
                    "scripts/certify_cadc_phase4_corpus.py",
                    "--plan",
                    str(plan),
                    "--config",
                    str(config_path),
                    "--output",
                    str(corpus_certificate),
                ],
                output=output,
                status=status,
                resume=args.resume,
                required_artifact=corpus_certificate,
            )
            _run(
                "build_dataset",
                [
                    python,
                    "scripts/build_cadc_phase4_dataset.py",
                    "--config",
                    str(config_path),
                    "--plan",
                    str(plan),
                    "--corpus-certificate",
                    str(corpus_certificate),
                    "--output",
                    str(dataset),
                ],
                output=output,
                status=status,
                resume=args.resume,
                required_artifact=dataset / "dataset_build_receipt.json",
            )
            _run(
                "repeat_pilot",
                [
                    python,
                    "scripts/analyze_cadc_phase4_repeat_pilot.py",
                    "--config",
                    str(config_path),
                    "--dataset",
                    str(dataset),
                    "--output",
                    str(repeat_pilot),
                ],
                output=output,
                status=status,
                resume=args.resume,
                required_artifact=repeat_pilot,
            )
        _run(
            "train",
            [
                python,
                "scripts/train_cadc_phase4.py",
                "--config",
                str(config_path),
                "--dataset",
                str(dataset),
                "--output",
                str(models),
                "--resume",
            ],
            output=output,
            status=status,
            resume=args.resume,
            required_artifact=models / "training_receipt.json",
        )
        _run(
            "calibrate",
            [
                python,
                "scripts/calibrate_cadc_phase4.py",
                "--config",
                str(config_path),
                "--input",
                str(models),
                "--output",
                str(calibration),
            ],
            output=output,
            status=status,
            resume=args.resume,
            required_artifact=calibration / "calibration_receipt.json",
        )
        _run(
            "score_artifacts",
            [
                python,
                "scripts/build_cadc_phase4_scored_artifacts.py",
                "--config",
                str(config_path),
                "--dataset",
                str(dataset),
                "--calibration",
                str(calibration),
                "--output",
                str(scored),
            ],
            output=output,
            status=status,
            resume=args.resume,
            required_artifact=scored / "scored_artifacts_receipt.json",
        )
        _run(
            "evaluate",
            [
                python,
                "scripts/evaluate_cadc_phase4.py",
                "--config",
                str(config_path),
                "--input",
                str(calibration),
                "--dataset",
                str(dataset),
                "--output",
                str(evaluation),
            ],
            output=output,
            status=status,
            resume=args.resume,
            required_artifact=evaluation / "evaluation.json",
        )
        _run(
            "negative_controls",
            [
                python,
                "scripts/run_cadc_phase4_negative_controls.py",
                "--config",
                str(config_path),
                "--input",
                str(calibration),
                "--dataset",
                str(dataset),
                "--output",
                str(negative),
            ],
            output=output,
            status=status,
            resume=args.resume,
            required_artifact=negative,
        )
        _run(
            "casebook",
            [
                python,
                "scripts/build_cadc_phase4_casebook.py",
                "--config",
                str(config_path),
                "--input",
                str(calibration),
                "--dataset",
                str(dataset),
                "--output",
                str(casebook),
            ],
            output=output,
            status=status,
            resume=args.resume,
            required_artifact=casebook / "casebook_manifest.json",
        )
        _run(
            "profile",
            [
                python,
                "scripts/profile_cadc_phase4.py",
                "--config",
                str(config_path),
                "--dataset",
                str(dataset),
                "--output",
                str(performance),
            ],
            output=output,
            status=status,
            resume=args.resume,
            required_artifact=performance,
        )
        _run(
            "hotpath_audit",
            [
                python,
                "scripts/audit_cadc_phase4_hotpaths.py",
                "--output",
                str(hotpath),
            ],
            output=output,
            status=status,
            resume=args.resume,
            required_artifact=hotpath,
        )
    except Exception:
        atomic_json(prior_status, status)
        subprocess.run(cert_command, cwd=ROOT, check=False)
        raise
    if _release_hash(ROOT) != current_source:
        raise RuntimeError(
            "Phase 4 source changed during acceptance; refusing to certify stale evidence"
        )
    atomic_json(prior_status, status)
    _run(
        "certifier",
        cert_command,
        output=output,
        status=status,
        resume=args.resume,
        required_artifact=certificate,
    )
    atomic_json(output / "command_status.final.json", status)
    telemetry.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
