#!/usr/bin/env python3
"""Run the factual control and isolated counterfactual acceptance checks."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import shutil
import sys
import traceback
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from owl.core.actions import Action  # noqa: E402
from owl.core.config import SimulationConfig, load_config  # noqa: E402
from owl.core.init import initialize_world  # noqa: E402
from owl.counterfactual.scheduler import CounterfactualScheduler  # noqa: E402
from owl.counterfactual.schema import (  # noqa: E402
    COUNTERFACTUAL_SCHEMA_DIGEST,
    COUNTERFACTUAL_SCHEMA_VERSION,
    BranchStatus,
)
from owl.counterfactual.source import CounterfactualSourceCollector  # noqa: E402
from owl.counterfactual.staging import stage_counterfactual_result  # noqa: E402
from owl.counterfactual.state_hash import compare_state_science, hash_state  # noqa: E402
from owl.counterfactual.writer import CounterfactualWriter  # noqa: E402
from owl.experiments.controller import _release_hash  # noqa: E402
from owl.gpu.memory_model import build_counterfactual_memory_plan  # noqa: E402
from owl.gpu.run_context import PersistentOWLDeviceRun  # noqa: E402
from owl.record.cadc_schema import (  # noqa: E402
    CADC_ACTION_TRANSITION_SCHEMA_DIGEST,
    CADC_ACTION_TRANSITION_SCHEMA_VERSION,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def record_progress(output: Path, stage: str) -> None:
    atomic_json(
        output / "acceptance_progress.json",
        {
            "schema_version": "owl.cadc.phase3-acceptance-progress.v1",
            "stage": stage,
            "pid": os.getpid(),
        },
    )


def verify_phase25_gate(certificate_path: Path, hardening_path: Path) -> dict[str, Any]:
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    hardening = json.loads(hardening_path.read_text(encoding="utf-8"))
    failures = []
    if certificate.get("schema_version") != "owl.phase2.5.target-gpu-certificate.v1":
        failures.append("wrong Phase 2.5 certificate schema")
    if certificate.get("passed") is not True:
        failures.append("Phase 2.5 target certificate did not pass")
    if certificate.get("classification") != "PHASE2_5_TARGET_GPU_CERTIFIED":
        failures.append("wrong Phase 2.5 classification")
    if certificate.get("phase3_unlocked") is not True:
        failures.append("Phase 2.5 certificate does not unlock Phase 3")
    if certificate.get("phase4_unlocked") is not False:
        failures.append("Phase 2.5 certificate must not unlock Phase 4")
    if certificate.get("cadc_schema_digest") != CADC_ACTION_TRANSITION_SCHEMA_DIGEST:
        failures.append("factual v2 digest mismatch")
    source = str(certificate.get("source_sha256", ""))
    if hardening.get("passed") is not True:
        failures.append("Phase 2.5 hardening receipt did not pass")
    if hardening.get("base_certified_source_sha256") != source:
        failures.append("hardening receipt is not derived from certified source")
    if failures:
        raise RuntimeError("; ".join(failures))
    return {
        "certificate": certificate,
        "certificate_sha256": sha256_file(certificate_path),
        "hardening": hardening,
        "hardening_sha256": sha256_file(hardening_path),
    }


def synthetic_initial_state(cfg: SimulationConfig) -> Any:
    state = initialize_world(cfg, np.random.default_rng(int(cfg.world.seed)))
    for name in (
        "health",
        "resource",
        "boundary",
        "integration",
        "food",
        "toxin",
        "waste",
        "mobility",
        "predation",
        "aggression",
        "grazing",
        "cooperation",
        "curiosity",
        "reproduction_rate",
        "emit_strength",
        "emit_efficiency",
        "coupling_strength",
    ):
        value = getattr(state, name, None)
        if isinstance(value, np.ndarray):
            value.fill(0)
    state.occupancy.fill(-1)
    state.obstacle.fill(False)
    state.readout.fill(int(Action.REST))
    focal = (5, 5)
    for y, x, ow_id in ((5, 5, 55), (3, 7, 37), (4, 5, 45)):
        state.health[y, x] = 1.0
        state.resource[y, x] = 1.0
        state.boundary[y, x] = 1.0
        state.integration[y, x] = 1.0
        state.occupancy[y, x] = ow_id
        state.mobility[y, x] = 1.0
    y, x = focal
    state.predation[y, x] = 1.0
    state.aggression[y, x] = 1.0
    state.grazing[y, x] = 1.0
    state.cooperation[y, x] = 1.0
    state.curiosity[y, x] = 1.0
    state.reproduction_rate[y, x] = 1.0
    state.emit_strength[y, x] = 1.0
    state.emit_efficiency[y, x] = 1.0
    state.coupling_strength[y, x] = 1.0
    state.food[y, x] = 1.0
    state.toxin[5, 3] = 1.0
    state.tick = 0
    return state


def device_metadata(run: PersistentOWLDeviceRun) -> dict[str, Any]:
    payload = {
        "backend": run.ds.backend.name,
        "is_gpu": run.ds.is_gpu,
        "backend_info": run.ds.backend.info,
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    if run.ds.is_gpu:
        cp = run.ds.xp
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props["name"]
        if isinstance(name, bytes):
            name = name.decode()
        payload["cuda_device"] = {
            "name": str(name),
            "device_id": int(cp.cuda.Device().id),
            "compute_capability": f"{props['major']}.{props['minor']}",
            "total_global_memory_bytes": int(props["totalGlobalMem"]),
            "multiprocessor_count": int(props["multiProcessorCount"]),
            "runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
            "driver_version": int(cp.cuda.runtime.driverGetVersion()),
            "cupy_version": str(cp.__version__),
        }
    return payload


def run_acceptance(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    record_progress(output, "verify_phase25_gate")
    phase25_certificate_path = Path(args.phase25_certificate).resolve()
    hardening_receipt_path = Path(args.hardening_receipt).resolve()
    gate = verify_phase25_gate(phase25_certificate_path, hardening_receipt_path)
    gate_root = output / "phase25_gate"
    gate_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(phase25_certificate_path, gate_root / "phase25_target_gpu_certificate.json")
    shutil.copy2(hardening_receipt_path, gate_root / "hardening_receipt.json")
    cfg = load_config(args.config)
    data = cfg.model_dump(mode="json")
    data["world"]["max_steps"] = int(args.ticks)
    data["counterfactual"]["backend"] = args.backend
    if args.backend == "numpy":
        data["counterfactual"]["branch_execution_mode"] = "segmented"
        data["raqic"]["full_gpu_strict"] = False
        data["raqic"]["fallback_on_backend_error"] = True
    cfg = SimulationConfig.model_validate(data)
    phase3_source = _release_hash(ROOT)
    config_sha256 = sha256_file(Path(args.config))
    initial = synthetic_initial_state(cfg)

    record_progress(output, "control_initialize")
    control_cfg = cfg.model_copy(deep=True)
    control_cfg.counterfactual.enabled = False
    control_cfg.recording.cadc.enabled = False
    control_started = perf_counter()
    control = PersistentOWLDeviceRun.from_config(
        control_cfg,
        initial_state=copy.deepcopy(initial),
        force_backend=args.backend,
        output_root=output / "factual_control",
    )
    try:
        record_progress(output, "control_ticks")
        for _ in range(int(args.ticks)):
            control.step()
        record_progress(output, "control_state_hash")
        control_hash = hash_state(control.ds)
    except Exception:
        control.close(checkpoint=False)
        raise
    finally:
        control_seconds = perf_counter() - control_started

    collector = CounterfactualSourceCollector(
        cfg,
        str(gate["hardening"]["hardened_source_sha256"]),
        run_id="phase3-acceptance",
        condition="synthetic-action-coverage",
    )
    factual_started = perf_counter()
    record_progress(output, "factual_initialize")
    try:
        factual = PersistentOWLDeviceRun.from_config(
            cfg,
            initial_state=copy.deepcopy(initial),
            force_backend=args.backend,
            output_root=output / "factual_observed",
            counterfactual_observer=collector,
        )
    except Exception:
        control.close(checkpoint=False)
        raise
    try:
        record_progress(output, "factual_ticks")
        for _ in range(int(args.ticks)):
            factual.step()
        factual_seconds = perf_counter() - factual_started
        record_progress(output, "factual_state_hash")
        factual_hash_before_branches = hash_state(factual.ds)
        record_progress(output, "factual_recovery_comparison")
        factual_recovery = compare_state_science(factual.ds, control.ds)
        if not factual_recovery.passed:
            raise AssertionError(f"observer changed factual science: {factual_recovery}")
        if not collector.sources:
            raise RuntimeError("no Phase 3 source state was selected")
        free_device = None
        if factual.ds.is_gpu:
            free_device = int(factual.ds.xp.cuda.runtime.memGetInfo()[0])
        record_progress(output, "counterfactual_memory_plan")
        memory_plan = build_counterfactual_memory_plan(
            factual.ds,
            cfg,
            scratch_bytes=int(factual.scratch.spec_bytes()),
            free_device_bytes=free_device,
        )
        if not memory_plan.passed:
            raise MemoryError("counterfactual memory plan cannot fit one branch")
        scheduler = CounterfactualScheduler(
            factual,
            cfg,
            active_branch_limit=int(memory_plan.max_active_branches),
        )
        branch_started = perf_counter()
        record_progress(output, "counterfactual_branches")
        results = [scheduler.run_source(source) for source in collector.sources]
        branch_seconds = perf_counter() - branch_started
        record_progress(output, "factual_nonmutation_hash")
        factual_hash_after_branches = hash_state(factual.ds)
        if factual_hash_after_branches.root != factual_hash_before_branches.root:
            raise AssertionError("counterfactual branches mutated factual state")
        record_progress(output, "counterfactual_branch_validation")
        failed_branches = [
            branch
            for result in results
            for branch in result.branches
            if branch.status != BranchStatus.COMPLETED
        ]
        failures = [branch.failure for branch in failed_branches]
        if failed_branches:
            diagnostics = {
                "schema_version": "owl.cadc.phase3-branch-failures.v1",
                "failure_count": len(failed_branches),
                "unique_failures": sorted({str(value) for value in failures}),
                "branches": [
                    {
                        "branch_id": branch.branch_id,
                        "source_decision_id": branch.source_decision_id,
                        "repeat_index": branch.repeat_index,
                        "forced_action": branch.forced_action,
                        "forced_action_name": Action(branch.forced_action).name,
                        "selected_anchor": branch.anchor,
                        "failure": branch.failure,
                        "traceback": list(branch.failure_traceback),
                    }
                    for branch in failed_branches
                ],
            }
            atomic_json(output / "branch_failures.json", diagnostics)
            raise RuntimeError(
                f"{len(failed_branches)} counterfactual branches failed; "
                f"first={failed_branches[0].failure}; see branch_failures.json"
            )
        record_progress(output, "columnar_staging")
        packets = tuple(
            packet
            for source, result in zip(collector.sources, results, strict=True)
            for packet in stage_counterfactual_result(source, result)
        )
        packet_bytes = sum(packet.nbytes for packet in packets)
        writer_started = perf_counter()
        record_progress(output, "parquet_writer")
        writer = CounterfactualWriter(
            output / "counterfactual",
            source_sha256=phase3_source,
            phase25_certificate_sha256=str(gate["certificate_sha256"]),
            factual_v2_digest=CADC_ACTION_TRANSITION_SCHEMA_DIGEST,
            max_packet_bytes=int(cfg.counterfactual.max_packet_bytes),
            max_pending_bytes=int(cfg.counterfactual.max_pending_bytes),
            row_group_rows=int(cfg.counterfactual.parquet_row_group_rows),
        )
        receipts = writer.write_packets(packets)
        writer_seconds = perf_counter() - writer_started
        actions = Counter(
            Action(branch.forced_action).name
            for result in results
            for branch in result.branches
            if not branch.anchor and branch.status == BranchStatus.COMPLETED
        )
        anchors = [branch for result in results for branch in result.branches if branch.anchor]
        anchor_pass = bool(anchors) and all(
            all(branch.anchor_matches.values()) for branch in anchors
        )
        overflow = sum(
            int(np.asarray(packet.event_overflow)[0])
            for result in results
            for branch in result.branches
            for packet in branch.evidence
        )
        branch_ticks = sum(len(branch.evidence) for result in results for branch in result.branches)
        branch_count = sum(len(result.branches) for result in results)
        pair_count = sum(len(result.pairs) for result in results)
        nonexec_count = sum(len(result.nonexecutable) for result in results)
        manifest = {
            "schema_version": "owl.cadc.phase3-acceptance.v1",
            "counterfactual_schema_version": COUNTERFACTUAL_SCHEMA_VERSION,
            "counterfactual_schema_digest": COUNTERFACTUAL_SCHEMA_DIGEST,
            "project_version": "0.9.9",
            "phase25": {
                "certified_source_sha256": gate["certificate"]["source_sha256"],
                "certificate_sha256": gate["certificate_sha256"],
                "classification": gate["certificate"]["classification"],
                "hardening_source_sha256": gate["hardening"]["hardened_source_sha256"],
                "hardening_receipt_sha256": gate["hardening_sha256"],
                "action_contract": "owl.action-transitions.v1",
                "factual_schema": CADC_ACTION_TRANSITION_SCHEMA_VERSION,
                "factual_schema_digest": CADC_ACTION_TRANSITION_SCHEMA_DIGEST,
            },
            "phase3_source_sha256": phase3_source,
            "config_sha256": config_sha256,
            "resolved_counterfactual_config": cfg.counterfactual.model_dump(mode="json"),
            "backend": args.backend,
            "requested_ticks": int(args.ticks),
            "device": device_metadata(factual),
            "factual_recovery": {
                "passed": factual_recovery.passed,
                "control_root": control_hash.root,
                "observed_root": factual_hash_before_branches.root,
                "categorical_failures": list(factual_recovery.categorical_failures),
                "floating_failures": list(factual_recovery.floating_failures),
                "max_abs_difference": factual_recovery.max_abs_difference,
            },
            "factual_nonmutation": {
                "passed": factual_hash_before_branches.root == factual_hash_after_branches.root,
                "before_root": factual_hash_before_branches.root,
                "after_root": factual_hash_after_branches.root,
            },
            "source_counts": {
                "source_states": len(collector.sources),
                "source_decisions": sum(item.decisions.count for item in collector.sources),
                "candidate_rows_per_decision": 22,
                "direction_rows_per_decision": 16,
                "clone_fields": len(collector.sources[0].state.manifest.fields),
                "clone_array_bytes": collector.sources[0].state.nbytes,
                "clone_scalar_names": list(collector.sources[0].state.manifest.scalar_names),
                "clone_metadata_names": list(collector.sources[0].state.manifest.metadata_names),
                "pointer_isolation_validated": True,
            },
            "branch_counts": {
                "branches": branch_count,
                "branch_ticks": branch_ticks,
                "pairs": pair_count,
                "nonexecutable": nonexec_count,
                "by_action": dict(sorted(actions.items())),
            },
            "anchor": {
                "count": len(anchors),
                "passed": anchor_pass,
                "tolerance_contract": {
                    "float32_atol": 9.5367431640625e-7,
                    "float64_atol": 1e-10,
                },
                "exact_hash_matches": {
                    branch.branch_id: branch.anchor_exact_hash_matches for branch in anchors
                },
            },
            "event_overflow": overflow,
            "memory_plan": memory_plan.to_dict(),
            "transfer": {
                "factual": factual.transfer_ledger.to_dict(),
                "branch": scheduler.transfer_ledger.to_dict(),
                "packet_d2h_bytes": packet_bytes,
            },
            "parquet": {
                "parts": [asdict(receipt) for receipt in receipts],
                "rows": {receipt.table_name: receipt.rows for receipt in receipts},
            },
            "performance": {
                "control_seconds": control_seconds,
                "factual_seconds": factual_seconds,
                "branch_seconds": branch_seconds,
                "writer_seconds": writer_seconds,
                "branches_per_second": branch_count / max(branch_seconds, 1e-12),
                "branch_ticks_per_second": branch_ticks / max(branch_seconds, 1e-12),
                "source_copy_bytes": collector.source_copy_bytes,
                "source_copy_count": collector.source_copy_count,
                "packet_bytes": packet_bytes,
                "execution_strategy": (
                    "multi_stream_whole_array"
                    if factual.ds.is_gpu and scheduler.last_worker_count > 1
                    else "single_lane_whole_array"
                ),
                "worker_count": scheduler.last_worker_count,
            },
            "qiskit_aer_gpu": {
                "exercised": factual.per_ow_qiskit is not None,
                "fail_closed": factual.per_ow_qiskit is None,
            },
            "passed": bool(
                factual_recovery.passed
                and factual_hash_before_branches.root == factual_hash_after_branches.root
                and anchor_pass
                and not overflow
                and memory_plan.passed
                and not failures
            ),
            "phase4_unlocked": False,
        }
        atomic_json(output / "phase3_acceptance_manifest.json", manifest)
        atomic_json(output / "command_status.json", {"acceptance_runner": 0})
        record_progress(output, "completed")
        return manifest
    finally:
        factual.close(checkpoint=False)
        control.close(checkpoint=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ticks", type=int, default=25)
    parser.add_argument("--backend", choices=("numpy", "cupy"), required=True)
    parser.add_argument("--phase25-certificate", required=True)
    parser.add_argument("--hardening-receipt", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        manifest = run_acceptance(args)
    except Exception as exc:
        output = Path(args.output).resolve()
        progress_path = output / "acceptance_progress.json"
        progress = (
            json.loads(progress_path.read_text(encoding="utf-8"))
            if progress_path.is_file()
            else {"stage": "bootstrap"}
        )
        certificate_path = Path(args.phase25_certificate).resolve()
        hardening_path = Path(args.hardening_receipt).resolve()
        atomic_json(
            output / "phase3_acceptance_manifest.json",
            {
                "schema_version": "owl.cadc.phase3-acceptance.v1",
                "project_version": "0.9.9",
                "phase3_source_sha256": _release_hash(ROOT),
                "backend": args.backend,
                "requested_ticks": int(args.ticks),
                "config_sha256": (
                    sha256_file(Path(args.config).resolve())
                    if Path(args.config).is_file()
                    else None
                ),
                "phase25_artifacts": {
                    "certificate_sha256": (
                        sha256_file(certificate_path) if certificate_path.is_file() else None
                    ),
                    "hardening_receipt_sha256": (
                        sha256_file(hardening_path) if hardening_path.is_file() else None
                    ),
                },
                "failure_stage": progress.get("stage", "unknown"),
                "failure": {
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc().splitlines(),
                },
                "passed": False,
                "phase4_unlocked": False,
                "failures": [f"{type(exc).__name__}: {exc}"],
            },
        )
        atomic_json(output / "command_status.json", {"acceptance_runner": 1})
        raise
    print(json.dumps({"passed": manifest["passed"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
