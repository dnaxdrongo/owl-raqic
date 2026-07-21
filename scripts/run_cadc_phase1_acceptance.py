#!/usr/bin/env python3
# ruff: noqa: E402
"""Run and validate a bounded 25-tick factual-recorder candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.dataset as pads

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from owl.core.actions import Action
from owl.core.config import load_config
from owl.gpu.run_context import PersistentOWLDeviceRun
from owl.record.cadc_schema import CADC_ACTION_COUNT, ContributionCode
from owl.record.replay_recorder import ReplayRecorder
from owl.viz.visual_snapshot import snapshot_from_device_state


def _host(value: Any) -> np.ndarray:
    if hasattr(value, "get"):
        return np.asarray(value.get())
    return np.asarray(value)


def _state_hash(run: PersistentOWLDeviceRun) -> str:
    digest = hashlib.sha256()
    for name in sorted(run.ds.arrays):
        value = run.ds.arrays[name]
        if not hasattr(value, "shape"):
            continue
        array = np.ascontiguousarray(_host(value))
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _cuda_device_metadata(backend: str) -> dict[str, Any] | None:
    """Return verified CUDA identity information for target candidates."""
    if backend != "cupy":
        return None
    import cupy as cp

    device_id = int(cp.cuda.runtime.getDevice())
    properties = cp.cuda.runtime.getDeviceProperties(device_id)
    raw_name = properties.get("name", b"")
    name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)
    free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
    return {
        "device_id": device_id,
        "name": name,
        "compute_capability": (
            f"{int(properties.get('major', -1))}.{int(properties.get('minor', -1))}"
        ),
        "multiprocessor_count": int(properties.get("multiProcessorCount", -1)),
        "total_global_memory_bytes": int(properties.get("totalGlobalMem", total_bytes)),
        "free_memory_bytes_after_acceptance": int(free_bytes),
        "runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
        "driver_version": int(cp.cuda.runtime.driverGetVersion()),
        "cupy_version": str(cp.__version__),
    }


def _compare_runs(control: PersistentOWLDeviceRun, evidence: PersistentOWLDeviceRun) -> None:
    common = sorted(set(control.ds.arrays) & set(evidence.ds.arrays))
    for name in common:
        left = _host(control.ds.arrays[name])
        right = _host(evidence.ds.arrays[name])
        if not np.array_equal(left, right, equal_nan=True):
            raise AssertionError(f"recorder-on factual mismatch at tick {control.ds.tick}: {name}")


def _table(root: Path, name: str) -> Any:
    return pads.dataset(root / f"{name}.parquet", format="parquet").to_table()


def _commit_telemetry(root: Path, expected_ticks: int) -> dict[str, Any]:
    commits = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((root / "commits").glob("tick_*.json"))
    ]
    if len(commits) != expected_ticks:
        raise AssertionError(f"CADC commit count {len(commits)} != {expected_ticks}")
    if any(int(item["event_overflow"]) != 0 for item in commits):
        raise AssertionError("CADC event buffer overflowed")
    return {
        "packet_count": len(commits),
        "transfer_bytes_total": int(sum(int(item["transfer_bytes"]) for item in commits)),
        "transfer_count_total": int(sum(int(item["transfer_count"]) for item in commits)),
        "packet_bytes_max": int(max(int(item["transfer_bytes"]) for item in commits)),
        "event_overflow_total": 0,
        "source_backends": sorted({str(item["source_backend"]) for item in commits}),
    }


def _validate_tables(root: Path, expected_decisions: int) -> dict[str, Any]:
    decisions = _table(root, "decisions")
    agent = _table(root, "agent_context")
    oracle = _table(root, "oracle_context")
    candidates = _table(root, "candidates")
    execution = _table(root, "execution")
    events = _table(root, "events")
    contributions = _table(root, "contributions")
    information = _table(root, "information")
    followups = _table(root, "information_followups")
    dense = _table(root, "dense_context")

    decision_id = decisions.column("decision_sequence").to_numpy()
    if decisions.num_rows != expected_decisions:
        raise AssertionError(f"decision total {decisions.num_rows} != {expected_decisions}")
    if np.unique(decision_id).size != decisions.num_rows:
        raise AssertionError("duplicate decision_sequence")
    for name, table in (
        ("agent_context", agent),
        ("oracle_context", oracle),
        ("execution", execution),
        ("dense_context", dense),
    ):
        if table.num_rows != decisions.num_rows:
            raise AssertionError(f"{name} does not have one row per decision")
    if candidates.num_rows != decisions.num_rows * CADC_ACTION_COUNT:
        raise AssertionError("candidate table is not exactly 22 rows per decision")
    candidate_id = candidates.column("candidate_sequence").to_numpy()
    if np.unique(candidate_id).size != candidates.num_rows:
        raise AssertionError("duplicate candidate_sequence")
    actions = candidates.column("action_index").to_numpy().reshape(-1, CADC_ACTION_COUNT)
    if not np.all(actions == np.arange(CADC_ACTION_COUNT, dtype=np.int16)):
        raise AssertionError("candidate action order differs from the immutable action axis")
    selected = decisions.column("selected_action").to_numpy()
    if not np.array_equal(selected, execution.column("selected_action").to_numpy()):
        raise AssertionError("selected action does not join decisions to execution")
    selected_candidates = actions == selected[:, None]
    if not np.all(np.sum(selected_candidates, axis=1) == 1):
        raise AssertionError("decision does not resolve to exactly one selected candidate")
    legal = candidates.column("policy_legal").to_numpy()
    executable = candidates.column("prechoice_executable").to_numpy()
    if not np.any(legal != executable):
        raise AssertionError("acceptance workload did not exercise legal/executable separation")
    event_sequence = events.column("event_sequence").to_numpy()
    if np.unique(event_sequence).size != events.num_rows:
        raise AssertionError("duplicate event_sequence")
    if events.num_rows and not np.all(
        np.isin(events.column("decision_sequence").to_numpy(), decision_id)
    ):
        raise AssertionError("event actor has no factual decision")
    contribution_id = contributions.column("contribution_sequence").to_numpy()
    if np.unique(contribution_id).size != contributions.num_rows:
        raise AssertionError("duplicate contribution_sequence")
    residual = contributions.column("contribution_code").to_numpy() == int(
        ContributionCode.RESIDUAL
    )
    if np.count_nonzero(residual) != decisions.num_rows:
        raise AssertionError("each decision must have one named reconciliation residual")
    contribution_codes = np.unique(contributions.column("contribution_code").to_numpy())
    contribution_count = int(contribution_codes.size)
    maximum_reconciliation_error = 0.0
    for name in (
        "health",
        "resource",
        "food",
        "toxin",
        "waste",
        "integration",
        "boundary",
        "signal_emission",
    ):
        delta = contributions.column(f"delta_{name}").to_numpy().reshape(
            -1, contribution_count
        )
        start = contributions.column(f"start_{name}").to_numpy().reshape(
            -1, contribution_count
        )[:, 0]
        end = contributions.column(f"end_{name}").to_numpy().reshape(
            -1, contribution_count
        )[:, 0]
        error = np.max(np.abs(delta.sum(axis=1) - (end - start)), initial=0.0)
        maximum_reconciliation_error = max(maximum_reconciliation_error, float(error))
    if maximum_reconciliation_error > 1e-6:
        raise AssertionError(
            f"contribution reconciliation error {maximum_reconciliation_error} exceeds 1e-6"
        )
    if not np.array_equal(
        decisions.column("dense_context_ref").to_numpy(),
        dense.column("dense_context_id").to_numpy(),
    ):
        raise AssertionError("orphan or reordered dense-context reference")
    source_information = information.column("decision_sequence").to_numpy()
    followup_source = followups.column("source_decision_sequence").to_numpy()
    if followups.num_rows != information.num_rows or not np.all(
        np.isin(source_information, followup_source)
    ):
        raise AssertionError("information record lacks exactly one bounded follow-up")

    action_directions_root = root / "action_directions.parquet"
    action_direction_rows = 0
    if action_directions_root.exists():
        action_directions = _table(root, "action_directions")
        action_direction_rows = int(action_directions.num_rows)
        if action_direction_rows != decisions.num_rows * 16:
            raise AssertionError("action-direction table is not exactly 16 rows per decision")

        # owl.cadc.factual.v2 action-contract joins. A failed execution has no
        # realized action and therefore must retain ABSENT_INT (-1); it must
        # never be decoded as the final action-axis entry.
        attempted = execution.column("attempted_action").to_numpy()
        realized = execution.column("realized_action").to_numpy()
        success = execution.column("execution_success").to_numpy()
        compiled = execution.column("compiled_execution_action").to_numpy()
        if not np.array_equal(attempted, selected):
            raise AssertionError("attempted action does not exactly join selected action")
        if not np.array_equal(realized[success], selected[success]):
            raise AssertionError("successful realized action differs from selected identity")
        if np.any(realized[~success] != -1):
            raise AssertionError("failed execution does not use ABSENT_INT realized action")
        selected_counts: dict[str, int] = {}
        successful_counts: dict[str, int] = {}
        for action in (Action.SENSE, Action.FLEE, Action.PURSUE):
            action_mask = selected == int(action)
            selected_counts[action.name] = int(np.count_nonzero(action_mask))
            successful_counts[action.name] = int(np.count_nonzero(action_mask & success))
            if selected_counts[action.name] == 0:
                raise AssertionError(f"acceptance workload did not select {action.name}")
        sense = selected == int(Action.SENSE)
        directional = (selected == int(Action.FLEE)) | (selected == int(Action.PURSUE))
        if np.any(compiled[sense] != int(Action.SENSE)):
            raise AssertionError("SENSE compiled identity is not authoritative SENSE")
        if np.any(
            (compiled[directional] < int(Action.MOVE_N))
            | (compiled[directional] > int(Action.MOVE_SW))
        ):
            raise AssertionError(
                "FLEE/PURSUE did not compile to an immutable movement primitive"
            )

        info_kind = information.column("information_kind").to_numpy()
        info_success = information.column("information_execution_success").to_numpy()
        sense_information = info_kind == int(Action.SENSE)
        communicate_information = info_kind == int(Action.COMMUNICATE)
        if int(np.count_nonzero(sense_information)) != selected_counts[Action.SENSE.name]:
            raise AssertionError("SENSE information rows do not join selected SENSE decisions")
        if not np.all(info_success[sense_information]):
            raise AssertionError(
                "successful active SENSE is missing information-state success"
            )
        if np.any(info_success[communicate_information]):
            raise AssertionError(
                "active-SENSE execution evidence leaked into COMMUNICATE rows"
            )
    else:
        selected_counts = {}
        successful_counts = {}

    return {
        "decisions": decisions.num_rows,
        "agent_context": agent.num_rows,
        "oracle_context": oracle.num_rows,
        "candidates": candidates.num_rows,
        "execution": execution.num_rows,
        "events": events.num_rows,
        "contributions": contributions.num_rows,
        "information": information.num_rows,
        "information_followups": followups.num_rows,
        "dense_context": dense.num_rows,
        "action_directions": action_direction_rows,
        "selected_action_contracts": selected_counts,
        "successful_action_contracts": successful_counts,
        "policy_legal_true": int(np.count_nonzero(legal)),
        "prechoice_executable_true": int(np.count_nonzero(executable)),
        "max_contribution_reconciliation_error_abs": maximum_reconciliation_error,
    }


def run_acceptance(args: argparse.Namespace) -> dict[str, Any]:
    from owl.experiments.controller import _release_hash

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    base = load_config(args.config)
    base.world.max_steps = int(args.ticks)
    base.debug.assert_invariants = True
    control_cfg = base.model_copy(deep=True)
    control_cfg.recording.cadc.enabled = False
    evidence_cfg = base.model_copy(deep=True)
    evidence_cfg.recording.cadc.enabled = True
    backend = str(args.backend)
    control = PersistentOWLDeviceRun.from_config(
        control_cfg,
        force_backend=backend,
        output_root=output / "control_scientific",
    )
    evidence = PersistentOWLDeviceRun.from_config(
        evidence_cfg,
        force_backend=backend,
        output_root=output / "evidence_scientific",
    )
    source_sha256 = _release_hash(ROOT)
    config_sha256 = hashlib.sha256(Path(args.config).read_bytes()).hexdigest()
    recorder = ReplayRecorder(
        output / "bundle",
        run_id="cadc-phase1-acceptance",
        condition="phase_interference",
        seed=int(evidence_cfg.world.seed),
        requested_ticks=int(args.ticks),
        recording_tier="analysis_full",
        action_names=[action.name for action in Action],
        source_sha256=source_sha256,
        config_sha256=config_sha256,
        cadc_config=evidence_cfg.recording.cadc,
    )
    control_seconds = 0.0
    evidence_seconds = 0.0
    record_seconds = 0.0
    expected_decisions = 0
    max_residual = 0.0
    state_hashes: list[str] = []
    try:
        for _ in range(int(args.ticks)):
            started = time.perf_counter()
            control.step()
            control_seconds += time.perf_counter() - started
            started = time.perf_counter()
            diagnostics = evidence.step()
            evidence_seconds += time.perf_counter() - started
            _compare_runs(control, evidence)
            buffer = evidence.cadc_buffer
            living = _host(buffer.arrays["pre_alive"]) > 0
            expected_decisions += int(np.count_nonzero(living))
            residual = _host(buffer.arrays["contribution_delta"])[-1]
            max_residual = max(max_residual, float(np.max(np.abs(residual[living]))))
            snapshot = snapshot_from_device_state(evidence.ds)
            started = time.perf_counter()
            recorder.record_device(evidence.ds, snapshot, diagnostics=diagnostics)
            record_seconds += time.perf_counter() - started
            state_hashes.append(_state_hash(evidence))
        manifest = recorder.close()
        final_control_hash = _state_hash(control)
        final_evidence_hash = _state_hash(evidence)
        if final_control_hash != final_evidence_hash:
            raise AssertionError("final recorder-on/off source-state hashes differ")
        factual_version = str(evidence.cadc_buffer.schema_version)
        cadc_root = output / "bundle" / "analysis" / (
            "cadc_v2" if factual_version == "owl.cadc.factual.v2" else "cadc_v1"
        )
        rows = _validate_tables(cadc_root, expected_decisions)
        transfer_telemetry = _commit_telemetry(cadc_root, int(args.ticks))
        size_bytes = sum(
            path.stat().st_size for path in (output / "bundle").rglob("*") if path.is_file()
        )
        packet_bytes = int(evidence.cadc_buffer.nbytes)
        payload = {
            "schema_version": "owl.cadc.phase1-local-acceptance.v1",
            "passed": True,
            "scope": "local_numpy" if backend == "numpy" else f"target_{backend}",
            "ticks": int(args.ticks),
            "backend": str(evidence.ds.backend.name),
            "cuda_device": _cuda_device_metadata(backend),
            "python": platform.python_version(),
            "world_shape": list(evidence.cadc_buffer.world_shape),
            "schema_digest": str(evidence.cadc_buffer.schema_digest),
            "factual_schema_version": factual_version,
            "source_sha256": source_sha256,
            "config_sha256": config_sha256,
            "action_order": [action.name for action in Action],
            "rows": rows,
            "packet_transfer_telemetry": transfer_telemetry,
            "recorder_on_off_exact": True,
            "final_control_state_hash": final_control_hash,
            "final_evidence_state_hash": final_evidence_hash,
            "tick_state_hashes": state_hashes,
            "max_named_residual_abs": max_residual,
            "device_buffer_bytes": packet_bytes,
            "configured_device_buffer_limit": int(
                evidence_cfg.recording.cadc.max_device_buffer_bytes
            ),
            "bundle_bytes": size_bytes,
            "timing_seconds": {
                "control_step_total": control_seconds,
                "instrumented_step_total": evidence_seconds,
                "record_total": record_seconds,
                "instrumented_step_overhead_fraction": (
                    evidence_seconds / control_seconds - 1.0 if control_seconds else None
                ),
            },
            "replay_completed_ticks": int(manifest.completed_ticks),
            "target_gpu_validation": "not_run" if backend == "numpy" else "candidate_only",
        }
        certificate = output / "phase1_local_acceptance.json"
        certificate.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if bool(evidence_cfg.action_transitions.enabled):
            (output / "phase25_local_acceptance.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        return payload
    finally:
        if not recorder._closed and recorder._ticks:
            recorder.close(state="FAILED_PARTIAL", failure="acceptance failure")
        control.close(checkpoint=False)
        evidence.close(checkpoint=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cadc_phase1_exact_25tick.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--ticks", type=int, default=25)
    parser.add_argument("--backend", choices=("numpy", "cupy"), default="numpy")
    args = parser.parse_args()
    payload = run_acceptance(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
