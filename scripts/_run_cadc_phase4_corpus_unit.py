#!/usr/bin/env python3
"""Run one corpus unit with the certified counterfactual engine on the target GPU."""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import os
import platform
import sys
import traceback
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np


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


@dataclass
class DelayedCollector:
    """Expose a certified collector only at the pre-registered source tick."""

    collector: Any
    source_tick: int

    def capture(self, run: Any, decision_event: Any) -> None:
        if int(run.ds.tick) == self.source_tick:
            self.collector.capture(run, decision_event)

    def after_postdecision(self, run: Any) -> None:
        if self.collector.sources:
            self.collector.after_postdecision(run)

    def close(self) -> None:
        self.collector.close()


def _slice_table_packet(packet: Any, start: int, stop: int) -> Any:
    """Return a row-preserving packet slice without changing column values."""

    rows = int(packet.rows)
    if not 0 <= start < stop <= rows:
        raise ValueError(f"invalid packet slice [{start}:{stop}] for {rows} rows")
    columns: dict[str, Any] = {}
    for name, value in packet.columns.items():
        if len(value) != rows:
            raise ValueError(
                f"{packet.table_name}.{name} has {len(value)} rows; expected {rows}"
            )
        columns[name] = value[start:stop]
    return type(packet)(packet.table_name, columns)


def _bounded_table_packets(
    packets: Iterable[Any],
    *,
    max_packet_bytes: int,
    max_pending_bytes: int,
) -> Iterator[Any]:
    """Split oversized host packets into deterministic contiguous row ranges.

    The immutable counterfactual writer remains unchanged and continues to enforce
    both limits. This CADC-MORE 2 adapter ensures that a scientifically valid,
    high-density event or contribution table reaches it as several bounded
    parts. Concatenating the emitted parts reproduces the original columns and
    row order exactly.
    """

    byte_limit = min(int(max_packet_bytes), int(max_pending_bytes))
    if byte_limit <= 0:
        raise ValueError("counterfactual packet byte limits must be positive")
    for packet in packets:
        packet_bytes = int(packet.nbytes)
        rows = int(packet.rows)
        if packet_bytes <= byte_limit:
            yield packet
            continue
        if rows <= 0:
            raise MemoryError(
                f"{packet.table_name} empty packet reports {packet_bytes:,} bytes"
            )

        # Start below the proportional limit. Variable-width string columns
        # are then handled by the exact binary fit below.
        suggested_rows = max(1, (rows * byte_limit * 9) // (packet_bytes * 10))
        start = 0
        while start < rows:
            proposed_stop = min(rows, start + suggested_rows)
            proposed = _slice_table_packet(packet, start, proposed_stop)
            if int(proposed.nbytes) <= byte_limit:
                yield proposed
                start = proposed_stop
                continue

            low = start + 1
            high = proposed_stop
            best: Any | None = None
            best_stop = start
            while low <= high:
                middle = (low + high) // 2
                candidate = _slice_table_packet(packet, start, middle)
                if int(candidate.nbytes) <= byte_limit:
                    best = candidate
                    best_stop = middle
                    low = middle + 1
                else:
                    high = middle - 1
            if best is None:
                single = _slice_table_packet(packet, start, start + 1)
                raise MemoryError(
                    f"{packet.table_name} row {start} is {int(single.nbytes):,} bytes; "
                    f"limit is {byte_limit:,}"
                )
            yield best
            start = best_stop



_OBSERVER_FLOAT_RESIDUAL_FIELDS = frozenset(
    {
        "arrays.raqic_probabilities",
        "arrays.raqic_record_confidence",
        "arrays.raqic_score",
    }
)
_OBSERVER_FLOAT_ABSOLUTE_LIMIT = float(np.finfo(np.float32).eps)


def _observer_state_comparison_passes(comparison: Any) -> bool:
    """Accept only the certified one-epsilon RAQIC observer residual."""

    if bool(comparison.passed):
        return True
    categorical = tuple(comparison.categorical_failures)
    floating = frozenset(str(name) for name in comparison.floating_failures)
    maximum = float(comparison.max_abs_difference)
    return (
        not categorical
        and bool(floating)
        and floating.issubset(_OBSERVER_FLOAT_RESIDUAL_FIELDS)
        and bool(np.isfinite(maximum))
        and maximum <= _OBSERVER_FLOAT_ABSOLUTE_LIMIT
    )

def _phase25_gate(certificate_path: Path, hardening_path: Path) -> dict[str, Any]:
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    hardening = json.loads(hardening_path.read_text(encoding="utf-8"))
    failures = []
    if certificate.get("passed") is not True:
        failures.append("Phase 2.5 certificate did not pass")
    if certificate.get("classification") != "PHASE2_5_TARGET_GPU_CERTIFIED":
        failures.append("Phase 2.5 classification mismatch")
    if certificate.get("phase3_unlocked") is not True:
        failures.append("Phase 2.5 did not unlock Phase 3")
    if hardening.get("passed") is not True:
        failures.append("Phase 2.5 hardening did not pass")
    if hardening.get("base_certified_source_sha256") != certificate.get("source_sha256"):
        failures.append("Phase 2.5 hardening/certificate source mismatch")
    if failures:
        raise RuntimeError("; ".join(failures))
    return {
        "certificate": certificate,
        "certificate_sha256": sha256_file(certificate_path),
        "hardening": hardening,
    }


def _device_metadata(run: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "backend": run.ds.backend.name,
        "is_gpu": bool(run.ds.is_gpu),
        "backend_info": run.ds.backend.info,
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    if run.ds.is_gpu:
        cp = run.ds.xp
        properties = cp.cuda.runtime.getDeviceProperties(0)
        name = properties["name"]
        if isinstance(name, bytes):
            name = name.decode()
        payload["cuda_device"] = {
            "name": str(name),
            "device_id": int(cp.cuda.Device().id),
            "compute_capability": f"{properties['major']}.{properties['minor']}",
            "total_global_memory_bytes": int(properties["totalGlobalMem"]),
            "multiprocessor_count": int(properties["multiProcessorCount"]),
            "runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
            "driver_version": int(cp.cuda.runtime.driverGetVersion()),
            "cupy_version": str(cp.__version__),
        }
    return payload


def _effective_qiskit_mode(cfg: Any) -> str:
    """Resolve the execution mode while accepting the supported alias."""
    mode = str(getattr(cfg.raqic, "qiskit_decision_mode", "off"))
    if bool(getattr(cfg.raqic, "use_qiskit_for_all", False)) and mode in {
        "off",
        "validation_sample",
    }:
        return "every_ow_static_exact"
    return mode


def _action_family_lexsort(xp: Any, sequence: Any, actions: Any) -> Any:
    """Return action-major stable order using the NumPy/CuPy common API."""
    keys = xp.stack((sequence, actions), axis=0)
    return xp.lexsort(keys)


def _preflight_hardware(backend: str) -> dict[str, Any]:
    """Fail closed on the requested backend and return scalar device evidence."""
    payload: dict[str, Any] = {
        "requested_backend": backend,
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    if backend == "numpy":
        payload.update({"backend": "numpy", "is_gpu": False})
        return payload
    import cupy as cp

    count = int(cp.cuda.runtime.getDeviceCount())
    if count < 1:
        raise RuntimeError("CuPy corpus preflight found no CUDA device")
    properties = cp.cuda.runtime.getDeviceProperties(0)
    name = properties["name"]
    if isinstance(name, bytes):
        name = name.decode()
    payload.update(
        {
            "backend": "cupy",
            "is_gpu": True,
            "cuda_device_count": count,
            "cuda_device": {
                "name": str(name),
                "device_id": int(cp.cuda.Device().id),
                "compute_capability": f"{properties['major']}.{properties['minor']}",
                "total_global_memory_bytes": int(properties["totalGlobalMem"]),
                "multiprocessor_count": int(properties["multiProcessorCount"]),
                "runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
                "driver_version": int(cp.cuda.runtime.driverGetVersion()),
                "cupy_version": str(cp.__version__),
            },
        }
    )
    return payload


def _validate_non_qiskit_science_config(cfg: Any, backend: str) -> None:
    """Validate the certified dense counterfactual path without invoking Qiskit."""
    failures: list[str] = []
    if _effective_qiskit_mode(cfg) != "off":
        failures.append("Qiskit mode is not off")
    if bool(getattr(cfg.raqic, "qiskit_allow_automatic_execution_fallback", False)):
        failures.append("automatic Qiskit execution fallback is prohibited")
    if str(cfg.visualization.backend) != "none":
        failures.append("corpus execution requires visualization.backend=none")
    if not bool(cfg.counterfactual.enabled):
        failures.append("counterfactual execution is disabled")
    if str(cfg.counterfactual.source_boundary) != "post_selection_pre_actions":
        failures.append("counterfactual source boundary mismatch")
    if not bool(cfg.action_transitions.enabled):
        failures.append("authoritative action-transition contract is disabled")
    if str(cfg.action_transitions.action_contract_version) != "owl.action-transitions.v1":
        failures.append("authoritative action-transition contract version mismatch")
    if bool(cfg.action_transitions.legacy_unsupported_action_recovery):
        failures.append("legacy unsupported-action recovery is enabled")
    if backend == "cupy":
        if str(cfg.counterfactual.backend) != "cupy":
            failures.append("target-GPU corpus config does not select CuPy")
        if str(cfg.counterfactual.branch_execution_mode) != "target_gpu_required":
            failures.append("target-GPU-required branch execution is not enabled")
        if str(cfg.raqic.mode) != "gpu_full":
            failures.append("target-GPU corpus config does not select RAQIC gpu_full")
        if not bool(cfg.raqic.full_gpu_strict):
            failures.append("strict persistent GPU execution is disabled")
        if bool(cfg.raqic.fallback_on_backend_error):
            failures.append("GPU backend fallback is enabled")
    if failures:
        raise ValueError("; ".join(failures))


def _prepare_corpus_preflight(
    *,
    engine: Path,
    config_path: Path,
    output: Path,
    cfg: Any,
    backend: str,
    phase3_source: str,
    qiskit_validator: Any,
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    """Create explicit exercised/not-exercised Qiskit evidence for one unit."""
    qiskit_mode = _effective_qiskit_mode(cfg)
    qiskit_exercised = qiskit_mode != "off"
    if qiskit_exercised:
        receipt = qiskit_validator(
            repo=engine,
            config_path=config_path,
            run_root=output / "preflight",
            allow_cpu=backend == "numpy",
        )
        qiskit = dict(receipt.get("qiskit_execution", {}))
        qiskit.update(
            {
                "exercised": True,
                "mode": qiskit_mode,
                "evidence_status": "executed",
                "runtime_binding_required": True,
                "automatic_execution_fallback": bool(
                    getattr(
                        cfg.raqic,
                        "qiskit_allow_automatic_execution_fallback",
                        False,
                    )
                ),
            }
        )
        if qiskit.get("passed") is not True:
            raise RuntimeError("corpus unit lacks passing Qiskit preflight evidence")
        if backend == "cupy" and qiskit.get("runtime_binding_used") is not True:
            raise RuntimeError("target-GPU Qiskit corpus lacks runtime-binding evidence")
        receipt["qiskit_execution"] = qiskit
        atomic_json(output / "preflight" / "preflight_receipt.json", receipt)
        return receipt, qiskit, True

    _validate_non_qiskit_science_config(cfg, backend)
    qiskit = {
        "passed": True,
        "exercised": False,
        "mode": "off",
        "evidence_status": "not_exercised",
        "runtime_binding_required": False,
        "runtime_binding_used": False,
        "automatic_execution_fallback": False,
    }
    receipt = {
        "schema_version": "owl.cadc.phase4-corpus-unit-preflight.v2",
        "passed": True,
        "created_at": datetime.now(UTC).isoformat(),
        "scientific_ticks_started": 0,
        "repo": str(engine),
        "config": str(config_path),
        "source_sha256": phase3_source,
        "config_sha256": sha256_file(config_path),
        "hardware": _preflight_hardware(backend),
        "qiskit_execution": qiskit,
        "qiskit_evidence_contract": "explicit_exercised_or_not_exercised.v1",
        "visualization_backend": str(cfg.visualization.backend),
        "automatic_fallback_allowed": False,
    }
    atomic_json(output / "preflight" / "preflight_receipt.json", receipt)
    return receipt, qiskit, False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--unit-id", required=True)
    parser.add_argument("--context-family", required=True)
    parser.add_argument("--source-tick", required=True, type=int)
    parser.add_argument("--phase25-certificate", required=True)
    parser.add_argument("--hardening-receipt", required=True)
    parser.add_argument("--backend", choices=("numpy", "cupy"), required=True)
    parser.add_argument(
        "--branch-transfer-mode",
        choices=("immediate_reference", "deferred_bounded"),
        default="immediate_reference",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate one derived unit without starting scientific ticks",
    )
    parser.add_argument(
        "--worker-device-budget-bytes",
        type=int,
        default=0,
        help="this process's fail-closed share of the aggregate CUDA memory envelope",
    )
    args = parser.parse_args()
    if args.worker_device_budget_bytes < 0:
        raise ValueError("worker-device-budget-bytes cannot be negative")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    inventory_path = output / "corpus_unit_inventory.json"
    engine = Path(args.engine_root).resolve()
    engine_src = engine / "src"
    sys.path.insert(0, str(engine_src))

    try:
        from owl.core.actions import Action
        from owl.core.config import SimulationConfig, load_config
        from owl.core.init import initialize_world
        from owl.counterfactual.scheduler import CounterfactualScheduler
        from owl.counterfactual.source import CounterfactualSourceCollector
        from owl.counterfactual.staging import stage_counterfactual_result
        from owl.counterfactual.state_hash import compare_state_science, hash_state
        from owl.counterfactual.writer import CounterfactualWriter
        from owl.experiments.controller import (
            _release_hash,
            _snapshot_with_events,
            validate_preflight,
        )
        from owl.gpu.memory_model import build_counterfactual_memory_plan
        from owl.gpu.run_context import PersistentOWLDeviceRun
        from owl.record.cadc_schema import CADC_ACTION_TRANSITION_SCHEMA_DIGEST
        from owl.record.replay_recorder import ReplayRecorder

        class Phase4SourceCollector(CounterfactualSourceCollector):
            """Collect source rows in deterministic order with CuPy-compatible operations."""

            def _select_flats(self, buffer: Any) -> Any:
                if self.cfg.counterfactual.source_selection_mode != "action_family_stratified":
                    return super()._select_flats(buffer)
                xp = buffer.xp
                sequence = buffer.arrays["decision_sequence"].reshape(-1)
                live = xp.nonzero(sequence >= 0)[0]
                actions = buffer.arrays["selected_action"].reshape(-1)[live]
                order = _action_family_lexsort(xp, sequence[live], actions)
                live = live[order]
                remaining = int(self.cfg.counterfactual.max_source_decisions) - sum(
                    source.decisions.count for source in self.sources
                )
                return live[: max(remaining, 0)]

        phase3_source = _release_hash(engine)
        expected = "d17ef58692c7663eb0cc87ab4cdf7e74ca9b529091fcab4f15b6fe28e2a607a3"
        if phase3_source != expected:
            raise RuntimeError(
                f"immutable Phase 3 engine mismatch: expected {expected}, found {phase3_source}"
            )
        gate = _phase25_gate(
            Path(args.phase25_certificate).resolve(),
            Path(args.hardening_receipt).resolve(),
        )
        cfg = load_config(args.config)
        if not isinstance(cfg, SimulationConfig):
            raise TypeError("derived Phase 3 configuration did not validate")
        preflight, qiskit, qiskit_exercised = _prepare_corpus_preflight(
            engine=engine,
            config_path=Path(args.config).resolve(),
            output=output,
            cfg=cfg,
            backend=args.backend,
            phase3_source=phase3_source,
            qiskit_validator=validate_preflight,
        )
        if args.preflight_only:
            atomic_json(
                output / "corpus_unit_preflight_smoke.json",
                {
                    "schema_version": "owl.cadc.phase4-corpus-unit-preflight-smoke.v1",
                    "passed": True,
                    "unit_id": args.unit_id,
                    "phase3_source_sha256": phase3_source,
                    "config_sha256": sha256_file(Path(args.config).resolve()),
                    "backend": args.backend,
                    "qiskit_exercised": qiskit_exercised,
                    "qiskit_execution": qiskit,
                    "scientific_ticks_started": 0,
                },
            )
            return 0
        maximum_horizon = max(
            {*cfg.counterfactual.horizons}.union(
                *[set(values) for values in cfg.counterfactual.family_horizons.values()]
            )
        )
        total_ticks = args.source_tick + maximum_horizon
        payload = cfg.model_dump(mode="json")
        payload["world"]["max_steps"] = total_ticks
        payload["counterfactual"]["backend"] = args.backend
        if args.backend == "numpy":
            payload["counterfactual"]["branch_execution_mode"] = "segmented"
            payload["raqic"]["full_gpu_strict"] = False
            payload["raqic"]["fallback_on_backend_error"] = True
        cfg = SimulationConfig.model_validate(payload)
        initial = initialize_world(cfg, np.random.default_rng(int(cfg.world.seed)))

        control_cfg = cfg.model_copy(deep=True)
        control_cfg.counterfactual.enabled = False
        control_cfg.recording.cadc.enabled = False
        control_cfg.recording.enabled = False
        control = PersistentOWLDeviceRun.from_config(
            control_cfg,
            initial_state=copy.deepcopy(initial),
            force_backend=args.backend,
            output_root=output / "control_science",
        )
        collector = Phase4SourceCollector(
            cfg,
            str(gate["certificate"]["source_sha256"]),
            run_id=args.unit_id,
            condition=args.context_family,
        )
        delayed = DelayedCollector(collector, args.source_tick)
        factual = PersistentOWLDeviceRun.from_config(
            cfg,
            initial_state=copy.deepcopy(initial),
            force_backend=args.backend,
            output_root=output / "factual_science",
            counterfactual_observer=delayed,
        )
        recorder = ReplayRecorder(
            output / "factual_bundle",
            run_id=args.unit_id,
            condition=args.context_family,
            seed=int(cfg.world.seed),
            requested_ticks=total_ticks,
            recording_tier="metrics_only",
            source_sha256=phase3_source,
            config_sha256=sha256_file(Path(args.config)),
            action_names=[action.name for action in Action],
            hardware={**dict(preflight.get("hardware", {})), **_device_metadata(factual)},
            qiskit_execution=qiskit,
            cadc_config=cfg.recording.cadc,
        )
        active_recorder: ReplayRecorder | None = recorder
        started = perf_counter()
        try:
            for _ in range(total_ticks):
                control.step()
                diagnostics = factual.step()
                snapshot = _snapshot_with_events(factual, recording_tier="metrics_only")
                recorder.record_device(factual.ds, snapshot, diagnostics=diagnostics)
            factual_hash_before = hash_state(factual.ds)
            recovery = compare_state_science(factual.ds, control.ds)
            if not _observer_state_comparison_passes(recovery):
                raise AssertionError(f"corpus observer changed factual science: {recovery}")
            if not collector.sources:
                raise RuntimeError("pre-registered source tick produced no source decisions")
            actual_free_device = None
            free_device = None
            if factual.ds.is_gpu:
                actual_free_device = int(factual.ds.xp.cuda.runtime.memGetInfo()[0])
                free_device = actual_free_device
                if args.worker_device_budget_bytes:
                    free_device = min(
                        free_device,
                        int(args.worker_device_budget_bytes),
                    )
            memory = build_counterfactual_memory_plan(
                factual.ds,
                cfg,
                scratch_bytes=int(factual.scratch.spec_bytes()),
                free_device_bytes=free_device,
            )
            if not memory.passed:
                raise MemoryError("counterfactual memory plan cannot fit one branch")
            scheduler_class = CounterfactualScheduler
            if args.branch_transfer_mode == "deferred_bounded":
                if args.backend != "cupy":
                    raise ValueError("deferred branch transfers require CuPy")
                from _phase4_counterfactual_runtime import (
                    DeferredTransferCounterfactualScheduler,
                )

                scheduler_class = DeferredTransferCounterfactualScheduler
            scheduler = scheduler_class(
                factual,
                cfg,
                active_branch_limit=int(memory.max_active_branches),
            )
            results = [scheduler.run_source(source) for source in collector.sources]
            factual_hash_after = hash_state(factual.ds)
            if factual_hash_before.root != factual_hash_after.root:
                raise AssertionError("counterfactual corpus branches mutated factual state")
            failed = [
                branch
                for result in results
                for branch in result.branches
                if branch.status.value != "completed"
            ]
            if failed:
                raise RuntimeError(f"{len(failed)} counterfactual corpus branches failed")
            writer = CounterfactualWriter(
                output / "counterfactual",
                source_sha256=phase3_source,
                phase25_certificate_sha256=str(gate["certificate_sha256"]),
                factual_v2_digest=CADC_ACTION_TRANSITION_SCHEMA_DIGEST,
                max_packet_bytes=int(cfg.counterfactual.max_packet_bytes),
                max_pending_bytes=int(cfg.counterfactual.max_pending_bytes),
                row_group_rows=int(cfg.counterfactual.parquet_row_group_rows),
            )
            staged_packets = (
                packet
                for source, result in zip(collector.sources, results, strict=True)
                for packet in stage_counterfactual_result(source, result)
            )
            packets = _bounded_table_packets(
                staged_packets,
                max_packet_bytes=int(cfg.counterfactual.max_packet_bytes),
                max_pending_bytes=int(cfg.counterfactual.max_pending_bytes),
            )
            receipts = writer.write_packets(packets)
            recorder.close(state="SUCCEEDED")
            active_recorder = None
            counts: dict[str, int] = {}
            part_counts: dict[str, int] = {}
            for receipt in receipts:
                counts[receipt.table_name] = counts.get(receipt.table_name, 0) + receipt.rows
                part_counts[receipt.table_name] = part_counts.get(receipt.table_name, 0) + 1
            branch_runtimes = [
                float(branch.runtime_seconds)
                for result in results
                for branch in result.branches
            ]
            device_pool = None
            if factual.ds.is_gpu:
                pool = factual.ds.xp.get_default_memory_pool()
                device_pool = {
                    "used_bytes": int(pool.used_bytes()),
                    "reserved_bytes": int(pool.total_bytes()),
                    "free_device_bytes": int(
                        factual.ds.xp.cuda.runtime.memGetInfo()[0]
                    ),
                }
            inventory = {
                "schema_version": "owl.cadc.phase4-corpus-unit-inventory.v1",
                "unit_id": args.unit_id,
                "context_family": args.context_family,
                "seed": int(cfg.world.seed),
                "source_tick": args.source_tick,
                "phase3_source_sha256": phase3_source,
                "phase25_certificate_sha256": gate["certificate_sha256"],
                "preflight_receipt_sha256": sha256_file(
                    output / "preflight" / "preflight_receipt.json"
                ),
                "qiskit_execution": qiskit,
                "qiskit_exercised": qiskit_exercised,
                "qiskit_gpu_runtime_required": (
                    args.backend == "cupy" and qiskit_exercised
                ),
                "qiskit_evidence_contract": "explicit_exercised_or_not_exercised.v1",
                "execution_backend": args.backend,
                "branch_transfer_mode": args.branch_transfer_mode,
                "factual_v2_digest": CADC_ACTION_TRANSITION_SCHEMA_DIGEST,
                "passed": True,
                "device": _device_metadata(factual),
                "runtime_seconds": perf_counter() - started,
                "counterfactual_memory_plan": memory.to_dict(),
                "worker_device_budget_bytes": (
                    int(args.worker_device_budget_bytes)
                    if args.worker_device_budget_bytes
                    else None
                ),
                "actual_free_device_bytes_before_branches": actual_free_device,
                "effective_free_device_bytes_for_plan": free_device,
                "branch_worker_count": int(scheduler.last_worker_count),
                "branch_count": len(branch_runtimes),
                "branch_runtime_seconds": {
                    "sum": float(sum(branch_runtimes)),
                    "minimum": min(branch_runtimes, default=0.0),
                    "maximum": max(branch_runtimes, default=0.0),
                },
                "device_memory_observation": device_pool,
                "source_decisions": counts.get("source_decisions", 0),
                "branch_horizons": counts.get("counterfactual_micro_rollouts", 0),
                "candidate_pairs": counts.get("candidate_pairs", 0),
                "row_counts": counts,
                "part_counts": part_counts,
                "bounded_packet_limit_bytes": min(
                    int(cfg.counterfactual.max_packet_bytes),
                    int(cfg.counterfactual.max_pending_bytes),
                ),
                "factual_root": str(output / "factual_bundle"),
                "counterfactual_root": str(output / "counterfactual"),
                "factual_nonmutation_root": factual_hash_after.root,
            }
            atomic_json(inventory_path, inventory)
        finally:
            if active_recorder is not None:
                with contextlib.suppress(Exception):
                    active_recorder.close(
                        state="FAILED_PARTIAL", failure="unit_exception"
                    )
            factual.close(checkpoint=False)
            control.close(checkpoint=False)
        return 0
    except Exception as exc:
        atomic_json(
            inventory_path,
            {
                "schema_version": "owl.cadc.phase4-corpus-unit-inventory.v1",
                "unit_id": args.unit_id,
                "passed": False,
                "classification": "FAILED_CLOSED",
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc().splitlines(),
            },
        )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
