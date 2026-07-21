from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import queue
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from owl.core.advanced import ensure_advanced_fields
from owl.core.config import SimulationConfig, load_config
from owl.core.init import initialize_world
from owl.raqic.state import ensure_raqic_fields
from owl.record.snapshots import load_snapshot
from owl.runtime.run_result import RunResult

from .partition import partition_rows
from .rank_context import execute_rank
from .state_shard import merge_local_states


def _uid_worker(conn: Any) -> Any:
    try:
        import cupy as cp

        conn.send(("ok", cp.cuda.nccl.get_unique_id()))
    except BaseException as exc:
        conn.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        conn.close()


def _create_unique_id(ctx: Any) -> Any:
    parent, child = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_uid_worker, args=(child,))
    proc.start()
    status, payload = parent.recv()
    proc.join(timeout=30)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        raise TimeoutError("timed out creating NCCL unique id")
    if status != "ok":
        raise RuntimeError(f"cannot create NCCL unique id: {payload}")
    return payload


def _rank_worker(result_queue: Any, kwargs: Any) -> Any:
    report = execute_rank(**kwargs)
    result_queue.put(report.to_dict())


def _certify_collective_ledgers(
    report_dicts: list[dict[str, Any]],
    expected_rank_count: int | None = None,
) -> dict[str, Any]:
    """Validate collective symmetry and paired NCCL point-to-point calls."""
    failures: list[str] = []
    if expected_rank_count is not None and len(report_dicts) != int(expected_rank_count):
        failures.append(
            "rank report count mismatch: "
            f"expected {int(expected_rank_count)}, got {len(report_dicts)}"
        )
    reported_ranks = [int(report.get("rank", -1)) for report in report_dicts]
    if len(set(reported_ranks)) != len(reported_ranks):
        failures.append(f"duplicate rank reports: {reported_ranks}")
    collective_sequences = []
    sends: Counter[tuple[Any, ...]] = Counter()
    recvs: Counter[tuple[Any, ...]] = Counter()
    for report in report_dicts:
        rank = int(report["rank"])
        ledger = list(report.get("collective_ledger") or [])
        collectives = []
        for row in ledger:
            operation = str(row.get("operation"))
            count = int(row.get("count", -1))
            dtype = str(row.get("dtype"))
            tick = int(row.get("tick", -1))
            peer = int(row.get("peer_or_root", -1))
            phase = str(row.get("phase", "unspecified"))
            field_group = str(row.get("field_group", "unspecified"))
            if operation == "send":
                sends[(rank, peer, tick, count, dtype, phase, field_group)] += 1
            elif operation == "recv":
                recvs[(peer, rank, tick, count, dtype, phase, field_group)] += 1
            else:
                collectives.append((operation, count, dtype, peer, tick, phase, field_group))
        collective_sequences.append((rank, tuple(collectives)))
    if collective_sequences:
        reference_rank, reference = collective_sequences[0]
        for rank, sequence in collective_sequences[1:]:
            if sequence != reference:
                failures.append(
                    f"rank {rank} collective sequence differs from rank {reference_rank}"
                )
    if sends != recvs:
        missing_receives = list((sends - recvs).elements())[:16]
        missing_sends = list((recvs - sends).elements())[:16]
        if missing_receives:
            failures.append(f"unmatched NCCL sends: {missing_receives}")
        if missing_sends:
            failures.append(f"unmatched NCCL receives: {missing_sends}")
    boundary_ok = all(
        int((report.get("halo_stats") or {}).get("boundary_checks", 0)) > 0
        and int((report.get("halo_stats") or {}).get("boundary_elements", 0)) > 0
        and int((report.get("halo_stats") or {}).get("boundary_mismatch_count", 0)) == 0
        for report in report_dicts
    )
    if not boundary_ok:
        failures.append("one or more ranks did not complete boundary consistency checks")
    return {
        "passed": not failures,
        "failures": failures,
        "strategy": "redundant_overlap_target_owner_commit",
        "collective_sequences_match": not any("collective sequence" in item for item in failures),
        "point_to_point_pairs_match": not any("unmatched NCCL" in item for item in failures),
        "boundary_consistency_passed": boundary_ok,
        "rank_count": len(report_dicts),
    }


def _merge_metrics(report_dicts: Any) -> Any:
    rank_metrics = []
    for report in report_dicts:
        path = report.get("metrics_path")
        if path:
            rank_metrics.append(json.loads(Path(path).read_text(encoding="utf-8")))
    if not rank_metrics:
        return []
    ticks = len(rank_metrics[0])
    if any(len(items) != ticks for items in rank_metrics):
        raise RuntimeError("distributed ranks produced different metric lengths")
    merged = []
    for i in range(ticks):
        rows = [items[i] for items in rank_metrics]
        alive = sum(int(row["alive_count"]) for row in rows)
        health_sum = sum(float(row["health_sum"]) for row in rows)
        resource_sum = sum(float(row["resource_sum"]) for row in rows)
        merged.append(
            {
                "tick": int(rows[0]["tick"]),
                "alive_count": alive,
                "mean_health": health_sum / max(alive, 1),
                "mean_resource": resource_sum / max(alive, 1),
                "food_total": sum(float(row["food_total"]) for row in rows),
                "distributed_ranks": len(rows),
            }
        )
    return merged


def run_distributed(cfg: SimulationConfig, plan: Any) -> RunResult:
    """Run one spatially sharded simulation with one process per GPU."""
    devices = tuple(int(x) for x in plan.device_ids)
    if len(devices) < 2:
        raise ValueError("distributed execution requires at least two devices")
    shards = partition_rows(
        cfg.world.height,
        cfg.world.width,
        len(devices),
        cfg.world.patch_size,
        boundary_mode=cfg.world.boundary_mode,
    )
    timeout = float(cfg.raqic.full_gpu_distributed_timeout_seconds)
    output = Path(cfg.recording.metrics_path).parent / "distributed_v09"
    output.mkdir(parents=True, exist_ok=True)

    ctx = mp.get_context("spawn")
    unique_id = _create_unique_id(ctx)
    result_queue = ctx.Queue()
    processes = []
    cfg_data = cfg.model_dump(mode="python")
    for rank, (device, shard) in enumerate(zip(devices, shards, strict=True)):
        kwargs = {
            "rank": rank,
            "device_id": device,
            "world_size": len(devices),
            "unique_id": unique_id,
            "cfg_data": cfg_data,
            "plan": plan,
            "shard": shard,
            "output_dir": str(output),
        }
        proc = ctx.Process(target=_rank_worker, args=(result_queue, kwargs))
        proc.start()
        processes.append(proc)

    reports: list[dict[str, Any]] = []
    deadline = time.monotonic() + timeout
    try:
        while len(reports) < len(processes):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"distributed run exceeded {timeout:.1f}s without all rank reports"
                )
            try:
                reports.append(result_queue.get(timeout=min(1.0, remaining)))
            except queue.Empty as exc:
                failed = [
                    (rank, proc.exitcode)
                    for rank, proc in enumerate(processes)
                    if proc.exitcode not in (None, 0)
                ]
                if failed:
                    raise RuntimeError(f"distributed ranks exited early: {failed}") from exc
    except BaseException:
        for proc in processes:
            if proc.is_alive():
                proc.terminate()
        raise
    finally:
        for proc in processes:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
                proc.join()

    reports.sort(key=lambda item: int(item["rank"]))
    failures = [item for item in reports if not item.get("success")]
    if failures:
        failure_path = output / "distributed_failure.json"
        failure_path.write_text(
            json.dumps({"reports": reports}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise RuntimeError(
            "distributed execution failed: "
            + "; ".join(f"rank {item['rank']}: {item.get('error', 'unknown')}" for item in failures)
        )

    rng = np.random.default_rng(int(cfg.world.seed))
    base = initialize_world(cfg, rng)
    ensure_advanced_fields(base, cfg)
    if cfg.raqic.enabled:
        ensure_raqic_fields(base, cfg)
    locals_with_shards = [
        (shard, load_snapshot(report["snapshot_path"]))
        for shard, report in zip(shards, reports, strict=True)
    ]
    final_state = merge_local_states(base, locals_with_shards, cfg)
    metrics = _merge_metrics(reports)
    distributed_certification = _certify_collective_ledgers(
        reports, expected_rank_count=len(devices)
    )
    if not distributed_certification["passed"]:
        failure_path = output / "distributed_certification_failure.json"
        failure_path.write_text(
            json.dumps(distributed_certification, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise RuntimeError(
            "distributed communication certification failed: "
            + "; ".join(distributed_certification["failures"])
        )

    metadata = {
        "simulation_backend": "gpu_multi",
        "decision_backend": plan.decision_backend,
        "device_ids": list(devices),
        "world_size": len(devices),
        "shards": [shard.to_dict() for shard in shards],
        "rank_reports": reports,
        "fallback_count": 0,
        "checkpoint_count": len(reports),
        "distributed_certification": distributed_certification,
        "graph": {
            "rank_graphs": [item.get("graph", {}) for item in reports],
            "distributed_communication_graph_captured": False,
            "label": "rank-local compute graph with eager NCCL communication",
        },
    }
    manifest = output / "distributed_manifest.json"
    manifest.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifacts = tuple(
        [manifest]
        + [Path(item["snapshot_path"]) for item in reports]
        + [Path(item["metrics_path"]) for item in reports]
    )
    return RunResult(
        state=final_state,
        metrics=metrics,
        execution_plan=plan,
        execution_metadata=metadata,
        artifacts=artifacts,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run process-per-GPU OWL.")
    parser.add_argument("--devices", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg = cfg.model_copy(deep=True)
    cfg.raqic.full_gpu_multi_gpu = True
    cfg.raqic.full_gpu_devices = tuple(
        int(item) for item in args.devices.split(",") if item.strip()
    )
    from owl.runtime.capabilities import detect_runtime_capabilities
    from owl.runtime.execution_plan import compile_execution_plan

    plan = compile_execution_plan(cfg, detect_runtime_capabilities())
    result = run_distributed(cfg, plan)
    print(json.dumps(result.summary(), indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
