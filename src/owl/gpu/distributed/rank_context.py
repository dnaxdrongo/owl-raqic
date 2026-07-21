from __future__ import annotations

import contextlib
import json
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from owl.core.advanced import ensure_advanced_fields
from owl.core.config import SimulationConfig
from owl.core.init import initialize_world
from owl.gpu.run_context import PersistentOWLDeviceRun
from owl.gpu.stages.topdown_gpu import dispatch_parent_context_gpu
from owl.gpu.transfer_ledger import TransferLedger
from owl.raqic.state import ensure_raqic_fields
from owl.record.snapshots import save_snapshot

from .boundary_consistency import verify_and_commit_boundaries
from .distributed_visual import gather_rgba_frame, gather_visual_events, make_global_visual_frame
from .global_reductions import synchronize_global_context
from .halo_exchange import exchange_halos
from .nccl_transport import NCCLTransport
from .partition import SpatialShard
from .state_shard import create_local_state


@dataclass
class RankRunReport:
    rank: int
    success: bool
    device_id: int
    snapshot_path: str | None
    metrics_path: str | None
    error: str | None
    halo_stats: dict[str, Any]
    graph: dict[str, Any]
    collective_ledger_hash: str
    collective_ledger: list[dict[str, Any]]

    def to_dict(self) -> Any:
        return self.__dict__.copy()


def _weighted_rank_metric(run: Any, shard: SpatialShard) -> dict[str, Any]:
    ds = run.ds
    xp = ds.xp
    interior = shard.interior_rows
    health = ds.health[interior, :]
    obstacle = ds.obstacle[interior, :]
    live = (health > 0) & (~obstacle)
    alive = xp.sum(live, dtype=xp.int64)
    health_sum = xp.sum(xp.where(live, health, 0.0), dtype=xp.float64)
    resource_sum = xp.sum(xp.where(live, ds.resource[interior, :], 0.0), dtype=xp.float64)
    food_sum = xp.sum(ds.food[interior, :], dtype=xp.float64)
    values_device = xp.stack(
        [
            alive.astype(xp.float64),
            health_sum,
            resource_sum,
            food_sum,
        ]
    )
    values = ds.backend.asnumpy(values_device)
    transfer_ledger = ds.metadata.get("transfer_ledger")
    if isinstance(transfer_ledger, TransferLedger):
        transfer_ledger.record_d2h(
            int(values_device.nbytes),
            kind="metric",
            tick=int(ds.tick),
            source_stream="distributed-metric",
            synchronization="stream",
            scheduled=True,
            graph_compatible=False,
            reason="one compact rank metric record",
        )
    return {
        "tick": int(ds.tick),
        "rank": int(shard.rank),
        "alive_count": int(values[0]),
        "health_sum": float(values[1]),
        "resource_sum": float(values[2]),
        "food_total": float(values[3]),
    }


def execute_rank(
    *,
    rank: int,
    device_id: int,
    world_size: int,
    unique_id: int,
    cfg_data: dict[str, Any],
    plan: Any,
    shard: SpatialShard,
    output_dir: str,
) -> RankRunReport:
    """Run one spatial shard in one CUDA process."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    transport = None
    run = None
    try:
        import cupy as cp

        cp.cuda.Device(int(device_id)).use()
        cfg = SimulationConfig.model_validate(cfg_data)
        rng = np.random.default_rng(int(cfg.world.seed))
        global_state = initialize_world(cfg, rng)
        ensure_advanced_fields(global_state, cfg)
        if cfg.raqic.enabled:
            ensure_raqic_fields(global_state, cfg)
        local_state, local_cfg = create_local_state(global_state, cfg, shard)

        # Allocate disjoint identity ranges; a run cannot exhaust this stride
        # without exceeding the dense grid by orders of magnitude.
        id_stride = int(cfg.world.height * cfg.world.width * (cfg.world.max_steps + 2))
        local_state.next_ow_id = int(global_state.next_ow_id + rank * id_stride)

        local_plan = replace(
            plan,
            simulation_backend=(
                "gpu_graph" if plan.simulation_backend == "gpu_graph" else "gpu_persistent"
            ),
            device_ids=(int(device_id),),
            multi_gpu=False,
            visual_backend="none",
            require_certification=False,
            graph_requirement=(
                "allow_partial" if plan.graph_requirement == "full_tick" else plan.graph_requirement
            ),
        )
        transport = NCCLTransport.connect(
            rank=rank,
            world_size=world_size,
            unique_id=unique_id,
            device_id=device_id,
        )
        run = PersistentOWLDeviceRun.from_config(
            local_cfg,
            initial_state=local_state,
            plan=local_plan,
        )
        distributed_visual = None
        if getattr(plan, "visual_backend", "none") != "none":
            from owl.viz.controller import VisualController

            distributed_visual = VisualController.from_config(
                cfg,
                backend_name=(str(plan.visual_backend) if rank == 0 else "none"),
            )
        halo_totals = {"fields": 0, "elements_sent": 0, "elements_received": 0}
        rank_metrics = []

        for _ in range(int(cfg.world.max_steps)):
            tick = int(run.ds.tick) + 1
            with run.streams.compute:
                halo = exchange_halos(
                    run.ds,
                    shard,
                    transport,
                    run.streams.compute,
                    tick=tick,
                )
                run.ds.tick = tick
                run._execute_segment("predecision")
                synchronize_global_context(
                    run.ds,
                    local_cfg,
                    shard,
                    transport,
                    run.streams.compute,
                    tick=tick,
                )
                dispatch_parent_context_gpu(run.ds, local_cfg)
                run._execute_segment("decision")
                run._execute_segment("actions")
                boundary_actions = verify_and_commit_boundaries(
                    run.ds,
                    shard,
                    transport,
                    run.streams.compute,
                    tick=tick,
                    tolerance=float(cfg.raqic.full_gpu_shadow_tolerance),
                    strict=True,
                )
                run._execute_segment("postdecision")
                boundary_post = verify_and_commit_boundaries(
                    run.ds,
                    shard,
                    transport,
                    run.streams.compute,
                    tick=tick,
                    tolerance=float(cfg.raqic.full_gpu_shadow_tolerance),
                    strict=True,
                )
                halo2 = boundary_post
                synchronize_global_context(
                    run.ds,
                    local_cfg,
                    shard,
                    transport,
                    run.streams.compute,
                    tick=tick,
                )
                dispatch_parent_context_gpu(run.ds, local_cfg)
                if distributed_visual is not None and distributed_visual.render_due(tick):
                    shards = tuple(
                        __import__(
                            "owl.gpu.distributed.partition", fromlist=["partition_rows"]
                        ).partition_rows(
                            cfg.world.height,
                            cfg.world.width,
                            world_size,
                            cfg.world.patch_size,
                            boundary_mode=cfg.world.boundary_mode,
                        )
                    )
                    rgba = gather_rgba_frame(
                        run.ds, shard, shards, transport, run.streams.compute, tick=tick
                    )
                    events = gather_visual_events(
                        distributed_visual,
                        run.ds,
                        shard,
                        shards,
                        transport,
                        run.streams.compute,
                        tick=tick,
                        per_rank_capacity=max(
                            1,
                            int(cfg.raqic.full_gpu_visual_event_capacity) // world_size,
                        ),
                    )
                    if rank == 0 and rgba is not None and events is not None:
                        frame = make_global_visual_frame(distributed_visual, rgba, events, tick)
                        distributed_visual.submit(frame, simulation_ms=1.0)
            run.streams.compute.synchronize()
            run._steps_completed += 1
            for stats in (halo,):
                for key in halo_totals:
                    halo_totals[key] += int(getattr(stats, key))
            halo_totals.setdefault("boundary_checks", 0)
            halo_totals.setdefault("boundary_elements", 0)
            halo_totals.setdefault("boundary_mismatch_count", 0)
            halo_totals.setdefault("boundary_max_abs_residual_scaled", 0)
            halo_totals["boundary_checks"] += int(
                boundary_actions.checked_fields + boundary_post.checked_fields
            )
            halo_totals["boundary_elements"] += int(
                boundary_actions.compared_elements + boundary_post.compared_elements
            )
            halo_totals["boundary_mismatch_count"] += int(
                boundary_actions.global_mismatch or boundary_post.global_mismatch
            )
            halo_totals["boundary_max_abs_residual_scaled"] = max(
                int(halo_totals["boundary_max_abs_residual_scaled"]),
                int(
                    round(
                        max(
                            boundary_actions.max_abs_residual,
                            boundary_post.max_abs_residual,
                        )
                        * 1_000_000_000_000
                    )
                ),
            )
            rank_metrics.append(_weighted_rank_metric(run, shard))

        state = run.checkpoint()
        snapshot = output / f"rank_{rank:04d}_snapshot.npz"
        save_snapshot(state, str(snapshot))
        metrics_path = output / f"rank_{rank:04d}_metrics.json"
        metrics_path.write_text(
            json.dumps(rank_metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        report = RankRunReport(
            rank=rank,
            success=True,
            device_id=device_id,
            snapshot_path=str(snapshot),
            metrics_path=str(metrics_path),
            error=None,
            halo_stats=halo_totals,
            graph=run.graph_manager.graph_status(),
            collective_ledger_hash=transport.ledger_hash(),
            collective_ledger=transport.ledger_records(),
        )
        (output / f"rank_{rank:04d}_status.json").write_text(
            json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return report
    except BaseException as exc:
        if transport is not None:
            with contextlib.suppress(Exception):
                transport.abort()
        report = RankRunReport(
            rank=rank,
            success=False,
            device_id=device_id,
            snapshot_path=None,
            metrics_path=None,
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            halo_stats={},
            graph={},
            collective_ledger_hash="",
            collective_ledger=[],
        )
        (output / f"rank_{rank:04d}_status.json").write_text(
            json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return report
    finally:
        if "distributed_visual" in locals() and distributed_visual is not None:
            with contextlib.suppress(Exception):
                distributed_visual.close()
        if run is not None:
            with contextlib.suppress(Exception):
                run.close(checkpoint=False)
        if transport is not None:
            with contextlib.suppress(Exception):
                transport.close()
