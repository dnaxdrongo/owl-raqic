from __future__ import annotations

from typing import Any

from owl.core.actions import Action, EventKind, SignalChannel
from owl.core.state import EventRecord
from owl.gpu.array_write import write_array
from owl.gpu.stage_metrics import metric_int
from owl.gpu.stages.death_gpu import clear_cell_gpu
from owl.gpu.stencil import neighbor_sum_8
from owl_raqic.random_contract import RNGStream, uniform01


def _host_array(ds: Any, value: Any) -> Any:
    """Return a host NumPy view/copy for a backend array or scalar."""
    return ds.backend.asnumpy(value) if hasattr(value, "shape") else value


def _record_ingestion_events_gpu(
    ds: Any,
    cfg: Any,
    py: Any,
    px: Any,
    iy: Any,
    ix: Any,
    eligible: Any,
    probability: Any,
    success: Any,
    target_resource: Any,
) -> None:
    """Record CPU-compatible sparse ingestion events for GPU collision stage."""
    max_events = int(getattr(getattr(cfg, "recording", object()), "max_events", 4096))
    if max_events <= 0:
        return

    queue = ds.metadata.setdefault("event_queue", [])
    remaining = max(0, max_events - len(queue))
    if remaining <= 0:
        return

    xp = ds.xp
    attempt_idx = xp.nonzero(eligible)[0]
    if int(attempt_idx.shape[0]) == 0:
        return

    attempt_idx = attempt_idx[:remaining]

    py_h = _host_array(ds, py[attempt_idx]).reshape(-1)
    px_h = _host_array(ds, px[attempt_idx]).reshape(-1)
    iy_h = _host_array(ds, iy[attempt_idx]).reshape(-1)
    ix_h = _host_array(ds, ix[attempt_idx]).reshape(-1)
    probability_h = _host_array(ds, probability[attempt_idx]).reshape(-1)
    success_h = _host_array(ds, success[attempt_idx]).reshape(-1)
    target_resource_h = _host_array(ds, target_resource[attempt_idx]).reshape(-1)

    for idx in range(len(py_h)):
        ok = bool(success_h[idx])
        payload = {
            "success": ok,
            "probability": float(probability_h[idx]),
        }
        if ok:
            payload["resource_transfer"] = float(cfg.predation.resource_transfer) * float(
                target_resource_h[idx]
            )

        queue.append(
            EventRecord(
                kind=str(EventKind.INGESTION),
                tick=int(ds.tick),
                source=(int(py_h[idx]), int(px_h[idx])),
                target=(int(iy_h[idx]), int(ix_h[idx])),
                payload=payload,
            )
        )


def resolve_collisions_gpu(ds: Any, cfg: Any) -> dict[str, Any]:
    """Resolve ordinary and ingestion collisions under the shared contract."""
    xp = ds.xp
    ds.metadata["event_queue"] = []
    sy = ds.arrays.get("_collision_source_y", xp.zeros((0,), dtype=xp.int32))
    sx = ds.arrays.get("_collision_source_x", xp.zeros((0,), dtype=xp.int32))
    ty = ds.arrays.get("_collision_target_y", xp.zeros((0,), dtype=xp.int32))
    tx = ds.arrays.get("_collision_target_x", xp.zeros((0,), dtype=xp.int32))
    n = int(sy.shape[0])
    if n == 0:
        return {"collisions": 0, "ordinary": 0, "ingestion_success": 0}
    valid = (
        (ds.health[sy, sx] > 0)
        & (ds.boundary[sy, sx] > 0)
        & (ds.health[ty, tx] > 0)
        & (ds.boundary[ty, tx] > 0)
        & (~ds.obstacle[sy, sx])
        & (~ds.obstacle[ty, tx])
    )
    source_ingest = valid & (ds.readout[sy, sx] == int(Action.INGEST))
    target_ingest = valid & (~source_ingest) & (ds.readout[ty, tx] == int(Action.INGEST))
    ingestion = source_ingest | target_ingest
    ordinary = valid & (~ingestion)
    damage = 0.02 * (ds.aggression[sy, sx] + ds.aggression[ty, tx])
    damage = xp.where(damage > 0, damage, 0.005)
    hdelta = xp.zeros_like(ds.health)
    bdelta = xp.zeros_like(ds.boundary)
    oi = xp.nonzero(ordinary)[0]
    if int(oi.shape[0]):
        xp.add.at(hdelta, (sy[oi], sx[oi]), damage[oi])
        xp.add.at(hdelta, (ty[oi], tx[oi]), damage[oi])
        xp.add.at(bdelta, (sy[oi], sx[oi]), 0.5 * damage[oi])
        xp.add.at(bdelta, (ty[oi], tx[oi]), 0.5 * damage[oi])
    health = xp.clip(ds.health - hdelta, 0.0, 1.0)
    boundary = xp.clip(ds.boundary - bdelta, 0.0, 1.0)
    # Commit ordinary collision pressure before any successful ingestion clears
    # a target. Otherwise refreshing local references after ``clear_cell_gpu``
    # would silently restore the pre-collision health/boundary arrays whenever
    # an unrelated ingestion succeeded in the same collision batch.
    write_array(ds, "health", health)
    write_array(ds, "boundary", boundary)

    ii = xp.nonzero(ingestion)[0]
    success_count = xp.asarray(0, dtype=xp.int32)
    if int(ii.shape[0]):
        py = xp.where(source_ingest[ii], sy[ii], ty[ii])
        px = xp.where(source_ingest[ii], sx[ii], tx[ii])
        iy = xp.where(source_ingest[ii], ty[ii], sy[ii])
        ix = xp.where(source_ingest[ii], tx[ii], sx[ii])
        eligible = (ds.predation[py, px] >= float(cfg.predation.min_predation_trait)) & (
            ds.resource[py, px] > float(cfg.resources.movement_cost)
        )
        pred_score = (
            1.5 * ds.predation[py, px]
            + 0.8 * ds.integration[py, px]
            + 0.5 * ds.resource[py, px]
            + 0.3 * ds.aggression[py, px]
        )
        resist = float(cfg.predation.resistance_weight) * (
            0.8 * health[iy, ix] + 0.8 * boundary[iy, ix] + 0.4 * ds.integration[iy, ix]
        )
        z = pred_score - resist - 0.3
        probability = xp.where(z >= 0, 1.0 / (1.0 + xp.exp(-z)), xp.exp(z) / (1.0 + xp.exp(z)))
        probability = xp.where(eligible, xp.clip(probability, 0.0, 1.0), 0.0)
        pred_id = xp.where(
            ds.occupancy[py, px] >= 0, ds.occupancy[py, px], py * ds.health.shape[1] + px
        )
        target_id = xp.where(
            ds.occupancy[iy, ix] >= 0, ds.occupancy[iy, ix], iy * ds.health.shape[1] + ix
        )
        draws = uniform01(
            int(cfg.world.seed),
            ds.arrays.get("_device_tick", int(ds.tick)),
            pred_id,
            RNGStream.INGESTION_OUTCOME,
            target_id,
            xp=xp,
            dtype=probability.dtype,
        )
        target_resource_all = xp.clip(
            ds.resource[iy, ix],
            0.0,
            float(cfg.resources.max_resource),
        )
        raw_success = eligible & (draws < probability)
        # Stable target-owner commit: the lowest coordinate-ordered collision
        # index wins if multiple predators successfully target one OW.
        flat_target = iy * ds.health.shape[1] + ix
        winner = xp.full((ds.health.size,), n + 1, dtype=xp.int32)
        candidate_index = ii.astype(xp.int32)
        xp.minimum.at(winner, flat_target, xp.where(raw_success, candidate_index, n + 1))
        success = raw_success & (candidate_index == winner[flat_target])
        failure = eligible & (~success)
        cadc_buffer = ds.metadata.get("cadc_device_buffer")
        if cadc_buffer is not None:
            from owl.record.cadc_capture import capture_ingestion_execution

            transfer_all = float(cfg.predation.resource_transfer) * target_resource_all
            capture_ingestion_execution(
                cadc_buffer,
                ds,
                py,
                px,
                iy,
                ix,
                eligible,
                success,
                probability,
                transfer_all,
            )
        _record_ingestion_events_gpu(
            ds,
            cfg,
            py,
            px,
            iy,
            ix,
            eligible,
            probability,
            success,
            target_resource_all,
        )
        si = xp.nonzero(success)[0]
        fi = xp.nonzero(failure)[0]
        resource = ds.resource.copy()
        memory = ds.memory.copy()
        food = ds.food.copy()
        signal_emission = ds.signal_emission.copy()
        if int(si.shape[0]):
            spy, spx, siy, six = py[si], px[si], iy[si], ix[si]
            target_resource = target_resource_all[si]
            transfer = float(cfg.predation.resource_transfer) * target_resource
            xp.add.at(resource, (spy, spx), transfer)
            resource = xp.minimum(resource, float(cfg.resources.max_resource))
            xp.add.at(
                memory, (spy, spx), float(cfg.predation.memory_transfer) * ds.memory[siy, six]
            )
            memory = xp.clip(memory, 0.0, 1.0)
            xp.add.at(food, (siy, six), 0.20 * target_resource)
            food = xp.clip(food, 0.0, 1.0)
            distress = int(SignalChannel.DISTRESS)
            if bool(cfg.communication.enabled) and distress < signal_emission.shape[-1]:
                xp.add.at(signal_emission, (siy, six, xp.full_like(siy, distress)), 0.10)
                signal_emission = xp.clip(signal_emission, 0.0, 1.0)
            dead = xp.zeros_like(ds.health, dtype=bool)
            dead[siy, six] = True
            write_array(ds, "resource", resource)
            write_array(ds, "memory", memory)
            write_array(ds, "food", food)
            write_array(ds, "signal_emission", signal_emission)
            clear_cell_gpu(ds, dead)
            # clear_cell_gpu rewrites the arrays above only at targets; refresh
            # local references for any following failed-attack deltas.
            health = ds.health.copy()
            boundary = ds.boundary.copy()
            resource = ds.resource.copy()
        if int(fi.shape[0]):
            fpy, fpx, fiy, fix = py[fi], px[fi], iy[fi], ix[fi]
            xp.add.at(resource, (fpy, fpx), -0.5 * float(cfg.resources.movement_cost))
            xp.add.at(health, (fpy, fpx), -0.03)
            xp.add.at(boundary, (fpy, fpx), -0.02)
            xp.add.at(health, (fiy, fix), -0.01)
        write_array(ds, "resource", xp.clip(resource, 0.0, float(cfg.resources.max_resource)))
        write_array(ds, "health", xp.clip(health, 0.0, 1.0))
        write_array(ds, "boundary", xp.clip(boundary, 0.0, 1.0))
        success_count = xp.sum(success)
    else:
        write_array(ds, "health", health)
        write_array(ds, "boundary", boundary)
    return {
        "collisions": n,
        "ordinary": metric_int(ds, xp.sum(ordinary)),
        "ingestion_success": metric_int(ds, success_count),
    }


def apply_inhibition_gpu(ds: Any, cfg: Any) -> dict[str, Any]:
    xp = ds.xp
    alive = (ds.health > 0) & (~ds.obstacle)
    inhibit = (ds.readout == int(Action.INHIBIT)) & alive
    strength = (
        inhibit.astype(xp.float32)
        * xp.clip(ds.aggression + ds.integration + ds.cooperation, 0.0, 3.0)
        / 3.0
    )
    pressure = neighbor_sum_8(strength, xp, "toroidal") / 8.0
    write_array(ds, "activation", xp.clip(ds.activation - 0.08 * pressure, 0.0, 1.0))
    write_array(ds, "integration", xp.clip(ds.integration - 0.04 * pressure, 0.0, 1.0))
    write_array(ds, "boundary", xp.clip(ds.boundary - 0.01 * pressure, 0.0, 1.0))
    write_array(
        ds,
        "resource",
        xp.clip(
            ds.resource - xp.where(inhibit, 0.5 * float(cfg.resources.movement_cost), 0.0),
            0.0,
            float(cfg.resources.max_resource),
        ),
    )
    idx = int(SignalChannel.THREAT)
    if bool(cfg.communication.enabled) and idx < ds.signal_emission.shape[-1]:
        out = ds.signal_emission.copy()
        out[..., idx] = xp.clip(out[..., idx] + 0.10 * strength, 0.0, 1.0)
        write_array(ds, "signal_emission", out)
    return {"inhibitors": metric_int(ds, xp.sum(inhibit))}
