from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from owl.core.actions import Action, EventKind
from owl.core.state import EventRecord
from owl.gpu.array_write import write_array
from owl.science.action_contract import movement_plan

_MOVE_2D = (
    "activation",
    "memory",
    "phase",
    "threshold",
    "integration",
    "health",
    "boundary",
    "age",
    "ow_type",
    "lineage_id",
    "parent_id",
    "mobility",
    "metabolism",
    "predation",
    "grazing",
    "cooperation",
    "aggression",
    "curiosity",
    "reproduction_rate",
    "toxin_resistance",
    "memory_capacity",
    "coupling_strength",
    "emit_strength",
    "emit_efficiency",
    "receive_sensitivity",
    "signal_precision",
    "honesty_bias",
    "deception_bias",
    "digestion",
    "waste",
    "age_stress",
    "last_intake",
    "prediction_error",
    "starvation_debt",
    "movement_loop_score",
    "development_stage",
    "symbiosis",
    "phase_frequency",
    "phase_lag",
    "parent_weight",
    "pre_resource",
    "pre_health",
    "pre_food",
    "pre_starvation_debt",
    "last_decision_urgency",
    "last_homeostatic_error",
    "noetic_B",
    "noetic_M",
    "noetic_P",
    "noetic_C",
    "noetic_K",
    "noetic_Theta",
    "noetic_N",
    "active_sense_food_memory",
    "active_sense_toxin_memory",
    "active_sense_alive_memory",
    "active_sense_ttl",
    "active_sense_new_cell_count",
    "active_sense_new_target_count",
    "flee_compiled_action",
    "pursue_compiled_action",
    "compiled_execution_action",
    "raqic_readout",
    "raqic_record_action",
    "raqic_record_readout",
    "raqic_record_confidence",
    "raqic_audit_flags",
    "raqic_trace_error",
    "raqic_min_eigenvalue",
    "raqic_backend_code",
    "raqic_legacy_shadow_readout",
    "raqic_compare_l1",
    "raqic_compare_kl",
)
_MOVE_ND = (
    "possibility",
    "channel_receptivity",
    "channel_emission_bias",
    "channel_trust_local",
    "signal_memory",
    "last_utilities",
    "last_logits",
    "last_action_probabilities",
    "action_cooldown",
    "pre_authority",
    "pre_utilities",
    "pre_parent_bias",
    "last_survival_value",
    "last_macro_probabilities",
    "deception_memory",
    "source_confidence",
    "neighbor_trust",
    "same_scale_weight",
    "genome",
    "raqic_probabilities",
    "raqic_score",
    "raqic_phase",
    "raqic_parent_intention",
    "raqic_legacy_shadow_possibility",
    "raqic_debug_density_diag",
    "action_target_y",
    "action_target_x",
    "action_target_ow_id",
    "action_target_kind",
    "action_target_source",
    "action_target_distance",
    "action_target_confidence",
    "action_direction_y",
    "action_direction_x",
    "action_direction_executable",
    "action_direction_score",
    "action_direction_distance_delta",
    "action_direction_hazard",
    "action_direction_opportunity",
)


def _action_deltas(xp: Any, action_count: int) -> Any:
    """Return dense action-indexed movement deltas for graph-static callers."""
    dy = xp.zeros((action_count,), dtype=xp.int32)
    dx = xp.zeros((action_count,), dtype=xp.int32)
    mapping = {
        Action.MOVE_N: (-1, 0),
        Action.MOVE_S: (1, 0),
        Action.MOVE_E: (0, 1),
        Action.MOVE_W: (0, -1),
        Action.MOVE_NE: (-1, 1),
        Action.MOVE_NW: (-1, -1),
        Action.MOVE_SE: (1, 1),
        Action.MOVE_SW: (1, -1),
    }
    for action, (yy, xx) in mapping.items():
        if int(action) < action_count:
            dy[int(action)] = yy
            dx[int(action)] = xx
    return dy, dx


def _host_array(ds: Any, value: Any) -> Any:
    """Return host NumPy data for a backend array."""
    return ds.backend.asnumpy(value)


def _record_movement_collision_events_gpu(ds: Any, cfg: Any, plan: Any) -> None:
    """Record CPU-compatible sparse collision events for GPU movement."""
    max_events = int(getattr(getattr(cfg, "recording", object()), "max_events", 4096))
    queue: list[EventRecord] = []
    ds.metadata["event_queue"] = queue
    if max_events <= 0:
        return

    xp = ds.xp
    idx = xp.nonzero(plan.collision)[0]
    if int(idx.shape[0]) == 0:
        return

    idx = idx[:max_events]
    sy_h = _host_array(ds, plan.mover_y[idx]).reshape(-1)
    sx_h = _host_array(ds, plan.mover_x[idx]).reshape(-1)
    ty_h = _host_array(ds, plan.target_y[idx]).reshape(-1)
    tx_h = _host_array(ds, plan.target_x[idx]).reshape(-1)
    priority_h = _host_array(ds, plan.priority[idx]).reshape(-1)

    for i in range(len(sy_h)):
        queue.append(
            EventRecord(
                kind=str(EventKind.COLLISION),
                tick=int(ds.tick),
                source=(int(sy_h[i]), int(sx_h[i])),
                target=(int(ty_h[i]), int(tx_h[i])),
                payload={
                    "simultaneous": True,
                    "priority": int(priority_h[i]),
                },
            )
        )


def _debug_dump_movement_plan_gpu(ds: Any, cfg: Any, plan: Any) -> None:
    """Dump exact movement-stage input and GPU movement plan for parity diagnosis."""
    if os.environ.get("OWL_V0952_DEBUG_MOVEMENT_PLAN") != "1":
        return

    out_dir = Path("reports/v0951_b200_smoke/debug_movement_plan")
    out_dir.mkdir(parents=True, exist_ok=True)

    def host(value: Any) -> Any:
        return ds.backend.asnumpy(value) if hasattr(value, "shape") else value

    tick = int(ds.tick)
    device_tick = host(ds.arrays.get("_device_tick", ds.xp.asarray(tick)))
    try:
        device_tick_int = int(np.asarray(device_tick).reshape(-1)[0])
    except Exception:
        device_tick_int = tick

    np.savez_compressed(
        out_dir / f"movement_input_tick_{tick}.npz",
        readout=host(ds.readout),
        health=host(ds.health),
        obstacle=host(ds.obstacle),
        occupancy=host(ds.occupancy),
        tick=np.asarray([tick], dtype=np.int64),
        device_tick=np.asarray([device_tick_int], dtype=np.int64),
    )

    np.savez_compressed(
        out_dir / f"movement_gpu_plan_tick_{tick}.npz",
        mover_y=host(plan.mover_y),
        mover_x=host(plan.mover_x),
        target_y=host(plan.target_y),
        target_x=host(plan.target_x),
        accepted=host(plan.accepted),
        blocked=host(plan.blocked),
        collision=host(plan.collision),
        priority=host(plan.priority),
    )


def apply_movement_gpu(ds: Any, cfg: Any) -> dict[str, Any]:
    xp = ds.xp
    ds.metadata["event_queue"] = []
    execution_readout = (
        ds.compiled_execution_action
        if bool(cfg.action_transitions.enabled) and "compiled_execution_action" in ds.arrays
        else ds.readout
    )
    plan = movement_plan(
        execution_readout,
        ds.health,
        ds.obstacle,
        ds.occupancy,
        boundary_mode=str(cfg.world.boundary_mode),
        seed=int(cfg.world.seed),
        tick=ds.arrays.get("_device_tick", int(ds.tick)),
        xp=xp,
    )
    cadc_buffer = ds.metadata.get("cadc_device_buffer")
    if cadc_buffer is not None:
        from owl.record.cadc_capture import capture_movement_execution

        capture_movement_execution(cadc_buffer, ds, cfg, plan)
    _debug_dump_movement_plan_gpu(ds, cfg, plan)
    n = int(plan.mover_y.shape[0])
    if n == 0:
        write_array(ds, "_collision_source_y", xp.zeros((0,), dtype=xp.int32))
        write_array(ds, "_collision_source_x", xp.zeros((0,), dtype=xp.int32))
        write_array(ds, "_collision_target_y", xp.zeros((0,), dtype=xp.int32))
        write_array(ds, "_collision_target_x", xp.zeros((0,), dtype=xp.int32))
        return {"moved": 0, "collisions": 0}
    if not bool(ds.metadata.get("counterfactual_suppress_host_event_queue", False)):
        _record_movement_collision_events_gpu(ds, cfg, plan)
    sy, sx, ty, tx = plan.mover_y, plan.mover_x, plan.target_y, plan.target_x
    failed = ~plan.accepted
    resource = ds.resource.copy()
    resource[sy[failed], sx[failed]] -= 0.5 * float(cfg.resources.movement_cost)
    keep = xp.nonzero(plan.accepted)[0]
    if int(keep.shape[0]):
        ay, ax, by, bx = sy[keep], sx[keep], ty[keep], tx[keep]
        for name in _MOVE_2D:
            if name not in ds.arrays:
                continue
            arr = ds.arrays[name]
            if arr.shape != ds.health.shape:
                continue
            out = arr.copy()
            vals = arr[ay, ax].copy()
            out[ay, ax] = 0
            out[by, bx] = vals
            write_array(ds, name, out)
        for name in _MOVE_ND:
            if name not in ds.arrays:
                continue
            arr = ds.arrays[name]
            if arr.shape[:2] != ds.health.shape:
                continue
            out = arr.copy()
            vals = arr[ay, ax, ...].copy()
            out[ay, ax, ...] = 0
            out[by, bx, ...] = vals
            write_array(ds, name, out)
        occ = ds.occupancy.copy()
        old = ds.occupancy[ay, ax].copy()
        occ[ay, ax] = -1
        occ[by, bx] = xp.where(
            old >= 0, old, by.astype(old.dtype) * ds.health.shape[1] + bx.astype(old.dtype)
        )
        write_array(ds, "occupancy", occ)
        read = ds.readout.copy()
        oldread = ds.readout[ay, ax].copy()
        read[ay, ax] = int(Action.REST)
        read[by, bx] = oldread
        write_array(ds, "readout", read)
        for name in (
            "raqic_readout",
            "raqic_record_action",
            "raqic_record_readout",
            "raqic_legacy_shadow_readout",
        ):
            if name in ds.arrays:
                arr = ds.arrays[name].copy()
                arr[ay, ax] = int(Action.REST)
                write_array(ds, name, arr)
        for name in ("raqic_probabilities", "raqic_legacy_shadow_possibility"):
            if name in ds.arrays:
                arr = ds.arrays[name].copy()
                arr[ay, ax, :] = 0.0
                arr[ay, ax, int(Action.REST)] = 1.0
                write_array(ds, name, arr)
        parent = ds.parent_id.copy()
        parent[ay, ax] = -1
        ph, pw = ds.patch_arrays["integration"].shape
        psy = ds.health.shape[0] // ph
        psx = ds.health.shape[1] // pw
        parent[by, bx] = (by // psy) * pw + (bx // psx)
        write_array(ds, "parent_id", parent)
        lineage = ds.lineage_id.copy()
        lineage[ay, ax] = -1
        write_array(ds, "lineage_id", lineage)
        age = ds.age.copy()
        age[ay, ax] = 0
        write_array(ds, "age", age)
        poss = ds.possibility.copy()
        poss[ay, ax, :] = 0.0
        poss[ay, ax, int(Action.REST)] = 1.0
        write_array(ds, "possibility", poss)
        if "last_movement_action" in ds.arrays:
            lm = ds.last_movement_action.copy()
            lm[ay, ax] = int(Action.REST)
            # Movement memory follows the authoritative selected identity.
            # The compiled primitive remains separately available in
            # ``compiled_execution_action`` and recorder execution evidence.
            lm[by, bx] = oldread.astype(lm.dtype)
            write_array(ds, "last_movement_action", lm)
        moved_resource = resource[ay, ax].copy()
        resource[ay, ax] = 0.0
        resource[by, bx] = moved_resource - float(cfg.resources.movement_cost)
    write_array(ds, "resource", xp.clip(resource, 0.0, float(cfg.resources.max_resource)))
    ci = xp.nonzero(plan.collision)[0]
    write_array(ds, "_collision_source_y", sy[ci].astype(xp.int32))
    write_array(ds, "_collision_source_x", sx[ci].astype(xp.int32))
    write_array(ds, "_collision_target_y", ty[ci].astype(xp.int32))
    write_array(ds, "_collision_target_x", tx[ci].astype(xp.int32))
    return {"moved": int(keep.shape[0]), "collisions": int(ci.shape[0])}


def propose_movements_gpu(ds: Any, cfg: Any) -> Any:
    p = movement_plan(
        ds.readout,
        ds.health,
        ds.obstacle,
        ds.occupancy,
        boundary_mode=str(cfg.world.boundary_mode),
        seed=int(cfg.world.seed),
        tick=ds.arrays.get("_device_tick", int(ds.tick)),
        xp=ds.xp,
    )
    return p.accepted, p.target_y, p.target_x


def resolve_movement_conflicts_gpu(ds: Any, moving: Any, ty: Any, tx: Any) -> Any:
    xp = ds.xp
    sy, sx = xp.nonzero(moving)
    return (
        sy,
        sx,
        ty[sy, sx] if ty.ndim == 2 else ty[moving],
        tx[sy, sx] if tx.ndim == 2 else tx[moving],
    )
