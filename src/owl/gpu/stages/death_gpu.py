from __future__ import annotations

from typing import Any

from owl.core.actions import Action, SignalChannel
from owl.core.constants import CELL_FIELDS_2D, CELL_FIELDS_3D
from owl.gpu.array_write import write_array
from owl.gpu.stage_metrics import metric_int

_ADVANCED_2D = (
    "digestion",
    "age_stress",
    "last_intake",
    "development_stage",
    "symbiosis",
    "prediction_error",
    "starvation_debt",
    "movement_loop_score",
)
_ADVANCED_ND = (
    "action_cooldown",
    "last_utilities",
    "last_logits",
    "last_action_probabilities",
    "deception_memory",
    "source_confidence",
    "genome",
)

_ACTION_TRANSITION_ZERO = (
    "active_sense_food_memory",
    "active_sense_toxin_memory",
    "active_sense_alive_memory",
    "active_sense_ttl",
    "active_sense_new_cell_count",
    "active_sense_new_target_count",
    "action_target_distance",
    "action_target_confidence",
    "action_direction_executable",
    "action_direction_score",
    "action_direction_distance_delta",
    "action_direction_hazard",
    "action_direction_opportunity",
)
_ACTION_TRANSITION_ABSENT = (
    "flee_compiled_action",
    "pursue_compiled_action",
    "compiled_execution_action",
    "action_target_y",
    "action_target_x",
    "action_target_ow_id",
    "action_target_kind",
    "action_target_source",
    "action_direction_y",
    "action_direction_x",
)


def detect_dead_cells_gpu(ds: Any, cfg: Any) -> Any:
    xp = ds.xp
    present = (
        (ds.occupancy >= 0)
        | (ds.health > 0)
        | (ds.resource > 0)
        | (ds.boundary > 0)
        | (ds.memory > 0)
        | (ds.integration > 0)
    ) & (~ds.obstacle)
    if "starvation_debt" in ds.arrays:
        starvation = (ds.starvation_debt >= 1.0) & (ds.health <= 0.05)
    else:
        starvation = xp.zeros_like(ds.health, dtype=bool)
    dead = present & ((ds.health <= 0) | starvation | (ds.boundary <= 0) | (ds.integration < 0))
    return dead & (~ds.obstacle)


def clear_cell_gpu(ds: Any, dead: Any) -> None:
    xp = ds.xp
    for name in CELL_FIELDS_2D + _ADVANCED_2D:
        if name in ds.arrays and ds.arrays[name].shape == ds.health.shape:
            write_array(ds, name, xp.where(dead, 0, ds.arrays[name]))
    for name in CELL_FIELDS_3D + _ADVANCED_ND:
        if name in ds.arrays and ds.arrays[name].shape[:2] == ds.health.shape:
            write_array(ds, name, xp.where(dead[..., None], 0, ds.arrays[name]))
    if "neighbor_trust" in ds.arrays:
        write_array(ds, "neighbor_trust", xp.where(dead[..., None, None], 1.0, ds.neighbor_trust))
    for names, fill in ((_ACTION_TRANSITION_ZERO, 0), (_ACTION_TRANSITION_ABSENT, -1)):
        for name in names:
            if name not in ds.arrays:
                continue
            array = ds.arrays[name]
            mask = dead
            while mask.ndim < array.ndim:
                mask = mask[..., None]
            write_array(ds, name, xp.where(mask, fill, array))
    for name in ("signal_reception", "signal_emission"):
        if name in ds.arrays:
            write_array(ds, name, xp.where(dead[..., None], 0, ds.arrays[name]))
    if "readout" in ds.arrays:
        write_array(ds, "readout", xp.where(dead, int(Action.REST), ds.readout))
    if "occupancy" in ds.arrays:
        write_array(ds, "occupancy", xp.where(dead, -1, ds.occupancy))
    if "parent_id" in ds.arrays:
        write_array(ds, "parent_id", xp.where(dead, -1, ds.parent_id))
    if "lineage_id" in ds.arrays:
        write_array(ds, "lineage_id", xp.where(dead, -1, ds.lineage_id))
    if "age" in ds.arrays:
        write_array(ds, "age", xp.where(dead, 0, ds.age))
    if "possibility" in ds.arrays:
        poss = xp.where(dead[..., None], 0.0, ds.possibility)
        poss[..., int(Action.REST)] = xp.where(dead, 1.0, poss[..., int(Action.REST)])
        write_array(ds, "possibility", poss)
    if "last_movement_action" in ds.arrays:
        write_array(
            ds, "last_movement_action", xp.where(dead, int(Action.REST), ds.last_movement_action)
        )
    for name in (
        "raqic_readout",
        "raqic_record_action",
        "raqic_record_readout",
        "raqic_legacy_shadow_readout",
    ):
        if name in ds.arrays:
            write_array(ds, name, xp.where(dead, int(Action.REST), ds.arrays[name]))
    for name in (
        "raqic_probabilities",
        "raqic_parent_intention",
        "raqic_debug_density_diag",
        "last_action_probabilities",
        "last_macro_probabilities",
        "raqic_legacy_shadow_possibility",
    ):
        if name in ds.arrays:
            arr = xp.where(dead[..., None], 0.0, ds.arrays[name])
            arr[..., int(Action.REST)] = xp.where(dead, 1.0, arr[..., int(Action.REST)])
            write_array(ds, name, arr)
    for name in ("raqic_score", "raqic_phase"):
        if name in ds.arrays:
            write_array(ds, name, xp.where(dead[..., None], 0.0, ds.arrays[name]))
    for name in (
        "raqic_record_confidence",
        "raqic_trace_error",
        "raqic_min_eigenvalue",
        "raqic_backend_code",
        "raqic_compare_l1",
        "raqic_compare_kl",
    ):
        if name in ds.arrays:
            write_array(ds, name, xp.where(dead, 0, ds.arrays[name]))
    if "raqic_audit_flags" in ds.arrays:
        write_array(ds, "raqic_audit_flags", xp.where(dead[..., None], 0, ds.raqic_audit_flags))


def apply_death_gpu(ds: Any, cfg: Any) -> dict[str, Any]:
    xp = ds.xp
    dead = detect_dead_cells_gpu(ds, cfg)
    cadc_buffer = ds.metadata.get("cadc_device_buffer")
    if cadc_buffer is not None:
        from owl.record.cadc_capture import capture_death_event

        capture_death_event(cadc_buffer, ds, dead)
    if "last_death_mask" in ds.arrays:
        write_array(ds, "last_death_mask", dead)
    residue = 0.20 * xp.clip(ds.resource, 0, float(cfg.resources.max_resource)) / max(
        float(cfg.resources.max_resource), float(cfg.actions.epsilon)
    ) + 0.05 * xp.clip(ds.boundary, 0, 1)
    write_array(ds, "food", xp.clip(ds.food + xp.where(dead, residue, 0.0), 0, 1))
    idx = int(SignalChannel.DISTRESS)
    if bool(cfg.communication.enabled) and idx < ds.signal_emission.shape[-1]:
        out = ds.signal_emission.copy()
        out[..., idx] = xp.clip(out[..., idx] + 0.10 * dead.astype(out.dtype), 0, 1)
        write_array(ds, "signal_emission", out)
    count = metric_int(ds, xp.sum(dead))
    clear_cell_gpu(ds, dead)
    return {"dead_count": count}
