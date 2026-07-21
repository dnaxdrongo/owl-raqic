"""CuPy/NumPy device wrappers for completed high-level action contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from owl.core.actions import Action
from owl.gpu.array_write import write_array
from owl.science.action_transitions import (
    ActionTransitionContext,
    compile_selected_execution_action,
    resolve_action_transition_context,
    summarize_radius_fields,
)


@dataclass(frozen=True)
class ActiveSenseDeviceResult:
    attempted: Any
    success: Any
    no_new_information: Any
    cost: Any
    newly_observed_count: Any
    newly_observed_target_count: Any
    memory_changed: Any


def action_transition_context_from_device(ds: Any) -> ActionTransitionContext:
    """Return a zero-copy view of the persistent resolver evidence."""
    return ActionTransitionContext(
        target_y=ds.action_target_y,
        target_x=ds.action_target_x,
        target_ow_id=ds.action_target_ow_id,
        target_kind=ds.action_target_kind,
        target_source=ds.action_target_source,
        target_distance=ds.action_target_distance,
        target_confidence=ds.action_target_confidence,
        direction_y=ds.action_direction_y,
        direction_x=ds.action_direction_x,
        direction_executable=ds.action_direction_executable,
        direction_score=ds.action_direction_score,
        direction_distance_delta=ds.action_direction_distance_delta,
        direction_hazard=ds.action_direction_hazard,
        direction_opportunity=ds.action_direction_opportunity,
        flee_compiled_action=ds.flee_compiled_action,
        pursue_compiled_action=ds.pursue_compiled_action,
        flee_executable=ds.flee_compiled_action >= 0,
        pursue_executable=ds.pursue_compiled_action >= 0,
    )


def prepare_action_transition_context_gpu(ds: Any, cfg: Any) -> Any:
    if not bool(cfg.action_transitions.enabled):
        return None
    context = resolve_action_transition_context(
        health=ds.health,
        resource=ds.resource,
        obstacle=ds.obstacle,
        occupancy=ds.occupancy,
        food=ds.food,
        toxin=ds.toxin,
        predation=ds.predation,
        aggression=ds.aggression,
        mobility=ds.mobility,
        cfg=cfg,
        xp=ds.xp,
    )
    for name in (
        "target_y",
        "target_x",
        "target_ow_id",
        "target_kind",
        "target_source",
        "target_distance",
        "target_confidence",
        "direction_y",
        "direction_x",
        "direction_executable",
        "direction_score",
        "direction_distance_delta",
        "direction_hazard",
        "direction_opportunity",
        "flee_compiled_action",
        "pursue_compiled_action",
    ):
        field = name if name.startswith(("flee_", "pursue_")) else f"action_{name}"
        write_array(ds, field, getattr(context, name))
    return context


def compile_selected_action_transition_gpu(ds: Any, cfg: Any) -> None:
    if not bool(cfg.action_transitions.enabled):
        return
    write_array(
        ds,
        "compiled_execution_action",
        compile_selected_execution_action(
            ds.readout, ds.flee_compiled_action, ds.pursue_compiled_action, xp=ds.xp
        ),
    )
    write_array(ds, "active_sense_ttl", ds.xp.maximum(ds.active_sense_ttl - 1, 0))


def _radius_summary(ds: Any, radius: int, cfg: Any) -> tuple[Any, ...]:
    return summarize_radius_fields(
        health=ds.health,
        obstacle=ds.obstacle,
        food=ds.food,
        toxin=ds.toxin,
        radius=radius,
        threat_threshold=float(cfg.action_transitions.perceived_threat_threshold),
        boundary_mode=str(cfg.world.boundary_mode),
        xp=ds.xp,
    )


def apply_active_sense_transition_gpu(ds: Any, cfg: Any) -> ActiveSenseDeviceResult:
    xp = ds.xp
    zeros_b = xp.zeros(ds.health.shape, dtype=bool)
    zeros_f = xp.zeros(ds.health.shape, dtype=ds.health.dtype)
    zeros_i = xp.zeros(ds.health.shape, dtype=xp.int32)
    if not bool(cfg.action_transitions.enabled and cfg.action_transitions.active_sense_enabled):
        return ActiveSenseDeviceResult(
            zeros_b, zeros_b, zeros_b, zeros_f, zeros_i, zeros_i, zeros_b
        )
    attempted = (ds.readout == int(Action.SENSE)) & (ds.health > 0.0) & (~ds.obstacle)
    cost_value = float(cfg.action_transitions.active_sense_cost)
    success = attempted & (ds.resource >= cost_value)
    ordinary_radius = int(cfg.action_transitions.active_sense_ordinary_radius)
    active_radius = ordinary_radius + int(cfg.action_transitions.active_sense_radius_bonus)
    ordinary = _radius_summary(ds, ordinary_radius, cfg)
    enhanced = _radius_summary(ds, active_radius, cfg)
    newly_observed = xp.maximum(enhanced[3] - ordinary[3], 0)
    newly_targets = xp.maximum(enhanced[4] - ordinary[4], 0)
    before_food = ds.active_sense_food_memory.copy()
    before_toxin = ds.active_sense_toxin_memory.copy()
    write_array(
        ds,
        "active_sense_food_memory",
        xp.where(success, enhanced[0], ds.active_sense_food_memory),
    )
    write_array(
        ds,
        "active_sense_toxin_memory",
        xp.where(success, enhanced[1], ds.active_sense_toxin_memory),
    )
    write_array(
        ds,
        "active_sense_alive_memory",
        xp.where(success, enhanced[2], ds.active_sense_alive_memory),
    )
    write_array(
        ds,
        "active_sense_ttl",
        xp.where(
            success,
            int(cfg.action_transitions.active_sense_memory_persistence),
            ds.active_sense_ttl,
        ),
    )
    write_array(
        ds,
        "active_sense_new_cell_count",
        xp.where(success, newly_observed, 0).astype(xp.int32),
    )
    write_array(
        ds,
        "active_sense_new_target_count",
        xp.where(success, newly_targets, 0).astype(xp.int32),
    )
    cost = success.astype(ds.health.dtype) * cost_value
    write_array(ds, "resource", xp.clip(ds.resource - cost, 0.0, float(cfg.resources.max_resource)))
    memory_changed = success & (
        (xp.abs(ds.active_sense_food_memory - before_food) > float(cfg.actions.epsilon))
        | (xp.abs(ds.active_sense_toxin_memory - before_toxin) > float(cfg.actions.epsilon))
    )
    no_new = success & (newly_observed <= 0) & (newly_targets <= 0)
    return ActiveSenseDeviceResult(
        attempted,
        success,
        no_new,
        cost,
        xp.where(success, newly_observed, 0).astype(xp.int32),
        xp.where(success, newly_targets, 0).astype(xp.int32),
        memory_changed,
    )
