"""CPU ownership wrappers for the versioned action-transition contract."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from owl.core.actions import Action
from owl.core.advanced import ensure_action_transition_fields
from owl.core.state import WorldState
from owl.science.action_transitions import (
    ActionTransitionContext,
    compile_selected_execution_action,
    resolve_action_transition_context,
    summarize_radius_fields,
)


@dataclass(frozen=True)
class ActiveSenseResult:
    attempted: np.ndarray
    success: np.ndarray
    no_new_information: np.ndarray
    cost: np.ndarray
    newly_observed_count: np.ndarray
    newly_observed_target_count: np.ndarray
    memory_changed: np.ndarray


def prepare_action_transition_context(
    state: WorldState, cfg: object
) -> ActionTransitionContext | None:
    """Populate agent-visible targets only when the action-transition contract is enabled."""
    if not bool(cfg.action_transitions.enabled):
        return None
    ensure_action_transition_fields(state, cfg)
    context = resolve_action_transition_context(
        health=state.health,
        resource=state.resource,
        obstacle=state.obstacle,
        occupancy=state.occupancy,
        food=state.food,
        toxin=state.toxin,
        predation=state.predation,
        aggression=state.aggression,
        mobility=state.mobility,
        cfg=cfg,
        xp=np,
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
        destination = getattr(state, f"action_{name}", None)
        if name in {"flee_compiled_action", "pursue_compiled_action"}:
            destination = getattr(state, name)
        assert isinstance(destination, np.ndarray)
        destination[...] = getattr(context, name)
    return context


def compile_selected_action_transition(state: WorldState, cfg: object) -> None:
    """Compile physical primitives after selection while preserving readout."""
    if not bool(cfg.action_transitions.enabled):
        return
    ensure_action_transition_fields(state, cfg)
    assert state.compiled_execution_action is not None
    assert state.flee_compiled_action is not None
    assert state.pursue_compiled_action is not None
    state.compiled_execution_action[...] = compile_selected_execution_action(
        state.readout, state.flee_compiled_action, state.pursue_compiled_action, xp=np
    )
    assert state.active_sense_ttl is not None
    state.active_sense_ttl[...] = np.maximum(state.active_sense_ttl - 1, 0)


def _radius_summary(state: WorldState, radius: int, cfg: object) -> tuple[np.ndarray, ...]:
    return summarize_radius_fields(
        health=state.health,
        obstacle=state.obstacle,
        food=state.food,
        toxin=state.toxin,
        radius=radius,
        threat_threshold=float(cfg.action_transitions.perceived_threat_threshold),
        boundary_mode=str(cfg.world.boundary_mode),
        xp=np,
    )


def apply_active_sense_transition(state: WorldState, cfg: object) -> ActiveSenseResult:
    """Charge and persist enhanced sensing after the current choice."""
    shape = state.health.shape
    zeros_b = np.zeros(shape, dtype=bool)
    zeros_f = np.zeros(shape, dtype=np.float32)
    zeros_i = np.zeros(shape, dtype=np.int32)
    if not bool(cfg.action_transitions.enabled and cfg.action_transitions.active_sense_enabled):
        return ActiveSenseResult(zeros_b, zeros_b, zeros_b, zeros_f, zeros_i, zeros_i, zeros_b)
    ensure_action_transition_fields(state, cfg)
    attempted = (state.readout == int(Action.SENSE)) & (state.health > 0.0) & (~state.obstacle)
    cost_value = float(cfg.action_transitions.active_sense_cost)
    success = attempted & (state.resource >= cost_value)
    ordinary_radius = int(cfg.action_transitions.active_sense_ordinary_radius)
    active_radius = ordinary_radius + int(cfg.action_transitions.active_sense_radius_bonus)
    ordinary = _radius_summary(state, ordinary_radius, cfg)
    enhanced = _radius_summary(state, active_radius, cfg)
    newly_observed = np.maximum(enhanced[3] - ordinary[3], 0)
    newly_targets = np.maximum(enhanced[4] - ordinary[4], 0)
    assert state.active_sense_food_memory is not None
    assert state.active_sense_toxin_memory is not None
    assert state.active_sense_alive_memory is not None
    assert state.active_sense_ttl is not None
    assert state.active_sense_new_cell_count is not None
    assert state.active_sense_new_target_count is not None
    before_food = state.active_sense_food_memory.copy()
    before_toxin = state.active_sense_toxin_memory.copy()
    for destination, value in (
        (state.active_sense_food_memory, enhanced[0]),
        (state.active_sense_toxin_memory, enhanced[1]),
        (state.active_sense_alive_memory, enhanced[2]),
    ):
        destination[success] = value[success].astype(destination.dtype, copy=False)
    state.active_sense_ttl[success] = int(cfg.action_transitions.active_sense_memory_persistence)
    state.active_sense_new_cell_count[success] = newly_observed[success]
    state.active_sense_new_target_count[success] = newly_targets[success]
    cost = success.astype(np.float32) * np.float32(cost_value)
    state.resource -= cost.astype(state.resource.dtype, copy=False)
    np.clip(state.resource, 0.0, cfg.resources.max_resource, out=state.resource)
    memory_changed = success & (
        (np.abs(state.active_sense_food_memory - before_food) > cfg.actions.epsilon)
        | (np.abs(state.active_sense_toxin_memory - before_toxin) > cfg.actions.epsilon)
    )
    no_new = success & (newly_observed <= 0) & (newly_targets <= 0)
    return ActiveSenseResult(
        attempted,
        success,
        no_new,
        cost,
        np.where(success, newly_observed, 0).astype(np.int32),
        np.where(success, newly_targets, 0).astype(np.int32),
        memory_changed,
    )
