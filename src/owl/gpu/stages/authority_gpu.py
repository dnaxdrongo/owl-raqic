from __future__ import annotations

from typing import Any

from owl.core.actions import DIAGONAL_MOVES, MOVE_DELTAS, Action
from owl.gpu.array_write import write_array
from owl.gpu.stencil import neighbor_sum_8


def _neighbor_mean(x: Any, xp: Any) -> Any:
    return neighbor_sum_8(x, xp, "toroidal") / 8.0


def compute_authority_gpu(ds: Any, cfg: Any) -> None:
    """Backend-neutral translation of ``owl.engine.authority.compute_authority``."""
    xp = ds.xp
    h, w = ds.health.shape
    actions = int(ds.possibility.shape[-1])
    dtype = ds.health.dtype
    auth = xp.zeros((h, w, actions), dtype=dtype)
    alive = (ds.health > 0.0) & (~ds.obstacle)
    alive_f = alive.astype(dtype)
    resource = xp.clip(ds.resource, 0.0, float(cfg.resources.max_resource))
    max_resource = max(float(cfg.resources.max_resource), float(cfg.actions.epsilon))
    resource_norm = xp.clip(resource / max_resource, 0.0, 1.0)
    health = xp.clip(ds.health, 0.0, 1.0)
    boundary = xp.clip(ds.boundary, 0.0, 1.0)
    integration = xp.clip(ds.integration, 0.0, 1.0)

    auth[..., int(Action.REST)] = 1.0
    if bool(cfg.action_transitions.enabled):
        auth[..., int(Action.SENSE)] = (
            alive_f
            * bool(cfg.action_transitions.active_sense_enabled)
            * (resource >= float(cfg.action_transitions.active_sense_cost)).astype(dtype)
        )
    else:
        auth[..., int(Action.SENSE)] = alive_f
    enough_move = (resource >= float(cfg.resources.movement_cost)).astype(dtype)
    mobility = xp.clip(ds.mobility, 0.0, 1.0)
    for action in MOVE_DELTAS:
        value = alive_f * mobility * enough_move
        if action in DIAGONAL_MOVES and not bool(cfg.actions.diagonal_movement_enabled):
            value = xp.zeros_like(value)
        auth[..., int(action)] = value

    food_neighbor = ds.arrays.get("food_mean")
    if food_neighbor is None:
        food_neighbor = _neighbor_mean(ds.food, xp)
    local_food = xp.clip(0.5 * ds.food + 0.5 * food_neighbor, 0.0, 1.0)
    food_threshold = 0.001 if bool(getattr(cfg.decision_homeostasis, "enabled", False)) else 0.005
    auth[..., int(Action.FEED)] = (
        alive_f * xp.clip(ds.grazing, 0.0, 1.0) * (local_food > food_threshold).astype(dtype)
    )
    if bool(cfg.communication.enabled):
        afford = (resource >= float(cfg.communication.base_emit_cost)).astype(dtype)
        emit = xp.clip(ds.emit_strength, 0.0, 1.0) * xp.clip(ds.emit_efficiency, 0.0, 1.0)
        auth[..., int(Action.COMMUNICATE)] = alive_f * afford * emit
    auth[..., int(Action.INHIBIT)] = alive_f * xp.maximum(
        xp.clip(ds.aggression, 0.0, 1.0), 0.5 * xp.clip(ds.cooperation, 0.0, 1.0)
    )
    auth[..., int(Action.INTEGRATE)] = alive_f * xp.maximum(
        0.15, xp.clip(ds.coupling_strength, 0.0, 1.0)
    )
    auth[..., int(Action.REPAIR)] = alive_f * (resource > 0.03).astype(dtype)
    if bool(cfg.reproduction.enabled):
        auth[..., int(Action.REPRODUCE)] = (
            alive_f
            * (resource_norm >= float(cfg.reproduction.min_resource)).astype(dtype)
            * (health >= float(cfg.reproduction.min_health)).astype(dtype)
            * (boundary >= float(cfg.reproduction.min_boundary)).astype(dtype)
            * (integration >= float(cfg.reproduction.min_integration)).astype(dtype)
            * xp.clip(ds.reproduction_rate, 0.0, 1.0)
        )
    adjacent_life = ds.arrays.get("alive_density")
    if adjacent_life is None:
        adjacent_life = _neighbor_mean(alive_f, xp)
    if bool(cfg.predation.enabled):
        auth[..., int(Action.INGEST)] = (
            alive_f
            * (xp.clip(ds.predation, 0.0, 1.0) >= float(cfg.predation.min_predation_trait)).astype(
                dtype
            )
            * (resource > 0.05).astype(dtype)
            * (adjacent_life > 0.0).astype(dtype)
        )
    auth[..., int(Action.EXPEL)] = 0.0
    auth[..., int(Action.SPLIT)] = 0.0
    auth[..., int(Action.MERGE)] = 0.0
    danger = xp.clip(ds.toxin + xp.mean(xp.clip(ds.signal_reception, 0.0, 1.0), axis=-1), 0.0, 1.0)
    if bool(cfg.action_transitions.enabled):
        auth[..., int(Action.FLEE)] = (
            alive_f
            * mobility
            * enough_move
            * bool(cfg.action_transitions.flee_execution_enabled)
            * (ds.flee_compiled_action >= 0).astype(dtype)
        )
        auth[..., int(Action.PURSUE)] = (
            alive_f
            * mobility
            * enough_move
            * bool(cfg.action_transitions.pursue_execution_enabled)
            * (ds.pursue_compiled_action >= 0).astype(dtype)
        )
    else:
        auth[..., int(Action.FLEE)] = alive_f * mobility * (danger > 0.02).astype(dtype)
        auth[..., int(Action.PURSUE)] = (
            alive_f
            * xp.clip(ds.predation + ds.aggression, 0.0, 1.0)
            * (adjacent_life > 0.0).astype(dtype)
        )

    if bool(getattr(cfg.cross_scale_homeostasis, "enabled", False)):
        # Device-native reproduction viability equivalent to CPU gate.
        def smooth(x: Any, a: Any, b: Any) -> Any:
            t = xp.clip((x - a) / max(b - a, 1e-8), 0.0, 1.0)
            return t * t * (3.0 - 2.0 * t)

        density = xp.clip(_neighbor_mean(alive_f, xp), 0.0, 1.0)
        food = xp.clip(_neighbor_mean(xp.clip(ds.food, 0.0, 1.0), xp), 0.0, 1.0)
        crisis = ds.arrays.get("patch_crisis", xp.zeros_like(density))
        carrying = ds.arrays.get("patch_carrying_pressure", xp.zeros_like(density))
        capacity = xp.clip(
            1.0 - 0.65 * density - 0.75 * carrying - 0.75 * crisis + 0.40 * food, 0.0, 1.0
        )
        viability = (
            smooth(resource_norm, 0.45, 0.85)
            * smooth(health, 0.60, 0.95)
            * smooth(boundary, 0.60, 0.95)
            * capacity
        )
        auth[..., int(Action.REPRODUCE)] *= xp.clip(viability, 0.0, 1.0)

    auth = xp.clip(auth, 0.0, 1.0)
    enabled = list(cfg.actions.enabled_actions)
    if enabled:
        mask = xp.zeros((actions,), dtype=dtype)
        for name in enabled:
            key = str(name).upper()
            if key not in Action.__members__:
                raise ValueError(f"unknown enabled action {name!r}")
            mask[int(Action[key])] = 1.0
        mask[int(Action.REST)] = 1.0
        auth *= mask[None, None, :]
    dead = ~alive
    auth = xp.where(dead[..., None], 0.0, auth)
    auth[..., int(Action.REST)] = xp.where(dead, 1.0, auth[..., int(Action.REST)])
    auth = xp.clip(auth, 0.0, 1.0).astype(dtype)
    write_array(ds, "pre_authority", auth)
    write_array(ds, "_authority_bool", auth > 0.0)
    policy_legal = auth > 0.0
    if bool(cfg.action_transitions.enabled):
        configured = {str(name).upper() for name in cfg.actions.enabled_actions}
        for action in (Action.SENSE, Action.FLEE, Action.PURSUE):
            globally_enabled = not configured or action.name in configured
            policy_legal[..., int(action)] = alive & globally_enabled
    write_array(ds, "_policy_legal_bool", policy_legal)
