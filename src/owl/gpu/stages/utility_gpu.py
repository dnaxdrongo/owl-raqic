from __future__ import annotations

import math
from typing import Any

from owl.core.actions import DIAGONAL_MOVES, MOVE_DELTAS, REVERSE_MOVE_ACTION, Action, SignalChannel
from owl.gpu.array_write import write_array
from owl.gpu.stencil import central_gradient, neighbor_sum_8, shift_2d


def _neighbor_mean(x: Any, xp: Any) -> Any:
    return neighbor_sum_8(x, xp, "toroidal") / 8.0


def _gradient(x: Any, xp: Any) -> tuple[Any, Any]:
    return central_gradient(x, xp, "toroidal")


def _channel(ds: Any, cfg: Any, channel: SignalChannel) -> Any:
    xp = ds.xp
    idx = int(channel)
    if "signal_reception" not in ds.arrays or idx >= min(
        int(cfg.communication.num_channels), int(ds.signal_reception.shape[-1])
    ):
        return xp.zeros_like(ds.health)
    return xp.clip(ds.signal_reception[..., idx], 0.0, 1.0)


def _drives(ds: Any, cfg: Any) -> Any:
    xp = ds.xp
    alive = ((ds.health > 0.0) & (~ds.obstacle)).astype(ds.health.dtype)
    max_resource = max(float(cfg.resources.max_resource), float(cfg.actions.epsilon))
    hunger = 1.0 - xp.clip(ds.resource / max_resource, 0.0, 1.0)
    pain = 1.0 - xp.clip(ds.health, 0.0, 1.0)
    boundary_stress = 1.0 - xp.clip(ds.boundary, 0.0, 1.0)
    crowding_source = ds.arrays.get("alive_density")
    if crowding_source is None:
        crowding_source = _neighbor_mean(alive, xp)
    crowding = xp.clip(crowding_source, 0.0, 1.0)
    food_neighbor = ds.arrays.get("food_mean")
    if food_neighbor is None:
        food_neighbor = _neighbor_mean(ds.food, xp)
    toxin_neighbor = ds.arrays.get("toxin_mean")
    if toxin_neighbor is None:
        toxin_neighbor = _neighbor_mean(ds.toxin, xp)
    food_pressure = xp.clip(0.5 * ds.food + 0.5 * food_neighbor, 0.0, 1.0)
    toxin_pressure = xp.clip(0.5 * ds.toxin + 0.5 * toxin_neighbor, 0.0, 1.0)
    if bool(getattr(cfg.action_transitions, "enabled", False)) and (
        "active_sense_ttl" in ds.arrays
    ):
        remembered = ds.active_sense_ttl > 0
        food_pressure = xp.where(remembered, ds.active_sense_food_memory, food_pressure)
        toxin_pressure = xp.where(remembered, ds.active_sense_toxin_memory, toxin_pressure)
    signal_delta = xp.mean(xp.abs(ds.signal_reception - ds.signal_memory), axis=-1)
    novelty = xp.clip(
        0.60 * signal_delta
        + 0.25 * xp.abs(ds.food - food_neighbor)
        + 0.15 * xp.abs(ds.toxin - toxin_neighbor),
        0.0,
        1.0,
    )
    local_integration = _neighbor_mean(xp.clip(ds.integration, 0.0, 1.0), xp)
    social_need = (
        xp.clip(ds.cooperation, 0.0, 1.0) * crowding * (1.0 - xp.clip(local_integration, 0.0, 1.0))
    )
    result = {
        "hunger": hunger,
        "pain": pain,
        "boundary_stress": boundary_stress,
        "crowding": crowding,
        "food_pressure": food_pressure,
        "toxin_pressure": toxin_pressure,
        "novelty": novelty,
        "social_need": social_need,
    }
    for key, value in result.items():
        result[key] = xp.where(alive > 0.0, xp.clip(value, 0.0, 1.0), 0.0).astype(ds.health.dtype)
    return result, alive


def _mvp_utilities(ds: Any, cfg: Any) -> Any:
    xp = ds.xp
    h, w = ds.health.shape
    actions = int(ds.possibility.shape[-1])
    out = xp.zeros((h, w, actions), dtype=ds.health.dtype)
    drives, alive = _drives(ds, cfg)
    hunger, pain = drives["hunger"], drives["pain"]
    boundary_stress, crowding = drives["boundary_stress"], drives["crowding"]
    food_pressure, toxin_pressure = drives["food_pressure"], drives["toxin_pressure"]
    novelty, social_need = drives["novelty"], drives["social_need"]

    food_signal = _channel(ds, cfg, SignalChannel.FOOD)
    danger_signal = _channel(ds, cfg, SignalChannel.DANGER)
    threat_signal = _channel(ds, cfg, SignalChannel.THREAT)
    coord_signal = _channel(ds, cfg, SignalChannel.COORDINATION)
    distress_signal = _channel(ds, cfg, SignalChannel.DISTRESS)
    repro_signal = _channel(ds, cfg, SignalChannel.REPRODUCTION)
    territory_signal = _channel(ds, cfg, SignalChannel.TERRITORY)
    integration_signal = _channel(ds, cfg, SignalChannel.INTEGRATION)

    max_resource = max(float(cfg.resources.max_resource), float(cfg.actions.epsilon))
    resource = xp.clip(ds.resource, 0.0, float(cfg.resources.max_resource))
    resource_norm = xp.clip(resource / max_resource, 0.0, 1.0)
    health = xp.clip(ds.health, 0.0, 1.0)
    boundary = xp.clip(ds.boundary, 0.0, 1.0)
    integration = xp.clip(ds.integration, 0.0, 1.0)
    memory = xp.clip(ds.memory, 0.0, 1.0)

    out[..., int(Action.REST)] = (
        0.15 * pain + 0.10 * boundary_stress + 0.05 * toxin_pressure - 0.02 * hunger
    )
    signal_spread = xp.mean(xp.clip(ds.signal_reception, 0.0, 1.0), axis=-1)
    out[..., int(Action.SENSE)] = 0.45 * novelty + 0.20 * signal_spread + 0.10 * memory - 0.02
    out[..., int(Action.FEED)] = (
        1.50 * hunger * food_pressure * xp.clip(ds.grazing, 0.0, 1.0)
        + 0.35 * food_signal
        - 0.10 * toxin_pressure
        - 0.05
    )
    emergency = resource_norm <= float(cfg.resources.emergency_feed_threshold)
    out[..., int(Action.FEED)] += float(
        cfg.resources.emergency_feed_boost
    ) * hunger * food_pressure * emergency.astype(ds.health.dtype) + 0.50 * hunger * xp.clip(
        ds.food, 0.0, 1.0
    )
    out[..., int(Action.INHIBIT)] = (
        0.75 * threat_signal * xp.clip(ds.aggression, 0.0, 1.0)
        + 0.35 * danger_signal
        + 0.25 * territory_signal
        - 0.08
    )
    out[..., int(Action.INTEGRATE)] = (
        0.90 * coord_signal
        + 0.45 * integration_signal
        + 0.35 * social_need
        + 0.30 * memory
        + 0.20 * integration
        - 0.10
    )
    out[..., int(Action.REPAIR)] = (
        1.10 * boundary_stress + 0.80 * pain + 0.25 * distress_signal - 0.15 * hunger - 0.05
    )
    out[..., int(Action.REPRODUCE)] = (
        xp.clip(ds.reproduction_rate, 0.0, 1.0)
        * resource_norm
        * health
        * boundary
        * xp.maximum(integration, 0.05)
        + 0.20 * repro_signal
        - 0.40
    )
    prey_pressure = ds.arrays.get("alive_density")
    if prey_pressure is None:
        prey_pressure = _neighbor_mean(
            ((ds.health > 0.0) & (~ds.obstacle)).astype(ds.health.dtype), xp
        )
    out[..., int(Action.INGEST)] = (
        1.20 * xp.clip(ds.predation, 0.0, 1.0) * prey_pressure * hunger
        + 0.25 * threat_signal * xp.clip(ds.aggression, 0.0, 1.0)
        + 0.15 * distress_signal * xp.clip(ds.predation, 0.0, 1.0)
        - 0.35
    )
    out[..., int(Action.FLEE)] = (
        0.95 * (danger_signal + toxin_pressure + threat_signal) * xp.clip(ds.mobility, 0.0, 1.0)
        + 0.25 * pain
        - 0.05
    )
    out[..., int(Action.PURSUE)] = (
        0.70 * xp.clip(ds.predation, 0.0, 1.0) * prey_pressure
        + 0.30 * threat_signal * xp.clip(ds.aggression, 0.0, 1.0)
        - 0.10
    )
    out[..., int(Action.EXPEL)] = -0.35 + 0.15 * boundary_stress
    out[..., int(Action.SPLIT)] = -0.45 + 0.20 * boundary_stress + 0.10 * (1.0 - integration)
    out[..., int(Action.MERGE)] = -0.40 + 0.20 * coord_signal + 0.10 * crowding

    food_drive = xp.clip(food_pressure + 0.75 * food_signal, 0.0, 1.0)
    danger_drive = xp.clip(toxin_pressure + danger_signal + threat_signal, 0.0, 1.0)
    food_gy, food_gx = _gradient(food_drive, xp)
    danger_gy, danger_gx = _gradient(danger_drive, xp)
    social_gy, social_gx = _gradient(coord_signal, xp)
    mobility, curiosity = xp.clip(ds.mobility, 0.0, 1.0), xp.clip(ds.curiosity, 0.0, 1.0)
    occupied = ((ds.occupancy >= 0) | ds.obstacle | (ds.health > 0.0)).astype(ds.health.dtype)
    last = ds.arrays.get("last_movement_action")
    for action, (dy, dx) in MOVE_DELTAS.items():
        if action in DIAGONAL_MOVES and not bool(cfg.actions.diagonal_movement_enabled):
            out[..., int(action)] = -1.0
            continue
        toward_food = dy * food_gy + dx * food_gx
        away_danger = -(dy * danger_gy + dx * danger_gx)
        toward_social = dy * social_gy + dx * social_gx
        neighbor_food = shift_2d(xp.clip(ds.food, 0.0, 1.0), xp, -dy, -dx)
        neighbor_toxin = shift_2d(xp.clip(ds.toxin, 0.0, 1.0), xp, -dy, -dx)
        neighbor_blocked = shift_2d(occupied, xp, -dy, -dx)
        directional = (
            float(cfg.actions.movement_food_weight) * hunger * neighbor_food
            - float(cfg.actions.movement_toxin_weight) * neighbor_toxin
            - float(cfg.actions.movement_crowding_weight) * neighbor_blocked
        )
        inertia = 0.0
        if last is not None:
            inertia = float(cfg.actions.movement_persistence_bonus) * (last == int(action)).astype(
                ds.health.dtype
            ) - float(cfg.actions.movement_reverse_penalty) * (
                last == int(REVERSE_MOVE_ACTION[action])
            ).astype(ds.health.dtype)
        out[..., int(action)] = (
            0.95 * mobility * hunger * toward_food
            + 1.10 * mobility * away_danger
            + 0.25 * mobility * curiosity * novelty
            + 0.20 * mobility * xp.clip(ds.cooperation, 0.0, 1.0) * toward_social
            + mobility * directional
            + inertia
            - 0.05
        )

    if not bool(cfg.communication.enabled):
        out[..., int(Action.COMMUNICATE)] = -1.0
    else:
        emit_capacity = (
            xp.clip(ds.emit_strength, 0.0, 1.0) * xp.clip(ds.emit_efficiency, 0.0, 1.0) * resource
        )
        channel_interest = xp.mean(xp.clip(ds.channel_emission_bias, 0.0, 1.0), axis=-1)
        out[..., int(Action.COMMUNICATE)] = (
            0.35 * food_signal * hunger
            + 0.70 * danger_signal
            + 0.35 * threat_signal
            + 0.55 * coord_signal
            + 0.45 * distress_signal
            + 0.20 * repro_signal
            + 0.45 * social_need
            + 0.25 * pain
            + 0.15 * boundary_stress
            + 0.25 * xp.clip(ds.cooperation, 0.0, 1.0)
            + 0.30 * emit_capacity
            + 0.15 * channel_interest
            - float(cfg.communication.base_emit_cost)
        )

    out = xp.where(alive[..., None] > 0.0, out, 0.0)
    out = xp.where(ds.obstacle[..., None], 0.0, out)
    return out.astype(ds.health.dtype), drives


def compute_utilities_gpu(ds: Any, cfg: Any) -> None:
    """Compute the same utility law as ``owl.engine.utility.compute_utilities``.

    The baseline path is a direct array-namespace translation. Advanced policies add
    their declared epistemic/risk/effort terms without changing the base law.
    """
    xp = ds.xp
    util, drives = _mvp_utilities(ds, cfg)
    if bool(getattr(cfg.possibility, "advanced_enabled", False)):
        eps = float(cfg.actions.epsilon)
        probs = xp.clip(ds.possibility, eps, 1.0)
        entropy = -xp.sum(probs * xp.log(probs), axis=-1) / max(
            math.log(float(probs.shape[-1])), eps
        )
        epistemic = xp.zeros_like(util)
        e = xp.clip(
            0.45 * drives["novelty"] + 0.35 * entropy + 0.20 * (1.0 - xp.clip(ds.memory, 0.0, 1.0)),
            0.0,
            1.0,
        )
        epistemic[..., int(Action.SENSE)] = e
        for action in MOVE_DELTAS:
            epistemic[..., int(action)] = 0.25 * e * xp.clip(ds.curiosity, 0.0, 1.0)
        epistemic[..., int(Action.COMMUNICATE)] = 0.25 * e * xp.clip(ds.cooperation, 0.0, 1.0)
        epistemic[..., int(Action.INTEGRATE)] = 0.20 * e
        base_risk = xp.clip(
            0.45 * drives["toxin_pressure"]
            + 0.30 * drives["pain"]
            + 0.25 * drives["boundary_stress"],
            0.0,
            1.0,
        )
        risk = xp.zeros_like(util)
        for action in MOVE_DELTAS:
            risk[..., int(action)] = 0.4 * base_risk
        risk[..., int(Action.INGEST)] = 0.6 * base_risk + 0.2 * (
            1.0 - xp.clip(ds.predation, 0.0, 1.0)
        )
        risk[..., int(Action.REPRODUCE)] = 0.5 * (
            1.0 - xp.clip(ds.resource / max(float(cfg.resources.max_resource), eps), 0.0, 1.0)
        )
        risk[..., int(Action.FEED)] = 0.25 * drives["toxin_pressure"]
        low = 1.0 - xp.clip(ds.resource / max(float(cfg.resources.max_resource), eps), 0.0, 1.0)
        effort = xp.zeros_like(util)
        for action in MOVE_DELTAS:
            effort[..., int(action)] = (
                low
                * float(cfg.resources.movement_cost)
                / max(float(cfg.resources.max_resource), eps)
            )
        effort[..., int(Action.COMMUNICATE)] = (
            low
            * float(cfg.communication.base_emit_cost)
            / max(float(cfg.resources.max_resource), eps)
        )
        effort[..., int(Action.REPAIR)] = 0.15 * low
        effort[..., int(Action.REPRODUCE)] = 0.35 * low
        effort[..., int(Action.INGEST)] = 0.20 * low
        cooldown = ds.arrays.get("action_cooldown", 0.0)
        util = (
            util
            + float(cfg.possibility.epistemic_weight) * epistemic
            - float(cfg.possibility.risk_weight) * risk
            - float(cfg.possibility.effort_weight) * effort
            - float(cfg.possibility.cooldown_weight) * cooldown
        )
        # Decision-homeostasis fields are deliberately calculated on device; the
        # base utility remains the CPU pragmatic value.
        if bool(getattr(cfg.decision_homeostasis, "enabled", False)):
            res = xp.clip(ds.resource / max(float(cfg.resources.max_resource), eps), 0.0, 1.0)
            starv = xp.clip(ds.arrays.get("starvation_debt", 1.0 - res), 0.0, 1.0)
            patch_crisis = ds.arrays.get("patch_crisis", xp.zeros_like(res))
            home = xp.clip(
                0.35 * (1.0 - res)
                + 0.30 * starv
                + 0.20 * (1.0 - xp.clip(ds.health, 0.0, 1.0))
                + 0.15 * (1.0 - xp.clip(ds.boundary, 0.0, 1.0)),
                0.0,
                1.0,
            )
            urgency = xp.clip(
                0.55 * home + 0.25 * xp.clip(ds.toxin, 0.0, 1.0) + 0.20 * patch_crisis, 0.0, 1.0
            )
            write_array(ds, "last_decision_urgency", urgency.astype(ds.health.dtype))
            write_array(ds, "last_homeostatic_error", home.astype(ds.health.dtype))
    live = (ds.health > 0.0) & (~ds.obstacle)
    util = xp.where(live[..., None], util, 0.0).astype(ds.health.dtype)
    write_array(ds, "last_utilities", util)
    write_array(ds, "pre_utilities", util.copy())
