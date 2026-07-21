from __future__ import annotations

from typing import Any

from owl.core.actions import Action
from owl.gpu.array_write import write_array
from owl.gpu.stage_metrics import metric_float

_MOVE_ACTIONS = tuple(range(int(Action.MOVE_N), int(Action.MOVE_SW) + 1))


def apply_repair_and_integrate_gpu(ds: Any, cfg: Any) -> None:
    xp = ds.xp
    alive = (ds.health > 0) & (~ds.obstacle)
    repair = (ds.readout == int(Action.REPAIR)) & alive
    integrate = (ds.readout == int(Action.INTEGRATE)) & alive
    resource = ds.resource.copy()
    health = ds.health.copy()
    boundary = ds.boundary.copy()
    memory = ds.memory.copy()
    integration = ds.integration.copy()
    if bool(getattr(cfg.ecology, "advanced_enabled", False)):
        q = xp.clip(resource, 0.0, float(cfg.resources.max_resource))
        sat = q / (float(cfg.ecology.repair_half_saturation) + q)
        resource -= 0.025 * repair.astype(xp.float32) + 0.010 * integrate.astype(xp.float32)
        health += 0.030 * repair.astype(xp.float32) * sat
        boundary += 0.060 * repair.astype(xp.float32) * sat + 0.020 * integrate.astype(xp.float32)
        memory += 0.030 * integrate.astype(xp.float32) * xp.clip(ds.memory_capacity, 0.0, 1.0)
        integration += 0.015 * integrate.astype(xp.float32)
    else:
        rnorm = xp.clip(
            resource / max(float(cfg.resources.max_resource), float(cfg.actions.epsilon)), 0.0, 1.0
        )
        rcost = 0.020 * repair.astype(xp.float32)
        ramount = 0.060 * repair.astype(xp.float32) * rnorm
        spend = xp.minimum(resource, rcost)
        resource -= spend
        health += 0.50 * ramount
        boundary += ramount
        icost = 0.010 * integrate.astype(xp.float32)
        spend = xp.minimum(resource, icost)
        resource -= spend
        memory += 0.030 * integrate.astype(xp.float32) * xp.clip(ds.memory_capacity, 0.0, 1.0)
        boundary += 0.020 * integrate.astype(xp.float32) * rnorm
        integration += 0.010 * integrate.astype(xp.float32)
    write_array(ds, "resource", xp.clip(resource, 0.0, float(cfg.resources.max_resource)))
    write_array(ds, "health", xp.clip(health, 0.0, 1.0))
    write_array(ds, "boundary", xp.clip(boundary, 0.0, 1.0))
    write_array(ds, "memory", xp.clip(memory, 0.0, 1.0))
    write_array(ds, "integration", xp.clip(integration, 0.0, 1.0))


def apply_metabolism_damage_gpu(ds: Any, cfg: Any) -> dict[str, Any]:
    xp = ds.xp
    alive_bool = (ds.health > 0) & (~ds.obstacle)
    alive = alive_bool.astype(xp.float32)
    resource = ds.resource.copy()
    health = ds.health.copy()
    boundary = ds.boundary.copy()
    if bool(getattr(cfg.ecology, "advanced_enabled", False)):
        age_stress = xp.clip(
            ds.age.astype(xp.float32) / float(cfg.ecology.age_stress_scale), 0.0, 1.0
        )
        maintenance = float(cfg.resources.metabolism_base) * (
            0.5 + xp.clip(ds.metabolism, 0.0, 1.0) + 0.15 * age_stress
        )
        move = xp.zeros_like(alive)
        for action in _MOVE_ACTIONS:
            move = xp.where(ds.readout == action, 1.0, move)
        resource -= (maintenance + float(cfg.resources.movement_cost) * move) * alive
        rnorm = xp.clip(
            resource / max(float(cfg.resources.max_resource), float(cfg.actions.epsilon)), 0.0, 1.0
        )
        starving = alive_bool & (rnorm <= float(cfg.resources.emergency_feed_threshold))
        debt = ds.starvation_debt.copy()
        debt = xp.where(starving, debt + float(cfg.resources.starvation_debt_gain), debt)
        debt = xp.where(
            alive_bool & (~starving),
            debt - float(cfg.resources.starvation_debt_recovery) * rnorm,
            debt,
        )
        debt = xp.where(alive_bool, debt, 0.0)
        debt = xp.clip(debt, 0.0, 1.0)
        tox = xp.power(xp.clip(ds.toxin, 0.0, 1.0), float(cfg.ecology.toxin_damage_exponent)) * (
            1.0 - xp.clip(ds.toxin_resistance, 0.0, 1.0)
        )
        starvation_health_damage = float(cfg.resources.starvation_health_damage) * debt * alive
        toxin_health_damage = 0.035 * tox * alive
        health -= starvation_health_damage + toxin_health_damage
        boundary -= 0.015 * tox * alive
        write_array(ds, "age_stress", age_stress)
        write_array(ds, "starvation_debt", debt)
        starve_total = xp.sum(debt)
    else:
        metabolism = float(cfg.resources.metabolism_base) * xp.clip(ds.metabolism, 0.0, 1.0) * alive
        resource -= metabolism
        starvation = xp.maximum(0.0, 0.05 * float(cfg.resources.max_resource) - resource)
        tox = xp.clip(ds.toxin, 0.0, 1.0) * (1.0 - xp.clip(ds.toxin_resistance, 0.0, 1.0))
        starvation_health_damage = 0.02 * starvation
        toxin_health_damage = 0.03 * tox * alive
        health -= starvation_health_damage + toxin_health_damage
        boundary -= 0.01 * tox * alive
        boundary -= 0.01 * xp.maximum(0.0, 0.20 - health) * alive
        starve_total = xp.sum(starvation)
    write_array(
        ds,
        "resource",
        xp.where(ds.obstacle, 0.0, xp.clip(resource, 0.0, float(cfg.resources.max_resource))),
    )
    write_array(ds, "health", xp.where(ds.obstacle, 0.0, xp.clip(health, 0.0, 1.0)))
    write_array(ds, "boundary", xp.where(ds.obstacle, 0.0, xp.clip(boundary, 0.0, 1.0)))
    cadc_buffer = ds.metadata.get("cadc_device_buffer")
    if cadc_buffer is not None:
        from owl.record.cadc_capture import capture_damage_evidence

        capture_damage_evidence(
            cadc_buffer, ds, starvation_health_damage, toxin_health_damage
        )
    return {"starvation_total": metric_float(ds, starve_total)}


def clip_life_fields_gpu(ds: Any, cfg: Any) -> None:
    xp = ds.xp
    bounded = (
        "activation",
        "memory",
        "phase_coherence",
        "integration",
        "health",
        "boundary",
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
    )
    for name in bounded:
        if name in ds.arrays:
            write_array(ds, name, xp.clip(ds.arrays[name], 0.0, 1.0))
    if "resource" in ds.arrays:
        write_array(
            ds,
            "resource",
            xp.where(
                ds.obstacle, 0.0, xp.clip(ds.resource, 0.0, float(cfg.resources.max_resource))
            ),
        )
    for name in ("health", "boundary"):
        if name in ds.arrays:
            write_array(ds, name, xp.where(ds.obstacle, 0.0, xp.clip(ds.arrays[name], 0.0, 1.0)))
    for name in (
        "signal_reception",
        "signal_emission",
        "signal_memory",
        "channel_receptivity",
        "channel_emission_bias",
        "channel_trust_local",
        "digestion",
        "waste",
        "age_stress",
        "last_intake",
        "prediction_error",
        "starvation_debt",
        "movement_loop_score",
        "development_stage",
        "symbiosis",
        "genome",
        "action_cooldown",
        "deception_memory",
        "neighbor_trust",
    ):
        if name in ds.arrays:
            write_array(ds, name, xp.clip(ds.arrays[name], 0.0, 1.0))
