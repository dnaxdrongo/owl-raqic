from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from owl.core.actions import Action
from owl.gpu.array_write import write_array
from owl.gpu.stage_metrics import metric_float, metric_int


@dataclass
class BiologyDiagnostics:
    metabolism_cost: Any
    toxin_damage: Any
    starvation_damage: Any
    repair_gain: Any
    death_mask: Any


def fused_biology_update(ds: Any, cfg: Any, *, write_diagnostics: bool = False) -> dict[str, Any]:
    """Fused elementwise biology update.

    This is a vectorized, backend-neutral implementation. On CuPy it executes as
    a small number of elementwise kernels; on NumPy it is the dense reference.
    """
    xp = ds.xp
    live = (ds.health > 0) & ~ds.obstacle
    readout = ds.readout.astype(xp.int32)
    resource = ds.resource
    health = ds.health
    starvation = ds.arrays.get("starvation_debt", xp.zeros_like(health))
    toxin = ds.toxin
    base = float(cfg.resources.metabolism_base)
    move_cost = float(cfg.resources.movement_cost)
    repair_cost = float(getattr(cfg.ecology, "repair_resource_cost", 0.02))
    repair_gain = float(getattr(cfg.ecology, "repair_health_gain", 0.02))
    toxin_damage = toxin * float(getattr(cfg.resources, "toxin_health_damage", 0.02))
    movement_actions = (
        int(Action.MOVE_N),
        int(Action.MOVE_S),
        int(Action.MOVE_E),
        int(Action.MOVE_W),
        int(Action.MOVE_NE),
        int(Action.MOVE_NW),
        int(Action.MOVE_SE),
        int(Action.MOVE_SW),
        int(Action.FLEE),
        int(Action.PURSUE),
    )
    moving = xp.zeros_like(live)
    for a in movement_actions:
        moving |= readout == a
    repairing = readout == int(Action.REPAIR)
    cost = xp.where(
        live, base + xp.where(moving, move_cost, 0.0) + xp.where(repairing, repair_cost, 0.0), 0.0
    )
    resource2 = xp.maximum(resource - cost, 0.0)
    starvation2 = xp.where(
        live & (resource2 <= 0),
        xp.minimum(1.0, starvation + float(cfg.resources.starvation_debt_gain)),
        xp.maximum(0.0, starvation - float(cfg.resources.starvation_debt_recovery)),
    )
    starvation_damage = starvation2 * float(cfg.resources.starvation_health_damage)
    gain = xp.where(repairing & (resource > repair_cost), repair_gain, 0.0)
    health2 = xp.clip(health + gain - toxin_damage - starvation_damage, 0.0, 1.0)
    write_array(ds, "resource", xp.where(live, resource2, 0.0))
    write_array(ds, "starvation_debt", xp.where(live, starvation2, 0.0))
    write_array(ds, "health", xp.where(live, health2, 0.0))
    if write_diagnostics:
        ds.metadata["biology_diagnostics"] = BiologyDiagnostics(
            cost, toxin_damage, starvation_damage, gain, health2 <= 0
        )
    return {
        "live": metric_int(ds, xp.sum(live)),
        "mean_health": metric_float(ds, xp.mean(ds.health)) if ds.health.size else 0.0,
    }
