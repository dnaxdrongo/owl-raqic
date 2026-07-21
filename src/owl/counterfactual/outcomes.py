"""Backend-native raw multi-horizon outcome capture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from owl.counterfactual.schema import DeathEvidence, HorizonStatus


@dataclass(frozen=True)
class OutcomeDevicePacket:
    values: dict[str, Any]

    @property
    def nbytes(self) -> int:
        return sum(int(getattr(value, "nbytes", 0)) for value in self.values.values())


def _locate(ds: Any, ow_id: Any) -> tuple[Any, Any, Any]:
    xp = ds.xp
    matches = ds.occupancy == ow_id
    present = xp.any(matches)
    flat = xp.argmax(matches.reshape(-1)).astype(xp.int64)
    width = int(ds.health.shape[1])
    y = (flat // width).astype(xp.int32)
    x = (flat % width).astype(xp.int32)
    return present, y, x


def _scalar(xp: Any, value: Any, dtype: Any) -> Any:
    return xp.asarray(value, dtype=dtype).reshape(1)


def capture_outcome_device(
    ds: Any,
    forced: Any,
    *,
    horizon: int,
    source_tick: int,
    first_death_tick: int = -1,
    source_baseline: dict[str, Any] | None = None,
) -> OutcomeDevicePacket:
    """Reduce one branch to a compact device packet without a full state copy."""
    xp = ds.xp
    ow_id = forced.ow_id[0]
    present, y, x = _locate(ds, ow_id)
    alive = present & (ds.health[y, x] > 0.0) & (~ds.obstacle[y, x])

    def safe_float(value: Any) -> Any:
        return xp.where(present, value, xp.asarray(0, dtype=ds.health.dtype))

    def safe_int(value: Any, dtype: Any) -> Any:
        return xp.where(present, value, xp.asarray(-1, dtype=dtype))

    target_id = forced.target_ow_id[0]
    target_present, target_y, target_x = _locate(ds, target_id)
    semantic_target = target_id >= 0
    resolved_target_y = xp.where(semantic_target & target_present, target_y, forced.target_y[0])
    resolved_target_x = xp.where(semantic_target & target_present, target_x, forced.target_x[0])
    dy = xp.abs(y - resolved_target_y)
    dx = xp.abs(x - resolved_target_x)
    cfg = ds.metadata["cfg"]
    if str(cfg.world.boundary_mode) == "toroidal":
        dy = xp.minimum(dy, int(ds.health.shape[0]) - dy)
        dx = xp.minimum(dx, int(ds.health.shape[1]) - dx)
    distance = xp.maximum(dy, dx).astype(ds.health.dtype)
    population = xp.sum((ds.health > 0.0) & (~ds.obstacle), dtype=xp.int64)
    baseline = source_baseline or {}
    source_health = baseline.get("health", ds.health[y, x])
    source_resource = baseline.get("resource", ds.resource[y, x])
    source_boundary = baseline.get("boundary", ds.boundary[y, x])
    source_integration = baseline.get("integration", ds.integration[y, x])
    source_memory = baseline.get("memory", ds.memory[y, x])
    source_y = baseline.get("coordinate_y", forced.source_y[0])
    source_x = baseline.get("coordinate_x", forced.source_x[0])
    end_health = safe_float(ds.health[y, x])
    end_resource = safe_float(ds.resource[y, x])
    end_boundary = safe_float(ds.boundary[y, x])
    end_integration = safe_float(ds.integration[y, x])
    end_memory = safe_float(ds.memory[y, x])
    values = {
        "horizon": _scalar(xp, horizon, xp.int32),
        "source_tick": _scalar(xp, source_tick, xp.int64),
        "end_tick": _scalar(xp, int(ds.tick), xp.int64),
        "present": _scalar(xp, present, bool),
        "alive": _scalar(xp, alive, bool),
        "coordinate_y": _scalar(xp, safe_int(y, xp.int32), xp.int32),
        "coordinate_x": _scalar(xp, safe_int(x, xp.int32), xp.int32),
        "source_coordinate_y": _scalar(xp, source_y, xp.int32),
        "source_coordinate_x": _scalar(xp, source_x, xp.int32),
        "displacement_y": _scalar(xp, xp.where(present, y - source_y, 0), xp.int32),
        "displacement_x": _scalar(xp, xp.where(present, x - source_x, 0), xp.int32),
        "source_health": _scalar(xp, source_health, ds.health.dtype),
        "source_resource": _scalar(xp, source_resource, ds.health.dtype),
        "source_boundary": _scalar(xp, source_boundary, ds.health.dtype),
        "source_integration": _scalar(xp, source_integration, ds.health.dtype),
        "source_memory": _scalar(xp, source_memory, ds.health.dtype),
        "health": _scalar(xp, end_health, ds.health.dtype),
        "resource": _scalar(xp, end_resource, ds.health.dtype),
        "boundary": _scalar(xp, end_boundary, ds.health.dtype),
        "integration": _scalar(xp, end_integration, ds.health.dtype),
        "memory": _scalar(xp, end_memory, ds.health.dtype),
        "health_delta": _scalar(xp, end_health - source_health, ds.health.dtype),
        "resource_delta": _scalar(xp, end_resource - source_resource, ds.health.dtype),
        "boundary_delta": _scalar(xp, end_boundary - source_boundary, ds.health.dtype),
        "integration_delta": _scalar(xp, end_integration - source_integration, ds.health.dtype),
        "memory_delta": _scalar(xp, end_memory - source_memory, ds.health.dtype),
        "active_sense_food_memory": _scalar(
            xp, safe_float(ds.active_sense_food_memory[y, x]), ds.health.dtype
        ),
        "active_sense_toxin_memory": _scalar(
            xp, safe_float(ds.active_sense_toxin_memory[y, x]), ds.health.dtype
        ),
        "active_sense_alive_memory": _scalar(
            xp, safe_float(ds.active_sense_alive_memory[y, x]), ds.health.dtype
        ),
        "active_sense_ttl": _scalar(xp, safe_int(ds.active_sense_ttl[y, x], xp.int32), xp.int32),
        "active_sense_new_cell_count": _scalar(
            xp, safe_int(ds.active_sense_new_cell_count[y, x], xp.int32), xp.int32
        ),
        "active_sense_new_target_count": _scalar(
            xp, safe_int(ds.active_sense_new_target_count[y, x], xp.int32), xp.int32
        ),
        "selected_action": _scalar(xp, safe_int(ds.readout[y, x], xp.int16), xp.int16),
        "compiled_action": _scalar(
            xp, safe_int(ds.compiled_execution_action[y, x], xp.int16), xp.int16
        ),
        "semantic_target_y": _scalar(xp, forced.target_y[0], xp.int32),
        "semantic_target_x": _scalar(xp, forced.target_x[0], xp.int32),
        "semantic_target_ow_id": _scalar(xp, target_id, xp.int64),
        "target_distance": _scalar(xp, xp.where(present, distance, 0), ds.health.dtype),
        "source_target_distance": _scalar(xp, forced.target_distance[0], ds.health.dtype),
        "target_distance_delta": _scalar(
            xp, xp.where(present, distance - forced.target_distance[0], 0), ds.health.dtype
        ),
        "known_hazard": _scalar(xp, safe_float(ds.toxin[y, x]), ds.health.dtype),
        "contact_opportunity": _scalar(xp, present & semantic_target & (distance <= 1), bool),
        "parent_id": _scalar(xp, safe_int(ds.parent_id[y, x], xp.int64), xp.int64),
        "lineage_id": _scalar(xp, safe_int(ds.lineage_id[y, x], xp.int64), xp.int64),
        "age": _scalar(xp, safe_int(ds.age[y, x], xp.int32), xp.int32),
        "first_death_tick": _scalar(xp, first_death_tick, xp.int64),
        "death_evidence": _scalar(
            xp,
            xp.where(
                alive,
                int(DeathEvidence.NONE),
                xp.where(present, int(DeathEvidence.DEAD), int(DeathEvidence.ABSENT_AMBIGUOUS)),
            ),
            xp.int8,
        ),
        "horizon_status": _scalar(
            xp,
            xp.where(
                alive,
                0,
                xp.where(present, 1, 2),
            ),
            xp.int8,
        ),
        "population": _scalar(xp, population, xp.int64),
        "world_food": _scalar(xp, xp.sum(ds.food, dtype=xp.float64), xp.float64),
        "world_toxin": _scalar(xp, xp.sum(ds.toxin, dtype=xp.float64), xp.float64),
        "world_waste": _scalar(
            xp,
            xp.sum(ds.arrays.get("waste", xp.zeros_like(ds.food)), dtype=xp.float64),
            xp.float64,
        ),
    }
    return OutcomeDevicePacket(values)


def transfer_outcome(backend: Any, packet: OutcomeDevicePacket) -> OutcomeDevicePacket:
    """Transfer a compact outcome vector after all device derivation is complete."""
    return OutcomeDevicePacket(
        {name: backend.asnumpy(value) for name, value in packet.values.items()}
    )


def outcome_status_name(code: int) -> str:
    return {
        0: HorizonStatus.COMPLETED.value,
        1: HorizonStatus.FOCAL_DEAD.value,
        2: HorizonStatus.FOCAL_ABSENT.value,
    }.get(int(code), HorizonStatus.BRANCH_FAILED.value)
