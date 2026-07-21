from __future__ import annotations

from typing import Any

from owl.core.actions import Action, SignalChannel
from owl.gpu.array_write import write_array


def encode_experience_gpu(ds: Any, cfg: Any) -> Any:
    xp = ds.xp
    alive = ((ds.health > 0) & (~ds.obstacle)).astype(xp.float32)
    rn = xp.clip(
        ds.resource / max(float(cfg.resources.max_resource), float(cfg.actions.epsilon)), 0, 1
    )
    action = xp.zeros_like(ds.health, dtype=xp.float32)
    action += 0.40 * (ds.readout == int(Action.FEED))
    action += 0.30 * (ds.readout == int(Action.REPAIR))
    action += 0.35 * (ds.readout == int(Action.INTEGRATE))
    action += 0.20 * (ds.readout == int(Action.COMMUNICATE))
    action = xp.clip(action, 0, 1)
    traces = []
    for channel in (
        SignalChannel.FOOD,
        SignalChannel.DANGER,
        SignalChannel.COORDINATION,
        SignalChannel.INTEGRATION,
    ):
        traces.append(
            xp.clip(ds.signal_reception[..., int(channel)], 0, 1)
            if int(channel) < ds.signal_reception.shape[-1]
            else xp.zeros_like(ds.health)
        )
    signal = xp.maximum(xp.maximum(traces[0], traces[1]), xp.maximum(traces[2], traces[3]))
    exp = (
        (
            0.22 * rn
            + 0.22 * xp.clip(ds.health, 0, 1)
            + 0.18 * xp.clip(ds.boundary, 0, 1)
            + 0.20 * xp.clip(ds.integration, 0, 1)
            + 0.10 * action
            + 0.08 * signal
        )
        * xp.clip(ds.memory_capacity, 0, 1)
        * alive
    )
    return xp.clip(exp, 0, 1).astype(xp.float32)


def update_memory_gpu(ds: Any, cfg: Any) -> None:
    xp = ds.xp
    ret = float(getattr(getattr(cfg, "memory", object()), "decay", 0.95))
    ret = max(0, min(1, ret))
    alive = ((ds.health > 0) & (~ds.obstacle)).astype(xp.float32)
    write_array(
        ds,
        "memory",
        xp.clip((ret * ds.memory + (1 - ret) * encode_experience_gpu(ds, cfg)) * alive, 0, 1),
    )


def compute_identity_persistence_gpu(ds: Any, cfg: Any) -> Any:
    xp = ds.xp
    alive = ((ds.health > 0) & (~ds.obstacle)).astype(xp.float32)
    return xp.clip(
        (
            0.35 * xp.clip(ds.memory, 0, 1)
            + 0.30 * xp.clip(ds.boundary, 0, 1)
            + 0.20 * xp.clip(ds.health, 0, 1)
            + 0.15 * xp.clip(ds.integration, 0, 1)
        )
        * alive,
        0,
        1,
    ).astype(xp.float32)
