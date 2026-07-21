from __future__ import annotations

from typing import Any

from owl.core.actions import Action, SignalChannel
from owl.gpu.array_write import write_array
from owl.gpu.stage_metrics import metric_int
from owl.gpu.stencil import neighbor_sum_8


def _neighbor_mean(field: Any, xp: Any) -> Any:
    return neighbor_sum_8(field, xp, "toroidal") / 8.0


def compute_automatic_signal_intents_gpu(ds: Any, cfg: Any) -> Any:
    xp = ds.xp
    channels = int(cfg.communication.num_channels)
    out = xp.zeros((*ds.health.shape, channels), dtype=xp.float32)
    if not bool(cfg.communication.enabled):
        return out
    alive = ((ds.health > 0) & (~ds.obstacle)).astype(xp.float32)
    emit = xp.clip(ds.emit_strength, 0, 1) * alive
    resource = xp.clip(ds.resource, 0, 1)
    health = xp.clip(ds.health, 0, 1)
    boundary = xp.clip(ds.boundary, 0, 1)
    integration = xp.clip(ds.integration, 0, 1)
    crowd = _neighbor_mean(alive, xp)

    def setc(ch: Any, val: Any) -> Any:
        idx = int(ch)
        if idx < channels:
            out[..., idx] = val * emit * xp.clip(ds.channel_emission_bias[..., idx], 0, 1)

    setc(SignalChannel.FOOD, xp.clip(ds.grazing, 0, 1) * xp.clip(ds.food, 0, 1))
    setc(SignalChannel.DANGER, xp.maximum(xp.clip(ds.toxin, 0, 1), 1 - health))
    setc(SignalChannel.THREAT, xp.clip(ds.aggression, 0, 1) * crowd)
    setc(SignalChannel.COORDINATION, integration * xp.clip(ds.cooperation, 0, 1))
    setc(SignalChannel.DISTRESS, xp.maximum(1 - health, 1 - boundary))
    setc(
        SignalChannel.REPRODUCTION,
        xp.clip(ds.reproduction_rate, 0, 1) * resource * health * boundary * integration,
    )
    setc(SignalChannel.TERRITORY, boundary * crowd * (0.5 + 0.5 * xp.clip(ds.aggression, 0, 1)))
    setc(SignalChannel.INTEGRATION, integration * xp.clip(ds.coupling_strength, 0, 1))
    out *= xp.clip(ds.signal_precision, 0, 1)[..., None]
    out = xp.where(ds.obstacle[..., None], 0.0, xp.clip(out, 0, 1))
    return out


def emit_signals_gpu(ds: Any, cfg: Any) -> dict[str, Any]:
    xp = ds.xp
    if not bool(cfg.communication.enabled):
        write_array(ds, "signal_emission", xp.zeros_like(ds.signal_emission))
        return {"emitters": 0}
    intents = compute_automatic_signal_intents_gpu(ds, cfg)
    if bool(getattr(cfg.communication, "source_tracking_enabled", False)):
        # Normalize intentional policy only for explicit communicators.
        denom = xp.sum(intents, axis=-1, keepdims=True)
        intentional = xp.where(
            denom > float(cfg.actions.epsilon),
            intents / xp.maximum(denom, float(cfg.actions.epsilon)),
            0.0,
        )
        speaking = ((ds.readout == int(Action.COMMUNICATE)) & (ds.health > 0) & (~ds.obstacle))[
            ..., None
        ]
        intentional = xp.where(speaking, intentional, 0.0)
        mix = float(cfg.communication.intentional_mix)
        emission = (
            ((1 - mix) * intents + mix * intentional)
            * xp.clip(ds.emit_strength, 0, 1)[..., None]
            * xp.clip(ds.emit_efficiency, 0, 1)[..., None]
            * xp.clip(ds.signal_precision, 0, 1)[..., None]
            * (1 - 0.5 * xp.clip(ds.deception_bias, 0, 1))[..., None]
        )
    else:
        factor = (0.25 + 0.75 * xp.clip(ds.integration, 0, 1))[..., None]
        rf = xp.clip(ds.resource / float(cfg.resources.max_resource), 0, 1)[..., None]
        eff = xp.clip(ds.emit_efficiency, 0, 1)[..., None]
        emission = intents * factor * rf * eff
    emission = xp.clip(emission, 0, 1)
    total = xp.sum(emission, axis=-1)
    cost = (
        float(cfg.communication.base_emit_cost) * total / (0.10 + xp.clip(ds.emit_efficiency, 0, 1))
    )
    write_array(ds, "resource", xp.clip(ds.resource - cost, 0, float(cfg.resources.max_resource)))
    write_array(
        ds,
        "signal_emission",
        xp.where(ds.obstacle[..., None], 0.0, xp.clip(ds.signal_emission + emission, 0, 1)),
    )
    return {"emitters": metric_int(ds, xp.sum(total > 0))}


def update_signal_memory_gpu(ds: Any, cfg: Any) -> None:
    xp = ds.xp
    if not bool(cfg.communication.enabled):
        write_array(ds, "signal_memory", xp.zeros_like(ds.signal_memory))
        return
    alive = ((ds.health > 0) & (~ds.obstacle)).astype(xp.float32)
    write_array(
        ds,
        "signal_memory",
        xp.clip((0.97 * ds.signal_memory + 0.03 * ds.signal_reception) * alive[..., None], 0, 1),
    )


def update_channel_trust_gpu(ds: Any, cfg: Any) -> None:
    xp = ds.xp
    if not bool(cfg.communication.enabled):
        return
    prev_r = ds.arrays.get("_tick_start_resource", ds.resource)
    prev_h = ds.arrays.get("_tick_start_health", ds.health)
    prev_i = ds.arrays.get("_tick_start_integration", ds.integration)
    outcome = xp.clip(
        0.4 * (ds.resource - prev_r) + 0.4 * (ds.health - prev_h) + 0.2 * (ds.integration - prev_i),
        -1,
        1,
    )
    alive = ((ds.health > 0) & (~ds.obstacle)).astype(xp.float32)
    trust = (
        ds.channel_trust_local
        + float(cfg.communication.trust_lr)
        * outcome[..., None]
        * xp.clip(ds.signal_reception, 0, 1)
    ) * alive[..., None]
    write_array(ds, "channel_trust_local", xp.clip(trust, 0, 1))


def compute_signal_conflict_gpu(ds: Any, cfg: Any) -> Any:
    xp = ds.xp
    if not bool(cfg.communication.enabled):
        return xp.zeros_like(ds.health)
    r = xp.clip(ds.signal_reception, 0, 1)
    channels = r.shape[-1]

    def ch(c: Any) -> Any:
        return r[..., int(c)] if int(c) < channels else xp.zeros_like(ds.health)

    incompatible = ch(SignalChannel.FOOD) * ch(SignalChannel.DANGER) + ch(
        SignalChannel.THREAT
    ) * ch(SignalChannel.COORDINATION)
    conflict = 0.65 * incompatible + 0.35 * xp.std(r, axis=-1)
    return xp.where(ds.obstacle, 0.0, xp.clip(conflict, 0, 1)).astype(xp.float32)
