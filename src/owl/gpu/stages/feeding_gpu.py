from __future__ import annotations

from typing import Any

import numpy as np

from owl.core.actions import Action, EventKind
from owl.core.state import EventRecord
from owl.gpu.array_write import write_array
from owl.gpu.stage_metrics import metric_float, metric_int


def _record_gpu_ingestion_events(
    state: Any, pre_resource: Any, post_resource: Any, probability_field: Any, cfg: Any
) -> None:
    """Record CPU-compatible sparse ingestion events after GPU feeding."""
    if not hasattr(state, "event_queue"):
        return
    max_events = int(getattr(getattr(cfg, "recording", object()), "max_events", 4096))
    if max_events <= 0:
        return
    try:
        import cupy as cp
    except Exception:
        cp = None

    if cp is not None and isinstance(post_resource, cp.ndarray):
        transfer = cp.asnumpy(cp.maximum(post_resource - pre_resource, 0.0))
        probability = cp.asnumpy(probability_field)
    else:
        transfer = np.maximum(np.asarray(post_resource) - np.asarray(pre_resource), 0.0)
        probability = np.asarray(probability_field)

    ys, xs = np.nonzero(transfer > 0.0)
    if ys.size == 0:
        return

    existing = len(state.event_queue)
    remaining = max(0, max_events - existing)
    if remaining <= 0:
        return

    for y, x in zip(ys[:remaining], xs[:remaining], strict=False):
        amount = float(transfer[y, x])
        prob = float(probability[y, x]) if probability.shape == transfer.shape else 1.0
        state.event_queue.append(
            EventRecord(
                tick=int(state.tick),
                kind=EventKind.INGESTION,
                source=(int(y), int(x)),
                target=(int(y), int(x)),
                payload={
                    "success": True,
                    "probability": prob,
                    "resource_transfer": amount,
                },
            )
        )


def compute_intake_gpu(ds: Any, cfg: Any) -> Any:
    xp = ds.xp
    live = (ds.health > 0) & (~ds.obstacle)
    feed = (ds.readout == int(Action.FEED)) & live
    food = xp.clip(ds.food, 0.0, 1.0)
    grazing = xp.clip(ds.grazing, 0.0, 1.0)
    q = xp.clip(ds.resource, 0.0, float(cfg.resources.max_resource))

    if bool(getattr(cfg.ecology, "advanced_enabled", False)):
        capacity = xp.maximum(
            0.0,
            1.0 - q / max(float(cfg.resources.max_resource), float(cfg.actions.epsilon)),
        )
        monod = food / (float(cfg.ecology.monod_half_saturation) + food)
        intake = (
            feed.astype(xp.float32)
            * float(cfg.resources.feed_efficiency)
            * grazing
            * monod
            * capacity
        )
    else:
        remaining = xp.maximum(0.0, float(cfg.resources.max_resource) - q)
        intake = float(cfg.resources.feed_efficiency) * food * grazing * feed.astype(xp.float32)
        intake = xp.minimum(intake, remaining)
        intake = xp.minimum(intake, food)

    intake = xp.where(ds.obstacle, 0.0, xp.clip(intake, 0.0, 1.0))
    return intake


def apply_feeding_gpu(ds: Any, cfg: Any) -> dict[str, Any]:
    # Capture telemetry baseline state before array modifications
    state_obj = ds if hasattr(ds, "event_queue") else getattr(ds, "state", ds)
    pre_resource_for_events = (
        state_obj.resource.copy() if hasattr(state_obj, "resource") else ds.resource.copy()
    )

    xp = ds.xp
    intake = compute_intake_gpu(ds, cfg)
    food = xp.clip(ds.food - intake, 0.0, 1.0)

    if bool(getattr(cfg.ecology, "advanced_enabled", False)):
        immediate = float(cfg.resources.feeding_immediate_fraction) * intake
        buffered = (1.0 - float(cfg.resources.feeding_immediate_fraction)) * intake
        digestion = ds.digestion + buffered
        digested = float(cfg.ecology.digestion_decay) * digestion
        resource = ds.resource + immediate + float(cfg.ecology.digestion_efficiency) * digested
        waste = ds.waste + (1.0 - float(cfg.ecology.digestion_efficiency)) * digested
        digestion = digestion - digested
        write_array(ds, "digestion", xp.clip(digestion, 0.0, 1.0))
        write_array(ds, "waste", xp.clip(waste, 0.0, 1.0))
    else:
        resource = ds.resource + intake

    resource = xp.where(ds.obstacle, 0.0, xp.clip(resource, 0.0, float(cfg.resources.max_resource)))
    food = xp.where(ds.obstacle, 0.0, food)

    write_array(ds, "food", food)
    write_array(ds, "resource", resource)

    if bool(getattr(cfg.ecology, "advanced_enabled", False)) and "last_intake" in ds.arrays:
        write_array(ds, "last_intake", xp.where(ds.obstacle, 0.0, xp.clip(intake, 0.0, 1.0)))

    fed = intake > 0
    metrics = {
        "fed": metric_int(ds, xp.sum(fed)),
        "intake_total": metric_float(ds, xp.sum(intake)),
    }

    # Execute telemetry hook mapping inside active execution pathway
    resource_after_events = state_obj.resource if hasattr(state_obj, "resource") else ds.resource
    probability_for_events = getattr(state_obj, "last_action_probabilities", None)
    if probability_for_events is None:
        probability_for_events = resource_after_events
    elif getattr(probability_for_events, "ndim", 0) == 3:
        probability_for_events = probability_for_events[..., 0]

    _record_gpu_ingestion_events(
        state_obj,
        pre_resource_for_events,
        resource_after_events,
        probability_for_events,
        cfg,
    )

    return metrics
