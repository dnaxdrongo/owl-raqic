"""Backend-neutral top-down policy matching ``owl.engine.topdown``."""

from __future__ import annotations

from typing import Any

from owl.core.actions import Action, GlobalIntention, PatchIntention, SignalChannel
from owl.gpu.array_write import write_array, write_global_array, write_patch_array
from owl.gpu.stage_metrics import metric_int


def _channel(field: Any, channel: SignalChannel, cfg: Any) -> Any:
    idx = int(channel)
    if idx < min(int(cfg.communication.num_channels), int(field.shape[-1])):
        return field[..., idx]
    return 0.0


def _clip_backend_scalar(xp: Any, value: Any) -> Any:
    """Clip a host or device scalar without relying on NumPy-only coercion.

    ``numpy.clip`` accepts a Python float directly, while ``cupy.clip`` routes
    through the input object's ``clip`` method.  Normalizing with the active
    array namespace makes the operation backend invariant and does not copy an
    existing device scalar to the host.
    """
    return xp.clip(xp.asarray(value), 0.0, 1.0)


def _upsample(field: Any, patch_size: int, h: int, w: int, xp: Any) -> Any:
    return xp.repeat(xp.repeat(field, patch_size, axis=0), patch_size, axis=1)[:h, :w, ...]


def _compute_patch_policy(ds: Any, cfg: Any) -> Any:
    xp = ds.xp
    p = ds.patch_arrays
    integration = xp.clip(p["integration"], 0.0, 1.0)
    resource = xp.clip(p["resource"], 0.0, 1.0)
    health = xp.clip(p["health"], 0.0, 1.0)
    boundary = xp.clip(p["boundary"], 0.0, 1.0)
    synchrony = xp.clip(p["synchrony"], 0.0, 1.0)
    coherence = xp.clip(p["coherence"], 0.0, 1.0)
    signal = p["signal_pressure"]
    food = _channel(signal, SignalChannel.FOOD, cfg)
    danger = _channel(signal, SignalChannel.DANGER, cfg)
    threat = _channel(signal, SignalChannel.THREAT, cfg)
    coord = _channel(signal, SignalChannel.COORDINATION, cfg)
    distress = _channel(signal, SignalChannel.DISTRESS, cfg)
    repro = _channel(signal, SignalChannel.REPRODUCTION, cfg)
    territory = _channel(signal, SignalChannel.TERRITORY, cfg)
    integration_signal = _channel(signal, SignalChannel.INTEGRATION, cfg)

    shape = (*integration.shape, len(PatchIntention))
    scores = xp.zeros(shape, dtype=integration.dtype)
    if bool(getattr(cfg.cross_scale_homeostasis, "enabled", False)):
        crisis = xp.clip(p.get("patch_crisis", xp.zeros_like(integration)), 0.0, 1.0)
        carrying = xp.clip(p.get("patch_carrying_pressure", xp.zeros_like(integration)), 0.0, 1.0)
        food_mean = xp.clip(p.get("food_mean", xp.ones_like(integration)), 0.0, 1.0)
        starvation = xp.clip(p.get("starvation_debt_mean", xp.zeros_like(integration)), 0.0, 1.0)
        density = xp.clip(p.get("alive_density", xp.zeros_like(integration)), 0.0, 1.0)
        food_deficit = 1.0 - food_mean
        scores[..., int(PatchIntention.REST)] = (
            0.15 * resource + 0.10 * health + 0.10 * (1.0 - crisis)
        )
        scores[..., int(PatchIntention.SEEK_FOOD)] = (
            0.75 * food_deficit + 0.70 * starvation + 0.30 * food + 0.10 * crisis
        )
        scores[..., int(PatchIntention.AVOID_DANGER)] = (
            0.75 * danger + 0.50 * threat + 0.25 * distress
        )
        scores[..., int(PatchIntention.COORDINATE)] = (
            0.50 * coord + 0.35 * integration_signal + 0.20 * crisis
        )
        scores[..., int(PatchIntention.DEFEND)] = (
            0.50 * threat + 0.25 * (1.0 - boundary) + 0.20 * density
        )
        scores[..., int(PatchIntention.REPRODUCE)] = (
            0.40 * repro + 0.55 * resource * health * boundary * xp.maximum(integration, 0.05)
        ) * (1.0 - xp.clip(crisis + carrying, 0.0, 1.0))
        scores[..., int(PatchIntention.REPAIR)] = (
            0.60 * crisis + 0.45 * (1.0 - health) + 0.35 * (1.0 - boundary)
        )
        scores[..., int(PatchIntention.EXPLORE)] = (
            0.20 * (1.0 - density) + 0.10 * resource + 0.10 * (1.0 - crisis)
        )
    else:
        scores[..., int(PatchIntention.REST)] = 0.20 * resource + 0.20 * health + 0.10 * integration
        scores[..., int(PatchIntention.SEEK_FOOD)] = 0.80 * food + 0.35 * (1.0 - resource)
        scores[..., int(PatchIntention.AVOID_DANGER)] = (
            0.75 * danger + 0.50 * threat + 0.20 * distress
        )
        scores[..., int(PatchIntention.COORDINATE)] = (
            0.55 * coord
            + 0.40 * integration_signal
            + 0.35 * (1.0 - integration)
            + 0.15 * (1.0 - synchrony)
            + 0.15 * (1.0 - coherence)
        )
        scores[..., int(PatchIntention.DEFEND)] = (
            0.55 * threat + 0.35 * territory + 0.20 * (1.0 - boundary)
        )
        scores[..., int(PatchIntention.REPRODUCE)] = (
            0.50 * repro + 0.60 * resource * health * boundary * xp.maximum(integration, 0.05)
        )
        scores[..., int(PatchIntention.REPAIR)] = (
            0.65 * (1.0 - boundary) + 0.35 * (1.0 - health) + 0.25 * distress
        )
        scores[..., int(PatchIntention.EXPLORE)] = (
            0.20 * (1.0 - food) + 0.15 * resource + 0.10 * (1.0 - danger)
        )

    intention = xp.argmax(scores, axis=-1).astype(xp.int32)
    intention = xp.where(health <= 0.0, int(PatchIntention.REST), intention)
    write_patch_array(ds, "intention", intention)
    if "intention_scores" in p:
        write_patch_array(ds, "intention_scores", scores)

    bias = xp.zeros((*integration.shape, len(Action)), dtype=integration.dtype)
    intent = intention
    seek = intent == int(PatchIntention.SEEK_FOOD)
    for action in (Action.SENSE, Action.MOVE_N, Action.MOVE_S, Action.MOVE_E, Action.MOVE_W):
        bias[..., int(action)] += 0.08 * seek
    bias[..., int(Action.FEED)] += 0.35 * seek
    bias[..., int(Action.COMMUNICATE)] += 0.10 * seek
    avoid = intent == int(PatchIntention.AVOID_DANGER)
    bias[..., int(Action.FLEE)] += 0.35 * avoid
    bias[..., int(Action.REPAIR)] += 0.25 * avoid
    bias[..., int(Action.COMMUNICATE)] += 0.20 * avoid
    bias[..., int(Action.INHIBIT)] += 0.12 * avoid
    coordinate = intent == int(PatchIntention.COORDINATE)
    bias[..., int(Action.COMMUNICATE)] += 0.30 * coordinate
    bias[..., int(Action.INTEGRATE)] += 0.40 * coordinate
    bias[..., int(Action.SENSE)] += 0.10 * coordinate
    defend = intent == int(PatchIntention.DEFEND)
    bias[..., int(Action.INHIBIT)] += 0.30 * defend
    bias[..., int(Action.REPAIR)] += 0.20 * defend
    bias[..., int(Action.COMMUNICATE)] += 0.15 * defend
    bias[..., int(Action.INGEST)] += 0.08 * defend
    reproduce = intent == int(PatchIntention.REPRODUCE)
    bias[..., int(Action.REPRODUCE)] += 0.35 * reproduce
    bias[..., int(Action.COMMUNICATE)] += 0.12 * reproduce
    bias[..., int(Action.REST)] += 0.05 * reproduce
    repair = intent == int(PatchIntention.REPAIR)
    bias[..., int(Action.REPAIR)] += 0.40 * repair
    bias[..., int(Action.REST)] += 0.12 * repair
    bias[..., int(Action.INTEGRATE)] += 0.10 * repair
    bias[..., int(Action.INGEST)] -= 0.10 * repair
    explore = intent == int(PatchIntention.EXPLORE)
    for action in (
        Action.SENSE,
        Action.MOVE_N,
        Action.MOVE_S,
        Action.MOVE_E,
        Action.MOVE_W,
        Action.MOVE_NE,
        Action.MOVE_NW,
        Action.MOVE_SE,
        Action.MOVE_SW,
    ):
        bias[..., int(action)] += 0.08 * explore
    rest = intent == int(PatchIntention.REST)
    bias[..., int(Action.REST)] += 0.10 * rest
    bias[..., int(Action.INTEGRATE)] += 0.05 * rest

    scale = float(cfg.topdown.lambda_action_bias) * integration * health
    limit = float(cfg.topdown.max_parent_control)
    bias = xp.clip(bias * scale[..., None], -limit, limit)
    # Final CPU policy always adds lower-level survival pressure, even when
    # cross-scale homeostasis is otherwise disabled.
    crisis = xp.clip(p.get("patch_crisis", xp.zeros_like(integration)), 0.0, 1.0)
    carrying = xp.clip(p.get("patch_carrying_pressure", xp.zeros_like(integration)), 0.0, 1.0)
    starvation = xp.clip(p.get("starvation_debt_mean", xp.zeros_like(integration)), 0.0, 1.0)
    food_mean = xp.clip(p.get("food_mean", xp.ones_like(integration)), 0.0, 1.0)
    urgent = xp.clip(0.50 * crisis + 0.35 * starvation + 0.15 * (1.0 - food_mean), 0.0, 1.0)
    bias[..., int(Action.FEED)] += 0.35 * urgent
    bias[..., int(Action.SENSE)] += 0.16 * urgent
    bias[..., int(Action.REPAIR)] += 0.22 * crisis
    for action in (
        Action.MOVE_N,
        Action.MOVE_S,
        Action.MOVE_E,
        Action.MOVE_W,
        Action.MOVE_NE,
        Action.MOVE_NW,
        Action.MOVE_SE,
        Action.MOVE_SW,
    ):
        bias[..., int(action)] += 0.08 * urgent
    suppression = float(cfg.cross_scale_homeostasis.max_reproduction_suppression)
    bias[..., int(Action.REPRODUCE)] -= suppression * xp.clip(
        0.70 * crisis + 0.50 * carrying, 0.0, 1.0
    )

    bias = xp.clip(bias, -limit, limit)
    write_patch_array(ds, "policy_bias", bias)
    return bias


def _compute_global_policy(ds: Any, cfg: Any) -> Any:
    xp = ds.xp
    signal = ds.global_arrays["signal_pressure"]
    integration = ds.scalars["global_integration"]
    fragmentation = ds.scalars["global_fragmentation"]
    diversity = ds.scalars["global_diversity"]
    food = _channel(signal, SignalChannel.FOOD, cfg)
    danger = _channel(signal, SignalChannel.DANGER, cfg)
    threat = _channel(signal, SignalChannel.THREAT, cfg)
    coord = _channel(signal, SignalChannel.COORDINATION, cfg)
    distress = _channel(signal, SignalChannel.DISTRESS, cfg)
    repro = _channel(signal, SignalChannel.REPRODUCTION, cfg)
    integration_signal = _channel(signal, SignalChannel.INTEGRATION, cfg)
    crisis = _clip_backend_scalar(xp, ds.scalars.get("global_crisis", 0.0))
    carrying = _clip_backend_scalar(xp, ds.scalars.get("global_carrying_pressure", 0.0))
    starv = _clip_backend_scalar(xp, ds.scalars.get("global_starvation_pressure", 0.0))
    food_deficit = _clip_backend_scalar(xp, ds.scalars.get("global_food_deficit", 0.0))
    scores = xp.zeros((len(GlobalIntention),), dtype=signal.dtype)
    scores[int(GlobalIntention.REST)] = 0.20 * integration + 0.20 * (1.0 - fragmentation)
    scores[int(GlobalIntention.EXPAND)] = 0.20 * diversity + 0.20 * food + 0.10 * (1.0 - crisis)
    scores[int(GlobalIntention.CONSERVE)] = 0.80 * crisis + 0.40 * starv + 0.20 * fragmentation
    scores[int(GlobalIntention.SEEK_FOOD)] = 0.70 * food_deficit + 0.60 * starv + 0.20 * food
    scores[int(GlobalIntention.AVOID_THREAT)] = 0.75 * danger + 0.50 * threat
    scores[int(GlobalIntention.COORDINATE)] = (
        0.45 * coord + 0.35 * integration_signal + 0.30 * fragmentation + 0.20 * crisis
    )
    scores[int(GlobalIntention.REPRODUCE)] = (
        0.60 * repro + 0.35 * integration + 0.25 * diversity
    ) * (
        1.0
        - float(cfg.cross_scale_homeostasis.max_reproduction_suppression)
        * xp.maximum(crisis, carrying)
    )
    scores[int(GlobalIntention.DEFEND)] = 0.55 * threat + 0.30 * danger
    scores[int(GlobalIntention.EXPLORE)] = 0.25 * diversity + 0.20 * (1.0 - crisis)
    scores[int(GlobalIntention.REPAIR)] = (
        0.45 * fragmentation + 0.40 * distress + 0.30 * (1.0 - integration) + 0.30 * crisis
    )
    intention = xp.argmax(scores).astype(xp.int32)
    ds.scalars["global_intention"] = metric_int(ds, intention)
    write_global_array(ds, "intention_scores", scores)

    # Vectorized equivalent of global_policy_to_bias; no host branch is needed.
    bias_by_intent = xp.zeros((len(GlobalIntention), len(Action)), dtype=signal.dtype)
    bias_by_intent[int(GlobalIntention.SEEK_FOOD), int(Action.SENSE)] = 0.08
    bias_by_intent[int(GlobalIntention.SEEK_FOOD), int(Action.FEED)] = 0.18
    bias_by_intent[int(GlobalIntention.SEEK_FOOD), int(Action.COMMUNICATE)] = 0.05
    bias_by_intent[int(GlobalIntention.CONSERVE), int(Action.REST)] = 0.10
    bias_by_intent[int(GlobalIntention.CONSERVE), int(Action.REPAIR)] = 0.10
    # The remaining CONSERVE terms depend on current starvation and are
    # applied below after selecting the active intention.
    bias_by_intent[int(GlobalIntention.REPAIR), int(Action.REPAIR)] = 0.16
    bias_by_intent[int(GlobalIntention.REPAIR), int(Action.REST)] = 0.06
    bias_by_intent[int(GlobalIntention.REPAIR), int(Action.SENSE)] = 0.04
    bias_by_intent[int(GlobalIntention.COORDINATE), int(Action.INTEGRATE)] = 0.14
    bias_by_intent[int(GlobalIntention.COORDINATE), int(Action.COMMUNICATE)] = 0.12
    bias_by_intent[int(GlobalIntention.COORDINATE), int(Action.SENSE)] = 0.05
    bias_by_intent[int(GlobalIntention.REPRODUCE), int(Action.REPRODUCE)] = 0.12
    bias_by_intent[int(GlobalIntention.REPRODUCE), int(Action.COMMUNICATE)] = 0.05
    bias_by_intent[int(GlobalIntention.AVOID_THREAT), int(Action.FLEE)] = 0.14
    bias_by_intent[int(GlobalIntention.AVOID_THREAT), int(Action.REPAIR)] = 0.10
    bias_by_intent[int(GlobalIntention.AVOID_THREAT), int(Action.COMMUNICATE)] = 0.06
    for action in (
        Action.SENSE,
        Action.MOVE_N,
        Action.MOVE_S,
        Action.MOVE_E,
        Action.MOVE_W,
        Action.MOVE_NE,
        Action.MOVE_NW,
        Action.MOVE_SE,
        Action.MOVE_SW,
    ):
        bias_by_intent[int(GlobalIntention.EXPLORE), int(action)] = 0.05
    bias_by_intent[int(GlobalIntention.REST), int(Action.REST)] = 0.05
    bias_by_intent[int(GlobalIntention.REST), int(Action.INTEGRATE)] = 0.03
    chosen = bias_by_intent[intention].copy()
    # Crisis-aware modifiers from the final CPU policy.
    chosen[int(Action.FEED)] += 0.12 * starv + 0.08 * crisis
    chosen[int(Action.REPAIR)] += 0.08 * crisis
    chosen[int(Action.SENSE)] += 0.04 * crisis
    # Intention-specific crisis scaling that cannot be encoded in a constant table.
    conserve_mask = intention == int(GlobalIntention.CONSERVE)
    coordinate_mask = intention == int(GlobalIntention.COORDINATE)
    reproduce_mask = intention == int(GlobalIntention.REPRODUCE)
    explore_mask = intention == int(GlobalIntention.EXPLORE)
    rest_mask = intention == int(GlobalIntention.REST)
    chosen[int(Action.FEED)] = xp.where(
        conserve_mask, chosen[int(Action.FEED)] + 0.08 * starv, chosen[int(Action.FEED)]
    )
    chosen[int(Action.REPRODUCE)] = xp.where(
        conserve_mask, chosen[int(Action.REPRODUCE)] - 0.18, chosen[int(Action.REPRODUCE)]
    )
    chosen[int(Action.INGEST)] = xp.where(
        conserve_mask, chosen[int(Action.INGEST)] - 0.04, chosen[int(Action.INGEST)]
    )
    chosen[int(Action.INTEGRATE)] = xp.where(
        coordinate_mask, 0.14 * (1.0 - crisis), chosen[int(Action.INTEGRATE)]
    )
    chosen[int(Action.REPRODUCE)] = xp.where(
        reproduce_mask, 0.12 * (1.0 - xp.maximum(crisis, carrying)), chosen[int(Action.REPRODUCE)]
    )
    for action in (
        Action.SENSE,
        Action.MOVE_N,
        Action.MOVE_S,
        Action.MOVE_E,
        Action.MOVE_W,
        Action.MOVE_NE,
        Action.MOVE_NW,
        Action.MOVE_SE,
        Action.MOVE_SW,
    ):
        chosen[int(action)] = xp.where(explore_mask, 0.05 * (1.0 - crisis), chosen[int(action)])
    chosen[int(Action.INTEGRATE)] = xp.where(
        rest_mask, 0.03 * (1.0 - crisis), chosen[int(Action.INTEGRATE)]
    )
    # Universal crisis suppression is applied after intention-specific base
    # terms, matching the CPU order (including the REPRODUCE branch).
    chosen[int(Action.REPRODUCE)] -= (
        float(cfg.cross_scale_homeostasis.max_reproduction_suppression)
        * xp.maximum(crisis, carrying)
        * 0.18
    )
    scale = 0.50 * float(cfg.topdown.lambda_action_bias) * xp.maximum(integration, 0.1)
    chosen = xp.clip(
        chosen * scale,
        -float(cfg.topdown.max_parent_control),
        float(cfg.topdown.max_parent_control),
    )
    write_global_array(ds, "policy_bias", chosen)
    return chosen


def dispatch_parent_context_gpu(ds: Any, cfg: Any, *, force_global: bool = False) -> None:
    xp = ds.xp
    scientific_stage_parity = bool(ds.metadata.get("scientific_stage_parity", False))
    parent_before = (
        ds.raqic_parent_intention.copy()
        if scientific_stage_parity and "raqic_parent_intention" in ds.arrays
        else None
    )
    h, w = ds.health.shape
    s = int(cfg.world.patch_size)
    patch_bias = _compute_patch_policy(ds, cfg)
    cadence = int(cfg.topdown.apex_update_every)
    if force_global or int(ds.tick) == 0 or int(ds.tick) % cadence == 0:
        global_bias = _compute_global_policy(ds, cfg)
    else:
        global_bias = ds.global_arrays.get(
            "policy_bias", xp.zeros((len(Action),), dtype=ds.health.dtype)
        )
    cell_bias = _upsample(patch_bias, s, h, w, xp) + global_bias[None, None, :]
    limit = float(cfg.topdown.max_parent_control)
    cell_bias = xp.clip(cell_bias, -limit, limit)
    write_array(ds, "pre_parent_bias", cell_bias)
    write_array(ds, "_parent_bias_for_conflict", cell_bias.copy())

    # ``raqic_parent_intention`` is not a normalized view of OWL's signed
    # homeostatic policy bias. It is the bounded top-down probability field
    # produced by the RAQIC bottom-up aggregation / intention recursion in
    # ``prepare_cross_scale_context_gpu``. Overwriting it here changed the
    # quantum-instrument input on GPU only.

    phase = _upsample(ds.patch_arrays["phase"], s, h, w, xp)
    write_array(ds, "_parent_phase", phase)

    if parent_before is not None:
        unchanged = xp.all(ds.raqic_parent_intention == parent_before)
        unchanged_value = unchanged.item() if hasattr(unchanged, "item") else unchanged
        if not bool(unchanged_value):
            raise RuntimeError(
                "dispatch_parent_context_gpu mutated raqic_parent_intention; "
                "RAQIC parent intention is owned by prepare_cross_scale_context_gpu"
            )


def apply_threshold_modulation_gpu(ds: Any, cfg: Any) -> None:
    xp = ds.xp
    s = int(cfg.world.patch_size)
    h, w = ds.health.shape
    intent = _upsample(ds.patch_arrays["intention"], s, h, w, xp)
    integration = _upsample(xp.clip(ds.patch_arrays["integration"], 0.0, 1.0), s, h, w, xp)
    health = _upsample(xp.clip(ds.patch_arrays["health"], 0.0, 1.0), s, h, w, xp)
    target = xp.full((h, w), 0.50, dtype=ds.threshold.dtype)
    active = intent != int(PatchIntention.REST)
    urgent = (
        (intent == int(PatchIntention.AVOID_DANGER))
        | (intent == int(PatchIntention.DEFEND))
        | (intent == int(PatchIntention.REPAIR))
    )
    target = xp.where(active, 0.40, target)
    target = xp.where(urgent, 0.30, target)
    alpha = xp.clip(
        float(cfg.topdown.lambda_threshold) * integration * health,
        0.0,
        float(cfg.topdown.max_parent_control),
    )
    alive = ds.health > 0.0
    updated = xp.where(alive, ds.threshold + alpha * (target - ds.threshold), ds.threshold)
    write_array(ds, "threshold", xp.clip(updated, 0.0, 1.0))
