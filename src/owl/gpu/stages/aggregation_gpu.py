"""Patch/global aggregation that recovers the CPU scientific contract.

The routines remain backend neutral (NumPy/CuPy) and use fixed-shape array
operations.  They intentionally mirror ``owl.engine.aggregation`` rather than
using a visually plausible but scientifically different summary law.
"""

from __future__ import annotations

from typing import Any

from owl.core.actions import Action
from owl.gpu.array_write import write_array, write_global_array, write_patch_array
from owl.gpu.stage_metrics import metric_float, metric_int
from owl.kernels.circular import weighted_patch_circular_statistics


def _block_view(field: Any, patch_size: int) -> Any:
    h, w = field.shape[:2]
    if h % patch_size or w % patch_size:
        raise ValueError(f"world shape {(h, w)} must be divisible by patch size {patch_size}")
    ph, pw = h // patch_size, w // patch_size
    if field.ndim == 2:
        return field.reshape(ph, patch_size, pw, patch_size).swapaxes(1, 2)
    return field.reshape(ph, patch_size, pw, patch_size, *field.shape[2:]).swapaxes(1, 2)


def _float32(value: Any, xp: Any) -> Any:
    """Return a backend array in the authoritative OWL physical dtype."""
    return value.astype(xp.float32, copy=False)


def _block_sum_2d(field: Any, patch_size: int, xp: Any) -> Any:
    """Mirror CPU float32 inputs, float64 reduction, float32 output."""
    blocks = _block_view(_float32(field, xp), patch_size)
    return xp.sum(blocks, axis=(2, 3), dtype=xp.float64).astype(xp.float32)


def _block_mean(field: Any, patch_size: int, xp: Any) -> Any:
    """Return a CPU-contract patch mean for a 2D field."""
    blocks = _block_view(_float32(field, xp), patch_size)
    count = float(patch_size * patch_size)
    return (xp.sum(blocks, axis=(2, 3), dtype=xp.float64) / count).astype(xp.float32)


def _block_weighted_mean(
    field: Any, weights: Any, patch_size: int, xp: Any, eps: float = 1e-8
) -> Any:
    """Mirror CPU weighted block means exactly across NumPy and CuPy.

    Cell values and weights are float32. Reductions accumulate in float64,
    while the observable patch result is float32. This avoids CuPy's default
    float32 reduction drift without changing the executable CPU contract.
    """
    values = _float32(field, xp)
    weight_values = _float32(weights, xp)
    block = _block_view(values, patch_size)
    wt = _block_view(weight_values, patch_size)

    denominator64 = xp.sum(wt, axis=(2, 3), dtype=xp.float64)
    if values.ndim == 2:
        numerator64 = xp.sum(block * wt, axis=(2, 3), dtype=xp.float64)
        quotient = numerator64 / xp.maximum(denominator64, xp.float64(eps))
        return xp.where(denominator64 > 0.0, quotient, 0.0).astype(xp.float32)

    expand = wt[(...,) + (None,) * (values.ndim - 2)]
    numerator64 = xp.sum(block * expand, axis=(2, 3), dtype=xp.float64)
    out_denominator = denominator64[(...,) + (None,) * (values.ndim - 2)]
    quotient = numerator64 / xp.maximum(out_denominator, xp.float64(eps))
    return xp.where(out_denominator > 0.0, quotient, 0.0).astype(xp.float32)


def _normalize_last_axis(values: Any, xp: Any, eps: float) -> Any:
    clipped = xp.clip(values, 0.0, None)
    total = xp.sum(clipped, axis=-1, keepdims=True)
    rest = xp.zeros_like(clipped)
    rest[..., int(Action.REST)] = 1.0
    return xp.where(total > eps, clipped / xp.maximum(total, eps), rest)


def _normalized_entropy(probability: Any, xp: Any, eps: float) -> Any:
    p = _normalize_last_axis(probability, xp, eps)
    actions = max(int(p.shape[-1]), 2)
    entropy = -xp.sum(xp.where(p > 0.0, p * xp.log(xp.maximum(p, eps)), 0.0), axis=-1) / xp.log(
        float(actions)
    )
    return xp.clip(entropy, 0.0, 1.0)


def _upsample(field: Any, patch_size: int, h: int, w: int, xp: Any) -> Any:
    out = xp.repeat(xp.repeat(field, patch_size, axis=0), patch_size, axis=1)
    return out[:h, :w, ...]


def _noetic_components(
    ds: Any,
    cfg: Any,
    *,
    parent_patch_integration: Any | None = None,
    parent_patch_pressure: Any | None = None,
    parent_patch_crisis: Any | None = None,
) -> Any:
    xp = ds.xp
    h, w = ds.health.shape
    eps = float(cfg.actions.epsilon)
    alive = (ds.health > 0.0) & (~ds.obstacle)
    resource = xp.clip(ds.resource / max(float(cfg.resources.max_resource), eps), 0.0, 1.0)
    B = xp.clip(0.35 * ds.health + 0.35 * ds.boundary + 0.30 * resource, 0.0, 1.0)
    M = xp.clip(ds.memory, 0.0, 1.0)
    P = _normalized_entropy(ds.possibility, xp, eps)
    coupling = ds.arrays.get("coupling_strength", xp.zeros_like(ds.health))
    C = xp.clip(0.5 * ds.integration + 0.5 * xp.clip(coupling, 0.0, 1.0), 0.0, 1.0)
    patch_integration = (
        parent_patch_integration
        if parent_patch_integration is not None
        else ds.patch_arrays.get("integration")
    )
    if patch_integration is None:
        parent = xp.zeros_like(ds.health)
    else:
        parent = _upsample(
            xp.clip(patch_integration, 0.0, 1.0),
            int(cfg.world.patch_size),
            h,
            w,
            xp,
        )
    parent_weight = ds.arrays.get("parent_weight", xp.zeros_like(ds.health))
    K = xp.clip(0.5 * parent + 0.5 * xp.clip(parent_weight, 0.0, 1.0), 0.0, 1.0)
    patch_pressure = (
        parent_patch_pressure
        if parent_patch_pressure is not None
        else ds.patch_arrays.get("patch_carrying_pressure")
    )
    if patch_pressure is None:
        pressure = xp.zeros_like(ds.health)
    else:
        pressure = _upsample(xp.clip(patch_pressure, 0.0, 1.0), int(cfg.world.patch_size), h, w, xp)
    Theta = xp.clip(0.5 * ds.threshold + 0.5 * pressure, 0.0, 1.0)
    patch_crisis = (
        parent_patch_crisis
        if parent_patch_crisis is not None
        else ds.patch_arrays.get("patch_crisis")
    )
    if patch_crisis is None:
        crisis = xp.zeros_like(ds.health)
    else:
        crisis = _upsample(xp.clip(patch_crisis, 0.0, 1.0), int(cfg.world.patch_size), h, w, xp)
    prediction_error = ds.arrays.get("prediction_error", xp.zeros_like(ds.health))
    E = xp.clip(0.5 * crisis + 0.5 * xp.clip(prediction_error, 0.0, 1.0), 0.0, 1.0)
    N = xp.clip(
        0.22 * B + 0.16 * M + 0.15 * P + 0.20 * C + 0.20 * K - 0.17 * Theta - 0.20 * E,
        0.0,
        1.0,
    )
    for name, arr in (
        ("noetic_B", B),
        ("noetic_M", M),
        ("noetic_P", P),
        ("noetic_C", C),
        ("noetic_K", K),
        ("noetic_Theta", Theta),
        ("noetic_N", N),
    ):
        arr = xp.where(alive, arr, 0.0)
        write_array(ds, name, arr.astype(ds.health.dtype, copy=False))
    return tuple(
        ds.arrays[name]
        for name in (
            "noetic_B",
            "noetic_M",
            "noetic_P",
            "noetic_C",
            "noetic_K",
            "noetic_Theta",
            "noetic_N",
        )
    )


def aggregate_patches_gpu(ds: Any, cfg: Any) -> dict[str, Any]:
    xp = ds.xp
    # CPU scientific contract computes cell noetic K/Theta/E against the prior
    # parent summary because the new PatchState is assigned only on return.
    old_parent_integration = (
        ds.patch_arrays.get("integration").copy() if "integration" in ds.patch_arrays else None
    )
    old_parent_pressure = (
        ds.patch_arrays.get("patch_carrying_pressure").copy()
        if "patch_carrying_pressure" in ds.patch_arrays
        else None
    )
    old_parent_crisis = (
        ds.patch_arrays.get("patch_crisis").copy() if "patch_crisis" in ds.patch_arrays else None
    )
    s = int(cfg.world.patch_size)
    eps = float(cfg.actions.epsilon)
    alive_bool = (ds.health > 0.0) & (ds.boundary > 0.0) & (~ds.obstacle)
    alive = alive_bool.astype(xp.float32)
    integration32 = xp.clip(ds.integration, 0.0, 1.0).astype(xp.float32, copy=False)
    weights = (alive * (xp.float32(0.10) + xp.float32(0.90) * integration32)).astype(xp.float32)

    write_patch_array(
        ds,
        "activation",
        _block_weighted_mean(xp.clip(ds.activation, 0.0, 1.0), weights, s, xp, eps),
    )
    write_patch_array(
        ds, "memory", _block_weighted_mean(xp.clip(ds.memory, 0.0, 1.0), weights, s, xp, eps)
    )
    write_patch_array(
        ds,
        "integration",
        _block_weighted_mean(xp.clip(ds.integration, 0.0, 1.0), weights, s, xp, eps),
    )
    resource = _block_weighted_mean(
        xp.clip(ds.resource, 0.0, float(cfg.resources.max_resource)), weights, s, xp, eps
    )
    write_patch_array(
        ds, "resource", xp.clip(resource / max(float(cfg.resources.max_resource), eps), 0.0, 1.0)
    )
    write_patch_array(
        ds, "health", _block_weighted_mean(xp.clip(ds.health, 0.0, 1.0), weights, s, xp, eps)
    )
    write_patch_array(
        ds, "boundary", _block_weighted_mean(xp.clip(ds.boundary, 0.0, 1.0), weights, s, xp, eps)
    )

    phase32 = ds.phase.astype(xp.float32, copy=False)
    phase, sync, _, _ = weighted_patch_circular_statistics(
        phase32,
        weights,
        s,
        xp,
        resultant_support_epsilon=float(cfg.phase.patch_resultant_support_epsilon),
    )
    write_patch_array(ds, "phase", phase)
    write_patch_array(ds, "synchrony", sync)

    possibility = _block_weighted_mean(ds.possibility, weights, s, xp, eps)
    empty = _block_sum_2d(weights, s, xp) <= 0.0
    normalized = _normalize_last_axis(possibility, xp, eps)
    rest = xp.zeros_like(normalized)
    rest[..., int(Action.REST)] = 1.0
    write_patch_array(ds, "possibility", xp.where(empty[..., None], rest, normalized))

    reception = ds.arrays.get("signal_reception", ds.signal)
    signal_pressure = _block_weighted_mean(reception, xp.maximum(alive, 0.0), s, xp, eps)
    write_patch_array(ds, "signal_pressure", xp.clip(signal_pressure, 0.0, 1.0))

    phase_blocks = _block_view(phase32, s)
    patch_phase = phase[:, :, None, None]
    align = (
        xp.float32(0.5) + xp.float32(0.5) * xp.cos(phase_blocks - patch_phase).astype(xp.float32)
    ).astype(xp.float32)
    alive_blocks = _block_view(alive, s)
    align_num = xp.sum(align * alive_blocks, axis=(2, 3), dtype=xp.float64)
    align_den = xp.sum(alive_blocks, axis=(2, 3), dtype=xp.float64)
    coherence = xp.where(
        align_den > 0.0,
        align_num / xp.maximum(align_den, xp.float64(eps)),
        0.0,
    ).astype(xp.float32)
    coherence = xp.clip(coherence, 0.0, 1.0).astype(xp.float32)
    write_patch_array(ds, "coherence", coherence)
    write_patch_array(
        ds,
        "cross_scale",
        xp.clip(xp.float32(0.5) * sync + xp.float32(0.5) * coherence, 0.0, 1.0).astype(xp.float32),
    )

    # The CPU aggregate creates zero policy/intention fields. The top-down stage
    # computes the actual parent policy immediately afterward.
    write_patch_array(
        ds, "intention", xp.zeros_like(ds.patch_arrays["integration"], dtype=xp.int32)
    )
    write_patch_array(ds, "policy_bias", xp.zeros_like(ds.patch_arrays["possibility"]))

    density = _block_mean(alive, s, xp)
    food_mean = _block_mean(xp.clip(ds.food, 0.0, 1.0), s, xp)
    starvation = ds.arrays.get(
        "starvation_debt",
        1.0 - xp.clip(ds.resource / max(float(cfg.resources.max_resource), eps), 0.0, 1.0),
    )
    starv_mean = _block_weighted_mean(xp.clip(starvation, 0.0, 1.0), alive, s, xp, eps)
    readout = ds.readout
    repro_frac = _block_weighted_mean(
        (readout == int(Action.REPRODUCE)).astype(ds.health.dtype), alive, s, xp, eps
    )
    move_mask = xp.zeros_like(ds.health)
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
        move_mask = move_mask + (readout == int(action)).astype(ds.health.dtype)
    move_frac = _block_weighted_mean(xp.clip(move_mask, 0.0, 1.0), alive, s, xp, eps)
    feed_frac = _block_weighted_mean(
        (readout == int(Action.FEED)).astype(ds.health.dtype), alive, s, xp, eps
    )
    death_mask = ds.arrays.get("last_death_mask", xp.zeros_like(ds.health, dtype=bool))
    death_pressure = _block_mean(death_mask.astype(ds.health.dtype), s, xp)

    if bool(getattr(cfg.cross_scale_homeostasis, "enabled", False)):
        carrying = xp.clip(
            float(cfg.cross_scale_homeostasis.crowding_pressure_weight) * density
            + float(cfg.cross_scale_homeostasis.food_deficit_weight) * (1.0 - food_mean)
            + float(cfg.cross_scale_homeostasis.starvation_pressure_weight) * starv_mean
            + float(cfg.cross_scale_homeostasis.reproduction_pressure_weight) * repro_frac,
            0.0,
            1.0,
        )
        resource_norm = xp.clip(ds.resource / max(float(cfg.resources.max_resource), eps), 0.0, 1.0)
        res_mean = _block_weighted_mean(resource_norm, alive, s, xp, eps)
        crisis = xp.clip(
            0.45 * starv_mean + 0.25 * (1.0 - res_mean) + 0.20 * carrying + 0.10 * death_pressure,
            0.0,
            1.0,
        )
    else:
        carrying = xp.zeros_like(density)
        crisis = xp.zeros_like(density)

    for name, value in (
        ("alive_density", density),
        ("food_mean", food_mean),
        ("starvation_debt_mean", starv_mean),
        ("reproduction_fraction", repro_frac),
        ("movement_fraction", move_frac),
        ("feed_fraction", feed_frac),
        ("death_pressure", death_pressure),
        ("patch_crisis", crisis),
        ("patch_carrying_pressure", carrying),
    ):
        write_patch_array(ds, name, value)

    noetic = _noetic_components(
        ds,
        cfg,
        parent_patch_integration=old_parent_integration,
        parent_patch_pressure=old_parent_pressure,
        parent_patch_crisis=old_parent_crisis,
    )
    for name, value in zip(
        ("noetic_B", "noetic_M", "noetic_P", "noetic_C", "noetic_K", "noetic_Theta", "noetic_N"),
        noetic,
        strict=True,
    ):
        write_patch_array(ds, name, _block_weighted_mean(value, xp.maximum(alive, 0.0), s, xp, eps))

    return {"patches_device": xp.asarray(ds.patch_arrays["integration"].size, dtype=xp.int64)}


def aggregate_global_gpu(ds: Any, cfg: Any, *, force: bool = True) -> dict[str, Any]:
    xp = ds.xp
    eps = float(cfg.actions.epsilon)
    if not force:
        cadence = int(cfg.topdown.apex_update_every)
        if int(ds.tick) != 0 and int(ds.tick) % cadence != 0:
            return {"skipped": True, "reason": "apex cadence"}
    if "integration" not in ds.patch_arrays:
        aggregate_patches_gpu(ds, cfg)
    integ = xp.clip(ds.patch_arrays["integration"], 0.0, 1.0)
    alive_patch = xp.clip(ds.patch_arrays["health"], 0.0, 1.0) > 0.0
    weights = integ * alive_patch.astype(integ.dtype)
    fallback_weights = alive_patch.astype(integ.dtype)
    weight_sum = xp.sum(weights)
    weights = xp.where(weight_sum > 0.0, weights, fallback_weights)
    weight_sum = xp.sum(weights)
    no_alive = xp.sum(fallback_weights) <= 0.0

    integration = xp.where(
        no_alive,
        0.0,
        xp.sum(integ * weights) / xp.maximum(weight_sum, eps),
    )
    signal = xp.sum(
        ds.patch_arrays["signal_pressure"] * weights[..., None], axis=(0, 1)
    ) / xp.maximum(weight_sum, eps)
    possibility = xp.sum(
        ds.patch_arrays["possibility"] * weights[..., None], axis=(0, 1)
    ) / xp.maximum(weight_sum, eps)
    possibility = _normalize_last_axis(possibility, xp, eps)
    rest = xp.zeros_like(possibility)
    rest[int(Action.REST)] = 1.0
    possibility = xp.where(no_alive, rest, possibility)

    fragmentation = xp.var(integ.astype(xp.float64)) if integ.size else xp.asarray(0.0)
    mean_policy = _normalize_last_axis(
        xp.mean(ds.patch_arrays["possibility"], axis=(0, 1)), xp, eps
    )
    diversity = -xp.sum(
        xp.where(mean_policy > 0.0, mean_policy * xp.log(mean_policy + eps), 0.0)
    ) / xp.log(float(max(int(mean_policy.shape[0]), 2)))
    diversity = xp.clip(diversity, 0.0, 1.0)
    complexity = xp.clip(integration * (1.0 - fragmentation) * (0.5 + 0.5 * diversity), 0.0, 1.0)

    # Final CPU aggregation always records bottom-up viability pressure.
    patch_weights = alive_patch.astype(integ.dtype) * xp.maximum(integ, 0.05)
    patch_denom = xp.sum(patch_weights)
    starv_field = ds.patch_arrays.get("starvation_debt_mean", xp.zeros_like(integ))
    food_field = ds.patch_arrays.get("food_mean", xp.ones_like(integ))
    carrying_field = ds.patch_arrays.get("patch_carrying_pressure", xp.zeros_like(integ))
    starv = xp.where(
        patch_denom > 0.0,
        xp.sum(xp.clip(starv_field, 0.0, 1.0) * patch_weights) / xp.maximum(patch_denom, eps),
        0.0,
    )
    food_mean = xp.where(
        patch_denom > 0.0,
        xp.sum(xp.clip(food_field, 0.0, 1.0) * patch_weights) / xp.maximum(patch_denom, eps),
        1.0,
    )
    carrying = xp.where(
        patch_denom > 0.0,
        xp.sum(xp.clip(carrying_field, 0.0, 1.0) * patch_weights) / xp.maximum(patch_denom, eps),
        0.0,
    )
    food_deficit = 1.0 - food_mean
    crisis = xp.clip(0.45 * starv + 0.35 * food_deficit + 0.20 * carrying, 0.0, 1.0)
    integration = xp.clip(integration * (1.0 - 0.55 * crisis), 0.0, 1.0)
    complexity = xp.clip(complexity * (1.0 - 0.35 * crisis), 0.0, 1.0)
    fragmentation = xp.clip(xp.maximum(fragmentation, crisis), 0.0, 1.0)

    write_global_array(ds, "signal_pressure", xp.clip(signal, 0.0, 1.0))
    write_global_array(ds, "possibility", possibility)
    # ``aggregate_global`` constructs a fresh CPU GlobalState. Its policy and
    # score buffers therefore begin at zero before top-down intention is
    # recomputed. Reset the device mirrors at the same scientific epoch rather
    # than carrying stale apex scores across aggregate objects.
    write_global_array(ds, "policy_bias", xp.zeros_like(possibility))
    write_global_array(ds, "intention_scores", xp.zeros((10,), dtype=possibility.dtype))
    ds.scalars["global_integration"] = metric_float(ds, xp.clip(integration, 0.0, 1.0))
    ds.scalars["global_readout"] = metric_int(ds, xp.argmax(possibility))
    ds.scalars["global_intention"] = metric_int(ds, xp.asarray(0, dtype=xp.int32))
    ds.scalars["global_fragmentation"] = metric_float(ds, xp.clip(fragmentation, 0.0, 1.0))
    ds.scalars["global_diversity"] = metric_float(ds, diversity)
    ds.scalars["global_complexity"] = metric_float(ds, complexity)
    ds.scalars["global_crisis"] = metric_float(ds, crisis)
    ds.scalars["global_carrying_pressure"] = metric_float(ds, carrying)
    ds.scalars["global_starvation_pressure"] = metric_float(ds, starv)
    ds.scalars["global_food_deficit"] = metric_float(ds, food_deficit)
    return {"global_integration_device": integration}
