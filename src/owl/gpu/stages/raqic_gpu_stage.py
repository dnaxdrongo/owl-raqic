from __future__ import annotations

from typing import Any

import numpy as np

from owl.core.actions import Action
from owl.gpu.array_write import write_array
from owl.raqic.feature_extraction import FEATURE_NAMES
from owl.raqic.precision import (
    raqic_backend_complex_dtype,
    raqic_backend_real_dtype,
)
from owl_raqic.gpu.actualization_extensions import (
    ActualizationExtensionConfig,
    aggregate_action_phase_context,
)
from owl_raqic.gpu.decision_engine import RAQICDenseDecisionEngine, RAQICDenseExecutionConfig
from owl_raqic.gpu.dense_types import RAQICDenseBatch


def _entropy_concentration_xp(parent_intention: Any, xp: Any, eps: float = 1e-08) -> Any:
    intention = xp.maximum(parent_intention, 0.0)
    sums = xp.sum(intention, axis=-1, keepdims=True)
    n = intention.shape[-1]
    norm = xp.where(
        sums > eps, intention / xp.maximum(sums, eps), xp.ones_like(intention) / float(n)
    )
    ent = -xp.sum(xp.where(norm > 0, norm * xp.log(norm + eps), 0.0), axis=-1) / xp.log(float(n))
    return xp.clip(1.0 - ent, 0.0, 1.0)


def _ensure_parent_intention(ds: Any, cfg: Any) -> Any:
    xp = ds.xp
    h, w = ds.health.shape
    actions = ds.possibility.shape[-1]
    if "raqic_parent_intention" not in ds.arrays or ds.raqic_parent_intention.shape != (
        h,
        w,
        actions,
    ):
        out = xp.zeros((h, w, actions), dtype=ds.health.dtype)
        out[..., int(Action.REST)] = 1.0
        write_array(ds, "raqic_parent_intention", out)
    sums = xp.sum(ds.raqic_parent_intention, axis=-1, keepdims=True)
    write_array(
        ds,
        "raqic_parent_intention",
        xp.where(
            sums > 0, ds.raqic_parent_intention / xp.maximum(sums, 1e-12), ds.raqic_parent_intention
        ),
    )


def quiesce_dead_raqic_fields_gpu(ds: Any) -> None:
    """Mirror the CPU tick-end RAQIC terminal-state contract on device.

    Movement clears accepted source cells before the death stage. Those cells
    are already empty, so ``apply_death_gpu`` does not rediscover them. The CPU
    reference subsequently calls ``quiesce_dead_raqic_fields`` for *all* dead
    or obstacle cells. This device equivalent must run after physical clipping
    and before post-tick aggregation/top-down snapshots.
    """
    if "raqic_probabilities" not in ds.arrays:
        return

    xp = ds.xp
    dead = (ds.health <= 0.0) | ds.obstacle
    rest_action = int(Action.REST)

    for name in (
        "raqic_probabilities",
        "raqic_parent_intention",
        "raqic_debug_density_diag",
        "raqic_pre_mixer_probabilities",
        "raqic_resonant_parent_intention",
        "raqic_shadow_probabilities",
    ):
        arr = ds.arrays.get(name)
        if arr is None or getattr(arr, "ndim", 0) != 3:
            continue
        quiesced = xp.where(dead[..., None], 0.0, arr)
        quiesced[..., rest_action] = xp.where(
            dead,
            1.0,
            quiesced[..., rest_action],
        )
        write_array(ds, name, quiesced)

    for name in (
        "raqic_readout",
        "raqic_record_action",
        "raqic_legacy_shadow_readout",
        "raqic_shadow_readout",
    ):
        arr = ds.arrays.get(name)
        if arr is not None:
            write_array(ds, name, xp.where(dead, rest_action, arr))

    for name in (
        "raqic_utility_innovation",
        "raqic_phase_alignment",
        "raqic_parent_action_phase",
        "raqic_parent_action_coherence",
    ):
        arr = ds.arrays.get(name)
        if arr is not None and getattr(arr, "ndim", 0) == 3:
            write_array(ds, name, xp.where(dead[..., None], 0.0, arr))
    for name in (
        "raqic_interference_delta_l1",
        "raqic_policy_kl",
        "raqic_utility_projection_fraction",
        "raqic_utility_score_cosine",
        "raqic_utility_orthogonality_residual",
        "raqic_utility_innovation_norm",
        "raqic_interference_norm_error",
        "raqic_interference_illegal_mass",
    ):
        arr = ds.arrays.get(name)
        if arr is not None:
            write_array(ds, name, xp.where(dead, 0.0, arr))


def _actualization_config_from_cfg(cfg: Any) -> ActualizationExtensionConfig:
    rq = cfg.raqic
    return ActualizationExtensionConfig(
        variant=str(getattr(rq, "actualization_variant", "stable_baseline")),
        utility_coupling=float(getattr(rq, "utility_coupling", 0.0)),
        utility_projection_epsilon=float(getattr(rq, "utility_projection_epsilon", 1e-8)),
        utility_bound_floor=float(getattr(rq, "utility_bound_floor", 1.0)),
        phase_resonance_coupling=float(getattr(rq, "phase_resonance_coupling", 0.0)),
        interference_mixer_strength=float(getattr(rq, "interference_mixer_strength", 0.0)),
        interference_trotter_steps=int(getattr(rq, "interference_trotter_steps", 1)),
        shadow_only=bool(getattr(rq, "experimental_shadow_only", False)),
    )


def ensure_actualization_graph_buffers_gpu(ds: Any, cfg: Any) -> None:
    """Preallocate fixed-shape RAQIC buffers before graph capture."""
    if not bool(ds.metadata.get("graph_static", False)):
        return
    xp = ds.xp
    h, w = ds.health.shape
    n = int(h * w)
    actions = int(ds.possibility.shape[-1])
    real_dtype = raqic_backend_real_dtype(cfg, xp)
    complex_dtype = raqic_backend_complex_dtype(cfg, xp)
    specs = (
        ("_graph_raqic_utilities", (n, actions), real_dtype),
        ("_graph_raqic_parent_action_phase", (n, actions), real_dtype),
        ("_graph_raqic_parent_action_coherence", (n, actions), real_dtype),
        ("_graph_raqic_amplitudes", (n, actions), complex_dtype),
        ("_graph_raqic_pair_left_scratch", (n,), complex_dtype),
        ("_graph_raqic_pair_right_scratch", (n,), complex_dtype),
        ("_graph_raqic_pre_mixer_probabilities", (n, actions), real_dtype),
    )
    for name, shape, dtype in specs:
        current = ds.arrays.get(name)
        if current is None or tuple(current.shape) != shape or current.dtype != xp.dtype(dtype):
            write_array(ds, name, xp.zeros(shape, dtype=dtype))


def aggregate_action_phase_context_gpu(
    ds: Any,
    cfg: Any,
    child_weights: Any | None = None,
    patch_confidence: Any | None = None,
) -> tuple[Any, Any, Any, Any]:
    """Aggregate prior-tick action phasors and dispatch patch/global context."""
    xp = ds.xp
    h, w = ds.health.shape
    size = int(cfg.world.patch_size)
    if child_weights is None:
        alive = ((ds.health > 0.0) & (~ds.obstacle)).astype(xp.float64)
        child_weights = alive
        denom = xp.sum(
            alive.reshape(h // size, size, w // size, size),
            axis=(1, 3),
            keepdims=True,
        )
        child_weights = (
            alive.reshape(h // size, size, w // size, size) / xp.maximum(denom, 1.0)
        ).reshape(h, w)
    context_dtype = raqic_backend_real_dtype(cfg, xp)
    arrays = aggregate_action_phase_context(
        ds.arrays.get("raqic_probabilities", ds.possibility),
        ds.arrays.get("raqic_phase", xp.zeros_like(ds.possibility)),
        child_weights,
        patch_confidence=patch_confidence,
        patch_size=size,
        patch_weight=float(cfg.raqic.phase_resonance_patch_weight),
        global_weight=float(cfg.raqic.phase_resonance_global_weight),
        support_epsilon=float(cfg.raqic.phase_resonance_support_epsilon),
        rest_index=int(Action.REST),
        xp=xp,
        dtype=context_dtype,
    )
    names = (
        "raqic_patch_action_phase",
        "raqic_patch_action_coherence",
        "raqic_global_action_phase",
        "raqic_global_action_coherence",
        "raqic_parent_action_phase",
        "raqic_parent_action_coherence",
    )
    for name, value in zip(names, arrays, strict=True):
        write_array(ds, name, value.astype(context_dtype, copy=False))
    return arrays[0], arrays[1], arrays[2], arrays[3]


def _normalize_intention_xp(values: Any, xp: Any, eps: float) -> Any:
    values = xp.maximum(values, 0.0)
    sums = xp.sum(values, axis=-1, keepdims=True)
    out = xp.where(sums > eps, values / xp.maximum(sums, eps), 0.0)
    bad = sums[..., 0] <= eps
    rest = xp.zeros_like(out)
    rest[..., int(Action.REST)] = 1.0
    return xp.where(bad[..., None], rest, out)


def prepare_cross_scale_context_gpu(ds: Any, cfg: Any) -> Any:
    """Device translation of ``OWLRAQICEngine.prepare_cross_scale_context``."""
    xp = ds.xp
    h, w = ds.health.shape
    s = int(cfg.world.patch_size)
    ph, pw = h // s, w // s
    actions = int(ds.possibility.shape[-1])
    eps = float(cfg.actions.epsilon)
    real_dtype = raqic_backend_real_dtype(cfg, xp)
    alive = ((ds.health > 0.0) & (~ds.obstacle)).astype(real_dtype)
    probs = xp.clip(ds.arrays.get("raqic_probabilities", ds.possibility), 0.0, 1.0).astype(
        real_dtype
    )
    B = (
        xp.clip(ds.boundary, 0.0, 1.0)
        .reshape(ph, s, pw, s)
        .transpose(0, 2, 1, 3)
        .reshape(ph, pw, s * s)
    )
    R = (
        xp.clip(ds.resource / max(float(cfg.resources.max_resource), eps), 0.0, 1.0)
        .reshape(ph, s, pw, s)
        .transpose(0, 2, 1, 3)
        .reshape(ph, pw, s * s)
    )
    Csrc = ds.arrays.get("noetic_C", ds.integration)
    C = xp.clip(Csrc, 0.0, 1.0).reshape(ph, s, pw, s).transpose(0, 2, 1, 3).reshape(ph, pw, s * s)
    A = alive.reshape(ph, s, pw, s).transpose(0, 2, 1, 3).reshape(ph, pw, s * s)
    P = (
        probs.reshape(ph, s, pw, s, actions)
        .transpose(0, 2, 1, 3, 4)
        .reshape(ph, pw, s * s, actions)
    )
    yy, xx = xp.indices((s, s), dtype=real_dtype)
    dist = xp.sqrt(
        (yy.reshape(-1) - (s - 1) / 2.0) ** 2 + (xx.reshape(-1) - (s - 1) / 2.0) ** 2
    ) / max(float(s), 1.0)
    raw = (
        xp.maximum(B, 0.0)
        * xp.maximum(C, 0.0)
        * xp.maximum(R, 0.0)
        * xp.exp(-xp.maximum(dist, 0.0))[None, None, :]
        * A
    )
    raw_sum = xp.sum(raw, axis=-1, keepdims=True)
    alive_count = xp.sum(A, axis=-1, keepdims=True)
    uniform_alive = xp.where(alive_count > 0.0, A / xp.maximum(alive_count, 1.0), 0.0)
    weights = xp.where(raw_sum > 1e-12, raw / xp.maximum(raw_sum, 1e-12), uniform_alive)
    needs_phase_context = (
        float(getattr(cfg.raqic, "phase_resonance_coupling", 0.0)) != 0.0
        or float(getattr(cfg.raqic, "interference_mixer_strength", 0.0)) != 0.0
        or bool(getattr(cfg.raqic, "record_actualization_diagnostics", False))
    )
    aggregate = xp.sum(weights[..., None] * P, axis=2)
    aggregate = _normalize_intention_xp(aggregate, xp, eps)
    empty = alive_count[..., 0] <= 0.0
    rest_patch = xp.zeros_like(aggregate)
    rest_patch[..., int(Action.REST)] = 1.0
    aggregate = xp.where(empty[..., None], rest_patch, aggregate)
    confidence = xp.clip(xp.sum(weights * C, axis=-1), 0.0, 1.0)
    if needs_phase_context:
        cell_weights = weights.reshape(ph, pw, s, s).transpose(0, 2, 1, 3).reshape(h, w)
        aggregate_action_phase_context_gpu(ds, cfg, cell_weights, confidence)
    write_array(ds, "raqic_patch_record_aggregate", aggregate.astype(ds.health.dtype))
    write_array(ds, "raqic_patch_confidence", confidence.astype(ds.health.dtype))

    patch_intention = ds.arrays.get("raqic_patch_intention", rest_patch).astype(real_dtype)
    eta = float(cfg.raqic.parent_intention_eta)
    patch_intention = _normalize_intention_xp(
        (1.0 - eta) * patch_intention + eta * aggregate, xp, eps
    )
    write_array(ds, "raqic_patch_intention", patch_intention.astype(ds.health.dtype))

    global_raw = xp.sum(aggregate * confidence[..., None], axis=(0, 1))
    global_aggregate = _normalize_intention_xp(global_raw[None, :], xp, eps)[0]
    write_array(ds, "raqic_global_record_aggregate", global_aggregate.astype(ds.health.dtype))
    global_intention = ds.arrays.get(
        "raqic_global_intention", xp.eye(actions, dtype=ds.health.dtype)[int(Action.REST)]
    ).astype(real_dtype)
    global_intention = _normalize_intention_xp(
        ((1.0 - eta) * global_intention + eta * global_aggregate)[None, :], xp, eps
    )[0]
    write_array(ds, "raqic_global_intention", global_intention.astype(ds.health.dtype))

    cell = xp.repeat(xp.repeat(patch_intention, s, axis=0), s, axis=1)[:h, :w, :]
    mixed = _normalize_intention_xp(0.75 * cell + 0.25 * global_intention[None, None, :], xp, eps)
    dead = (ds.health <= 0.0) | ds.obstacle
    rest = xp.zeros_like(mixed)
    rest[..., int(Action.REST)] = 1.0
    mixed = xp.where(dead[..., None], rest, mixed)
    write_array(ds, "raqic_parent_intention", mixed.astype(ds.health.dtype))
    return mixed


def build_raqic_dense_batch_from_device(ds: Any, cfg: Any) -> RAQICDenseBatch:
    xp = ds.xp
    _ensure_parent_intention(ds, cfg)
    h, w = ds.health.shape
    actions = int(ds.possibility.shape[-1])
    eligible = (ds.health > 0) & ~ds.obstacle
    graph_static = bool(ds.metadata.get("graph_static", False))
    if graph_static:
        if "_raqic_eligible_count" not in ds.arrays:
            write_array(ds, "_raqic_eligible_count", xp.zeros((), dtype=xp.int64))
        if "_raqic_processed_count" not in ds.arrays:
            write_array(ds, "_raqic_processed_count", xp.zeros((), dtype=xp.int64))
        ds.arrays["_raqic_eligible_count"][...] = xp.sum(eligible, dtype=xp.int64)
        ds.arrays["_raqic_processed_count"][...] = ds.arrays["_raqic_eligible_count"]
    if graph_static:
        cached = ds.metadata.get("_graph_all_cell_yx")
        if cached is None or cached.shape != (h * w, 2):
            yy = xp.repeat(xp.arange(h, dtype=xp.int32), w)
            xx = xp.tile(xp.arange(w, dtype=xp.int32), h)
            cached = xp.stack([yy, xx], axis=1)
            ds.metadata["_graph_all_cell_yx"] = cached
        yx = cached
    else:
        yx = xp.argwhere(eligible)
    n = int(yx.shape[0])
    if n == 0:
        return RAQICDenseBatch(
            ow_id=xp.zeros((0,), dtype=xp.int64),
            yx=xp.zeros((0, 2), dtype=xp.int32),
            features=xp.zeros((0, len(FEATURE_NAMES)), dtype=xp.float64),
            feature_bins=xp.zeros((0, len(FEATURE_NAMES)), dtype=xp.int32),
            adelic_codes=xp.zeros((0, len(FEATURE_NAMES)), dtype=xp.int32),
            authority_mask=xp.zeros((0, actions), dtype=bool),
            parent_intention=xp.zeros((0, actions), dtype=xp.float64),
            alive_mask=xp.zeros((0,), dtype=bool),
            scale_id=xp.zeros((0,), dtype=xp.int32),
            tick=int(ds.tick),
            feature_names=tuple(FEATURE_NAMES),
            action_names=tuple(action.name for action in Action),
            active_primes=tuple(cfg.raqic.active_primes),
            metadata={"eligible_cells": 0, "processed_cells": 0, "backend": ds.backend.name},
        )
    y = yx[:, 0].astype(xp.int32)
    x = yx[:, 1].astype(xp.int32)
    eps = float(cfg.actions.epsilon)
    resource = xp.clip(ds.resource / max(float(cfg.resources.max_resource), eps), 0.0, 1.0)
    toxin = xp.clip(ds.toxin, 0.0, 1.0)
    food = xp.clip(ds.food, 0.0, 1.0)
    starvation = xp.clip(ds.arrays.get("starvation_debt", xp.zeros_like(ds.health)), 0.0, 1.0)
    signal = (
        xp.clip(xp.mean(ds.signal_reception, axis=-1), 0.0, 1.0)
        if "signal_reception" in ds.arrays
        else xp.zeros_like(ds.health)
    )
    coherence_src = ds.arrays.get("noetic_C", ds.integration)
    coherence = xp.clip(coherence_src, 0.0, 1.0)
    pred = xp.clip(ds.arrays.get("prediction_error", xp.zeros_like(ds.health)), 0.0, 1.0)
    phase = xp.clip(ds.phase % (2.0 * np.pi) / (2.0 * np.pi), 0.0, 1.0)
    danger = xp.clip(
        ds.signal_reception[..., 1]
        if "signal_reception" in ds.arrays and ds.signal_reception.shape[-1] > 1
        else xp.zeros_like(ds.health),
        0.0,
        1.0,
    )
    threat = xp.clip(
        ds.signal_reception[..., 2]
        if "signal_reception" in ds.arrays and ds.signal_reception.shape[-1] > 2
        else xp.zeros_like(ds.health),
        0.0,
        1.0,
    )
    risk = xp.clip(0.45 * toxin + 0.25 * starvation + 0.15 * danger + 0.15 * threat, 0.0, 1.0)
    parent_context = _entropy_concentration_xp(ds.raqic_parent_intention, xp)
    field_map = {
        "resource": resource,
        "risk": risk,
        "memory": xp.clip(ds.memory, 0.0, 1.0),
        "coherence": coherence,
        "phase": phase,
        "boundary": xp.clip(ds.boundary, 0.0, 1.0),
        "signal": signal,
        "prediction_error": pred,
        "parent_context": parent_context,
        "food": food,
        "toxin": toxin,
    }
    if graph_static:
        features = ds.arrays.get("_graph_raqic_features")
        if features is None or features.shape != (n, len(FEATURE_NAMES)):
            features = xp.empty((n, len(FEATURE_NAMES)), dtype=xp.float64)
            write_array(ds, "_graph_raqic_features", features)
        bins = ds.arrays.get("_graph_raqic_bins")
        if bins is None or bins.shape != (n, len(FEATURE_NAMES)):
            bins = xp.empty((n, len(FEATURE_NAMES)), dtype=xp.int32)
            write_array(ds, "_graph_raqic_bins", bins)
        codes = ds.arrays.get("_graph_raqic_codes")
        if codes is None or codes.shape != (n, len(FEATURE_NAMES)):
            codes = xp.empty((n, len(FEATURE_NAMES)), dtype=xp.int32)
            write_array(ds, "_graph_raqic_codes", codes)
    else:
        features = xp.empty((n, len(FEATURE_NAMES)), dtype=xp.float64)
        bins = xp.empty((n, len(FEATURE_NAMES)), dtype=xp.int32)
        codes = xp.empty((n, len(FEATURE_NAMES)), dtype=xp.int32)
    for column, name in enumerate(FEATURE_NAMES):
        features[:, column] = field_map[name][y, x]
    bins[...] = xp.floor(xp.clip(features, 0.0, 1.0) * 255.0).astype(xp.int32)
    codes[...] = bins
    authority = ds.arrays.get("_authority_bool")
    if authority is None:
        authority = ds.arrays.get("pre_authority", xp.zeros((h, w, actions), dtype=bool)).astype(
            bool
        )
    if graph_static:
        selected_authority = ds.arrays.get("_graph_raqic_authority")
        if selected_authority is None or selected_authority.shape != (n, actions):
            selected_authority = xp.empty((n, actions), dtype=bool)
            write_array(ds, "_graph_raqic_authority", selected_authority)
        selected_authority[...] = authority.reshape(n, actions)
        selected_alive = ds.arrays.get("_graph_raqic_alive")
        if selected_alive is None or selected_alive.shape != (n,):
            selected_alive = xp.empty((n,), dtype=bool)
            write_array(ds, "_graph_raqic_alive", selected_alive)
        selected_alive[...] = eligible.reshape(n)
        rest_only = ds.arrays.get("_graph_raqic_rest_only")
        if rest_only is None or rest_only.shape != (n, actions):
            rest_only = xp.zeros((n, actions), dtype=bool)
            rest_only[:, int(Action.REST)] = True
            write_array(ds, "_graph_raqic_rest_only", rest_only)
        selected_authority[...] = xp.where(selected_alive[:, None], selected_authority, rest_only)
        parent_rows = ds.arrays.get("_graph_raqic_parent_rows")
        if parent_rows is None or parent_rows.shape != (n, actions):
            parent_rows = xp.empty((n, actions), dtype=xp.float64)
            write_array(ds, "_graph_raqic_parent_rows", parent_rows)
        parent_rows[...] = ds.raqic_parent_intention.reshape(n, actions)
        occupancy = ds.arrays.get("occupancy")
        ow_id = ds.arrays.get("_graph_raqic_ow_ids")
        if ow_id is None or ow_id.shape != (n,):
            ow_id = xp.empty((n,), dtype=xp.int64)
            write_array(ds, "_graph_raqic_ow_ids", ow_id)
        flat_indices = ds.arrays.get("_graph_raqic_flat_indices")
        if flat_indices is None or flat_indices.shape != (n,):
            flat_indices = xp.arange(n, dtype=xp.int64)
            write_array(ds, "_graph_raqic_flat_indices", flat_indices)
        if occupancy is None:
            ow_id[...] = flat_indices
        else:
            flat_occupancy = occupancy.reshape(n)
            ow_id[...] = xp.where(flat_occupancy >= 0, flat_occupancy, flat_indices).astype(
                xp.int64
            )
        scale_id = ds.arrays.get("_graph_raqic_scale_id")
        if scale_id is None or scale_id.shape != (n,):
            scale_id = xp.zeros((n,), dtype=xp.int32)
            write_array(ds, "_graph_raqic_scale_id", scale_id)
    else:
        selected_authority = authority[y, x, :].astype(bool)
        selected_alive = eligible[y, x]
        parent_rows = ds.raqic_parent_intention[y, x, :].astype(xp.float64)
        occupancy = ds.arrays.get("occupancy", xp.arange(h * w, dtype=xp.int64).reshape(h, w))
        ow_id = xp.where(
            occupancy[y, x] >= 0, occupancy[y, x], y.astype(xp.int64) * int(w) + x.astype(xp.int64)
        ).astype(xp.int64)
        scale_id = xp.zeros((n,), dtype=xp.int32)
    if graph_static:
        ensure_actualization_graph_buffers_gpu(ds, cfg)
        ds.arrays["_graph_raqic_utilities"][...] = ds.arrays["pre_utilities"].reshape(n, actions)
        if "raqic_parent_action_phase" in ds.arrays:
            ds.arrays["_graph_raqic_parent_action_phase"][...] = ds.arrays[
                "raqic_parent_action_phase"
            ].reshape(n, actions)
        if "raqic_parent_action_coherence" in ds.arrays:
            ds.arrays["_graph_raqic_parent_action_coherence"][...] = ds.arrays[
                "raqic_parent_action_coherence"
            ].reshape(n, actions)

    return RAQICDenseBatch(
        ow_id=ow_id,
        yx=yx.astype(xp.int32),
        features=features,
        feature_bins=bins,
        adelic_codes=codes,
        authority_mask=selected_authority,
        parent_intention=parent_rows,
        alive_mask=selected_alive.astype(bool),
        scale_id=scale_id,
        tick=int(ds.tick),
        feature_names=tuple(FEATURE_NAMES),
        action_names=tuple(action.name for action in Action),
        active_primes=tuple(cfg.raqic.active_primes),
        action_utilities=(
            None
            if "pre_utilities" not in ds.arrays
            else (
                ds.arrays["_graph_raqic_utilities"]
                if graph_static
                else ds.arrays["pre_utilities"][y, x, :].astype(xp.float64)
            )
        ),
        parent_action_phase=(
            None
            if "raqic_parent_action_phase" not in ds.arrays
            else (
                ds.arrays["_graph_raqic_parent_action_phase"]
                if graph_static
                else ds.arrays["raqic_parent_action_phase"][y, x, :].astype(xp.float64)
            )
        ),
        parent_action_coherence=(
            None
            if "raqic_parent_action_coherence" not in ds.arrays
            else (
                ds.arrays["_graph_raqic_parent_action_coherence"]
                if graph_static
                else ds.arrays["raqic_parent_action_coherence"][y, x, :].astype(xp.float64)
            )
        ),
        interference_amplitude_output=(
            ds.arrays.get("_graph_raqic_amplitudes") if graph_static else None
        ),
        interference_left_scratch=(
            ds.arrays.get("_graph_raqic_pair_left_scratch") if graph_static else None
        ),
        interference_right_scratch=(
            ds.arrays.get("_graph_raqic_pair_right_scratch") if graph_static else None
        ),
        metadata={
            "eligible_cells": n if not graph_static else "device_count",
            "processed_cells": n,
            "backend": ds.backend.name,
            "graph_static_all_cells": graph_static,
        },
    )


def run_raqic_gpu_stage(ds: Any, cfg: Any) -> dict[str, Any]:
    xp = ds.xp
    prepare_cross_scale_context_gpu(ds, cfg)
    batch = build_raqic_dense_batch_from_device(ds, cfg)
    backend_name = "cupy" if ds.is_gpu else "numpy"
    defer_host = bool(ds.metadata.get("defer_host_metrics", False))
    phase_mode = str(getattr(cfg.raqic, "full_gpu_phase_mode", "scalar_reference"))
    phase_policy = str(getattr(cfg.raqic, "full_gpu_phase_policy", "audit_or_visual"))
    visual_phase = str(getattr(cfg.visualization, "overlay", "")) == "phase"
    validation_due = bool(getattr(cfg.raqic, "gpu_validate_qiskit", False)) and (
        int(getattr(cfg.raqic, "full_gpu_validation_every", 0)) <= 0
        or int(ds.tick) % max(1, int(getattr(cfg.raqic, "full_gpu_validation_every", 1))) == 0
    )
    audit_phase = (
        bool(getattr(cfg.raqic, "store_density_diagnostics", False))
        or bool(getattr(cfg.raqic, "debug_store_full_records", False))
        or bool(getattr(cfg.raqic, "gpu_validate_cpu", False))
        or validation_due
    )
    scientific_stage_parity = bool(ds.metadata.get("scientific_stage_parity", False))
    extension_needs_phase = (
        float(getattr(cfg.raqic, "phase_resonance_coupling", 0.0)) != 0.0
        or float(getattr(cfg.raqic, "interference_mixer_strength", 0.0)) != 0.0
    )
    if scientific_stage_parity or extension_needs_phase:
        # Phase is part of the RAQIC scientific evidence contract. Production
        # execution may omit it when no audit/visual consumer exists, but a
        # differential certificate must compute it on both backends.
        compute_phase = True
    elif phase_policy == "always":
        compute_phase = True
    elif phase_policy == "skip":
        compute_phase = False
    else:
        compute_phase = bool(audit_phase or visual_phase or (not defer_host))
    host_diagnostics = not defer_host or bool(getattr(cfg.raqic, "gpu_validate_cpu", False))
    engine = RAQICDenseDecisionEngine(
        RAQICDenseExecutionConfig(
            seed=int(cfg.world.seed),
            beta_intention=float(cfg.raqic.beta_intention),
            temperature=float(cfg.raqic.action_temperature),
            epsilon_adelic=float(cfg.raqic.epsilon_adelic),
            prime_weights=dict(cfg.raqic.prime_weights),
            precision=str(getattr(cfg.raqic, "full_gpu_precision", cfg.raqic.gpu_precision)),
            backend=backend_name,
            strict_gpu=bool(cfg.raqic.strict_gpu and backend_name == "cupy"),
            audit_limit=int(cfg.raqic.gpu_audit_limit),
            tolerance=float(cfg.raqic.gpu_probability_tolerance),
            phase_mode=phase_mode,
            compute_phase=compute_phase,
            host_diagnostics=host_diagnostics,
            actualization=_actualization_config_from_cfg(cfg),
        )
    )
    result = engine.decide_batch(batch)
    h, w = ds.health.shape
    actions = int(ds.possibility.shape[-1])
    real_dtype = raqic_backend_real_dtype(cfg, xp)
    graph_static = bool(ds.metadata.get("graph_static", False))

    def output(name: str, shape: Any, dtype: Any, fill_value: Any = 0) -> Any:
        current = ds.arrays.get(name)
        if (
            current is None
            or tuple(current.shape) != tuple(shape)
            or current.dtype != xp.dtype(dtype)
        ):
            current = xp.empty(shape, dtype=dtype)
            write_array(ds, name, current)
        current.fill(fill_value)
        return current

    record_action_dtype = ds.arrays.get("raqic_record_action", ds.readout).dtype
    record_readout_dtype = ds.arrays.get("raqic_record_readout", ds.readout).dtype
    evidence_dtype = ds.arrays.get("raqic_record_confidence", ds.health).dtype
    audit_width = int(result.audit_flags.shape[-1]) if result.audit_flags.ndim == 2 else 8
    if graph_static:
        probs = output("raqic_probabilities", (h, w, actions), real_dtype, 0.0)
        probs[..., int(Action.REST)] = 1.0
        readout = output("raqic_readout", (h, w), xp.int32, int(Action.REST))
        record_action = output("raqic_record_action", (h, w), record_action_dtype, int(Action.REST))
        record_readout = output(
            "raqic_record_readout", (h, w), record_readout_dtype, int(Action.REST)
        )
        scores = output("raqic_score", (h, w, actions), real_dtype, 0.0)
        phases = output("raqic_phase", (h, w, actions), real_dtype, 0.0)
        conf = output("raqic_record_confidence", (h, w), evidence_dtype, 0.0)
        trace_error = output("raqic_trace_error", (h, w), evidence_dtype, 0.0)
        min_eigenvalue = output("raqic_min_eigenvalue", (h, w), evidence_dtype, 0.0)
        audit_flags = output("raqic_audit_flags", (h, w, audit_width), xp.int32, 0)
        backend_code = output("raqic_backend_code", (h, w), xp.int32, 0)
    else:
        probs = xp.zeros((h, w, actions), dtype=real_dtype)
        probs[..., int(Action.REST)] = 1.0
        readout = xp.full((h, w), int(Action.REST), dtype=xp.int32)
        record_action = xp.full((h, w), int(Action.REST), dtype=record_action_dtype)
        record_readout = xp.full((h, w), int(Action.REST), dtype=record_readout_dtype)
        scores = xp.zeros((h, w, actions), dtype=real_dtype)
        phases = xp.zeros((h, w, actions), dtype=real_dtype)
        conf = xp.zeros((h, w), dtype=evidence_dtype)
        trace_error = xp.zeros((h, w), dtype=evidence_dtype)
        min_eigenvalue = xp.zeros((h, w), dtype=evidence_dtype)
        audit_flags = xp.zeros((h, w, audit_width), dtype=xp.int32)
        backend_code = xp.zeros((h, w), dtype=xp.int32)
    diagnostic_enabled = bool(
        str(getattr(cfg.raqic, "actualization_variant", "stable_baseline")) != "stable_baseline"
        or bool(getattr(cfg.raqic, "experimental_shadow_only", False))
        or bool(getattr(cfg.raqic, "record_actualization_diagnostics", False))
    )
    diagnostics: dict[str, Any] = {}
    if diagnostic_enabled:
        vector_names = (
            "raqic_pre_mixer_probabilities",
            "raqic_utility_innovation",
            "raqic_phase_alignment",
            "raqic_resonant_parent_intention",
            "raqic_shadow_probabilities",
        )
        scalar_names = (
            "raqic_interference_delta_l1",
            "raqic_policy_kl",
            "raqic_utility_projection_fraction",
            "raqic_utility_score_cosine",
            "raqic_utility_orthogonality_residual",
            "raqic_utility_innovation_norm",
            "raqic_interference_norm_error",
            "raqic_interference_illegal_mass",
        )
        if graph_static:
            diagnostics = {
                name: output(name, (h, w, actions), real_dtype, 0.0) for name in vector_names
            }
            diagnostics.update(
                {name: output(name, (h, w), real_dtype, 0.0) for name in scalar_names}
            )
            diagnostics["raqic_shadow_readout"] = output(
                "raqic_shadow_readout", (h, w), xp.int32, int(Action.REST)
            )
        else:
            diagnostics = {
                name: xp.zeros((h, w, actions), dtype=real_dtype) for name in vector_names
            }
            diagnostics.update({name: xp.zeros((h, w), dtype=real_dtype) for name in scalar_names})
            diagnostics["raqic_shadow_readout"] = xp.full((h, w), int(Action.REST), dtype=xp.int32)
        diagnostics["raqic_pre_mixer_probabilities"][..., int(Action.REST)] = 1.0
        diagnostics["raqic_resonant_parent_intention"][..., int(Action.REST)] = 1.0
        diagnostics["raqic_shadow_probabilities"][..., int(Action.REST)] = 1.0

    if batch.n:
        yy = batch.yx[:, 0]
        xx = batch.yx[:, 1]
        probs[yy, xx, :] = result.probabilities
        readout[yy, xx] = result.readout
        record_action[yy, xx] = result.readout.astype(record_action.dtype)
        record_readout[yy, xx] = xp.argmax(result.probabilities, axis=1).astype(
            record_readout.dtype
        )
        scores[yy, xx, :] = result.scores
        phases[yy, xx, :] = result.phases
        conf[yy, xx] = result.confidence
        trace_error[yy, xx] = result.trace_error
        min_eigenvalue[yy, xx] = result.min_eigenvalue
        audit_flags[yy, xx, :] = result.audit_flags
        backend_code[yy, xx] = result.backend_code
        if diagnostic_enabled:
            row_fields = {
                "raqic_pre_mixer_probabilities": result.pre_mixer_probabilities,
                "raqic_utility_innovation": result.utility_innovation,
                "raqic_phase_alignment": result.phase_alignment,
                "raqic_resonant_parent_intention": result.resonant_parent_intention,
                "raqic_interference_delta_l1": result.interference_delta_l1,
                "raqic_policy_kl": result.policy_kl,
                "raqic_utility_projection_fraction": result.utility_projection_fraction,
                "raqic_utility_score_cosine": result.utility_score_cosine,
                "raqic_utility_orthogonality_residual": result.utility_orthogonality_residual,
                "raqic_utility_innovation_norm": result.utility_innovation_norm,
                "raqic_interference_norm_error": result.interference_norm_error,
                "raqic_interference_illegal_mass": result.interference_illegal_mass,
                "raqic_shadow_probabilities": result.shadow_probabilities,
                "raqic_shadow_readout": result.shadow_readout,
            }
            for name, values in row_fields.items():
                if values is not None:
                    diagnostics[name][yy, xx, ...] = values
    dead = (ds.health <= 0.0) | ds.obstacle
    if graph_static:
        probs[...] = xp.where(dead[..., None], 0.0, probs)
        probs[..., int(Action.REST)] = xp.where(dead, 1.0, probs[..., int(Action.REST)])
        readout[...] = xp.where(dead, int(Action.REST), readout)
        record_action[...] = xp.where(dead, int(Action.REST), record_action)
        record_readout[...] = xp.where(dead, int(Action.REST), record_readout)
        trace_error[...] = xp.where(dead, 0.0, trace_error)
        min_eigenvalue[...] = xp.where(dead, 0.0, min_eigenvalue)
        audit_flags[...] = xp.where(dead[..., None], 0, audit_flags)
        backend_code[...] = xp.where(dead, 0, backend_code)
    else:
        probs = xp.where(dead[..., None], 0.0, probs)
        probs[..., int(Action.REST)] = xp.where(dead, 1.0, probs[..., int(Action.REST)])
        readout = xp.where(dead, int(Action.REST), readout)
        record_action = xp.where(dead, int(Action.REST), record_action)
        record_readout = xp.where(dead, int(Action.REST), record_readout)
        trace_error = xp.where(dead, 0.0, trace_error)
        min_eigenvalue = xp.where(dead, 0.0, min_eigenvalue)
        audit_flags = xp.where(dead[..., None], 0, audit_flags)
        backend_code = xp.where(dead, 0, backend_code)
    write_array(ds, "raqic_probabilities", probs)
    write_array(ds, "raqic_readout", readout)
    write_array(ds, "raqic_record_action", record_action)
    write_array(ds, "raqic_record_readout", record_readout)
    if graph_static:
        ds.arrays["readout"][...] = readout.astype(ds.readout.dtype)
        ds.arrays["possibility"][...] = probs.astype(ds.possibility.dtype)
    else:
        write_array(ds, "readout", readout.astype(ds.readout.dtype))
        write_array(ds, "possibility", probs.astype(ds.possibility.dtype))
    write_array(ds, "raqic_score", scores)
    write_array(ds, "raqic_phase", phases)
    write_array(ds, "raqic_record_confidence", conf)
    write_array(ds, "raqic_trace_error", trace_error)
    write_array(ds, "raqic_min_eigenvalue", min_eigenvalue)
    write_array(ds, "raqic_audit_flags", audit_flags)
    write_array(ds, "raqic_backend_code", backend_code)
    for name, values in diagnostics.items():
        if values.ndim == 3:
            values = xp.where(dead[..., None], 0.0, values)
            if name in {
                "raqic_pre_mixer_probabilities",
                "raqic_resonant_parent_intention",
                "raqic_shadow_probabilities",
            }:
                values[..., int(Action.REST)] = xp.where(dead, 1.0, values[..., int(Action.REST)])
        else:
            values = xp.where(
                dead, int(Action.REST) if name == "raqic_shadow_readout" else 0.0, values
            )
        write_array(ds, name, values)
    return {
        "eligible_cells": None if graph_static else int(batch.metadata.get("eligible_cells", 0)),
        "processed_cells": None if graph_static else int(batch.metadata.get("processed_cells", 0)),
        "device_count_deferred": graph_static,
        "backend": backend_name,
        "phase_mode": phase_mode,
        "phase_computed": compute_phase,
        "phase_skipped_reason": None
        if compute_phase
        else "projective_readout_probability_phase_invariant",
        "result_metadata": result.metadata,
    }


def readout_to_action_masks_gpu(ds: Any) -> Any:
    return {action.name.lower(): ds.readout == int(action) for action in Action}
