"""Patch, region, and global aggregation utilities.

This module implements the bottom-up part of the fractal/mosaic layer:
cell-level observer windows are summarized into patch windows, and patch
windows are summarized into one apex/global state. All computations are dense
NumPy array operations; no object-per-cell structures are introduced.
"""

from __future__ import annotations

from typing import cast

import numpy as np

from owl.core.actions import Action, GlobalIntention
from owl.core.config import SimulationConfig
from owl.core.constants import DEFAULT_FLOAT_DTYPE, DEFAULT_INT_DTYPE
from owl.core.state import (
    GlobalState,
    PatchState,
    WorldState,
    action_shape,
    channel_shape,
    field_shape,
)
from owl.kernels.circular import weighted_patch_circular_statistics
from owl.kernels.numpy_kernels import normalize_last_axis


def _validate_patch_size(patch_size: int) -> int:
    """Return a validated positive patch size."""
    patch = int(patch_size)
    if patch <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size!r}")
    return patch


def _validate_patchable_shape(shape: tuple[int, int], patch_size: int) -> tuple[int, int]:
    """Validate exact patch tiling and return patch-grid shape."""
    patch = _validate_patch_size(patch_size)
    h, w = shape
    if h % patch or w % patch:
        raise ValueError(
            f"field shape {shape} must be exactly divisible by patch_size={patch}; "
            "patch aggregation never silently crops"
        )
    return h // patch, w // patch


def _block_mean_2d(field: np.ndarray, patch_size: int) -> np.ndarray:
    """Return a float32 patch-wise mean of a two-dimensional field."""
    blocks = block_view_2d(field, patch_size)
    return cast(np.ndarray, blocks.mean(axis=(2, 3), dtype=np.float64).astype(DEFAULT_FLOAT_DTYPE))


def _block_sum_2d(field: np.ndarray, patch_size: int) -> np.ndarray:
    """Return a float32 patch-wise sum of a two-dimensional field."""
    blocks = block_view_2d(field, patch_size)
    return cast(np.ndarray, blocks.sum(axis=(2, 3), dtype=np.float64).astype(DEFAULT_FLOAT_DTYPE))


def _block_mean_3d(field: np.ndarray, patch_size: int) -> np.ndarray:
    """Return a float32 patch-wise mean of a channel/action field.

    Parameters
    ----------
    field:
        Array with shape ``(height, width, depth)``.
    patch_size:
        Positive patch side length.

    Returns
    -------
    np.ndarray
        Array with shape ``(height // patch_size, width // patch_size, depth)``.
    """
    array = np.asarray(field)
    if array.ndim != 3:
        raise ValueError(f"expected a 3D field, got shape {array.shape}")
    h, w, depth = array.shape
    _validate_patchable_shape((h, w), patch_size)
    patch = int(patch_size)
    ph, pw = h // patch, w // patch
    blocks = array.reshape(ph, patch, pw, patch, depth).swapaxes(1, 2)
    return cast(np.ndarray, blocks.mean(axis=(2, 3), dtype=np.float64).astype(DEFAULT_FLOAT_DTYPE))


def _weighted_block_mean_2d(field: np.ndarray, weights: np.ndarray, patch_size: int) -> np.ndarray:
    """Return patch-wise weighted mean with a zero-safe fallback to zero."""
    values = np.asarray(field, dtype=np.float32)
    w = np.asarray(weights, dtype=np.float32)
    if values.shape != w.shape:
        raise ValueError(
            f"field and weights must have the same shape, got {values.shape} and {w.shape}"
        )

    numerator = _block_sum_2d(values * w, patch_size)
    denominator = _block_sum_2d(w, patch_size)
    out = np.zeros_like(numerator, dtype=DEFAULT_FLOAT_DTYPE)
    np.divide(numerator, denominator, out=out, where=denominator > 0.0)
    return out.astype(DEFAULT_FLOAT_DTYPE)


def _weighted_block_mean_3d(field: np.ndarray, weights: np.ndarray, patch_size: int) -> np.ndarray:
    """Return patch-wise weighted mean for a 3D action/channel tensor."""
    values = np.asarray(field, dtype=np.float32)
    w = np.asarray(weights, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(f"field must be 3D, got shape {values.shape}")
    if values.shape[:2] != w.shape:
        raise ValueError(
            f"weights must have shape {values.shape[:2]} for field {values.shape}, got {w.shape}"
        )

    h, width, depth = values.shape
    patch = _validate_patch_size(patch_size)
    ph, pw = _validate_patchable_shape((h, width), patch)
    value_blocks = values.reshape(ph, patch, pw, patch, depth).swapaxes(1, 2)
    weight_blocks = w.reshape(ph, patch, pw, patch).swapaxes(1, 2)

    numerator = np.sum(value_blocks * weight_blocks[..., None], axis=(2, 3), dtype=np.float64)
    denominator = np.asarray(np.sum(weight_blocks, axis=(2, 3), dtype=np.float64), dtype=np.float64)
    out = np.zeros((ph, pw, depth), dtype=DEFAULT_FLOAT_DTYPE)
    np.divide(numerator, denominator[..., None], out=out, where=denominator[..., None] > 0.0)
    return out.astype(DEFAULT_FLOAT_DTYPE)


def _patch_phase_statistics(
    phase: np.ndarray,
    weights: np.ndarray,
    patch_size: int,
    *,
    resultant_support_epsilon: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return stable patch phase, synchrony, resultant, and support mask."""
    patch_phase, synchrony, resultant, supported = weighted_patch_circular_statistics(
        np.asarray(phase),
        np.asarray(weights),
        patch_size,
        np,
        resultant_support_epsilon=resultant_support_epsilon,
    )
    return (
        np.asarray(patch_phase, dtype=DEFAULT_FLOAT_DTYPE),
        np.asarray(synchrony, dtype=DEFAULT_FLOAT_DTYPE),
        np.asarray(resultant, dtype=np.float64),
        np.asarray(supported, dtype=bool),
    )


def _patch_phase_mean(
    phase: np.ndarray,
    weights: np.ndarray,
    patch_size: int,
    *,
    resultant_support_epsilon: float = 1e-7,
) -> np.ndarray:
    """Return weighted circular phase mean per patch."""
    patch_phase, _, _, _ = _patch_phase_statistics(
        phase,
        weights,
        patch_size,
        resultant_support_epsilon=resultant_support_epsilon,
    )
    return patch_phase


def _patch_phase_synchrony(
    phase: np.ndarray,
    weights: np.ndarray,
    patch_size: int,
    *,
    resultant_support_epsilon: float = 1e-7,
) -> np.ndarray:
    """Return resultant-vector synchrony per patch in [0, 1]."""
    _, synchrony, _, _ = _patch_phase_statistics(
        phase,
        weights,
        patch_size,
        resultant_support_epsilon=resultant_support_epsilon,
    )
    return synchrony


def block_view_2d(field: np.ndarray, patch_size: int) -> np.ndarray:
    """Return a patch-block view of a 2D field.

    Parameters
    ----------
    field:
        Two-dimensional cell-level array with shape ``(height, width)``.
    patch_size:
        Positive patch side length. Both spatial dimensions must be exactly
        divisible by this value.

    Returns
    -------
    np.ndarray
        View-like array with shape
        ``(height // patch_size, width // patch_size, patch_size, patch_size)``.

    Raises
    ------
    ValueError
        If ``field`` is not two-dimensional, ``patch_size`` is not positive, or
        the field cannot be tiled exactly.
    """
    array = np.asarray(field)
    if array.ndim != 2:
        raise ValueError(f"block_view_2d requires a 2D field, got shape {array.shape}")
    patch = _validate_patch_size(patch_size)
    h, w = array.shape
    _validate_patchable_shape((h, w), patch)
    return array.reshape(h // patch, patch, w // patch, patch).swapaxes(1, 2)


def _base_aggregate_patches(state: WorldState, cfg: SimulationConfig) -> PatchState:
    """Aggregate cell arrays into patch-level observer windows.

    Parameters
    ----------
    state:
        Runtime dense state. This function does not mutate ``state``.
    cfg:
        Simulation coefficients. ``cfg.world.patch_size`` defines the patch
        tiling and must divide the grid exactly.

    Returns
    -------
    PatchState
        Patch-level arrays. Scalar fields have shape
        ``(height // patch_size, width // patch_size)``; possibility and policy
        fields have an additional action axis; signal pressure has a channel
        axis.

    Notes
    -----
    Physical state and communication pressure are block means. Possibility is a
    weighted block mean over living cells and is repaired onto the simplex.
    Phase is a weighted circular mean. Empty patches produce neutral values.
    """
    h, w = field_shape(state)
    ah, aw, actions = action_shape(state)
    ch, cw, channels = channel_shape(state)
    if (ah, aw) != (h, w):
        raise ValueError("state.possibility spatial shape must match cell shape")
    if (ch, cw) != (h, w):
        raise ValueError("state.signal spatial shape must match cell shape")
    if channels != cfg.communication.num_channels:
        raise ValueError("state.signal channel axis must match cfg.communication.num_channels")

    patch = _validate_patch_size(cfg.world.patch_size)
    ph, pw = _validate_patchable_shape((h, w), patch)

    alive = ((state.health > 0.0) & (state.boundary > 0.0) & (~state.obstacle)).astype(np.float32)
    weights = alive * (0.10 + 0.90 * np.clip(state.integration, 0.0, 1.0))

    activation = _weighted_block_mean_2d(np.clip(state.activation, 0.0, 1.0), weights, patch)
    memory = _weighted_block_mean_2d(np.clip(state.memory, 0.0, 1.0), weights, patch)
    integration = _weighted_block_mean_2d(np.clip(state.integration, 0.0, 1.0), weights, patch)
    resource = _weighted_block_mean_2d(
        np.clip(state.resource, 0.0, cfg.resources.max_resource), weights, patch
    )
    resource = np.clip(
        resource / max(cfg.resources.max_resource, cfg.actions.epsilon), 0.0, 1.0
    ).astype(DEFAULT_FLOAT_DTYPE)
    health = _weighted_block_mean_2d(np.clip(state.health, 0.0, 1.0), weights, patch)
    boundary = _weighted_block_mean_2d(np.clip(state.boundary, 0.0, 1.0), weights, patch)

    phase, synchrony, _, _ = _patch_phase_statistics(
        state.phase,
        weights,
        patch,
        resultant_support_epsilon=cfg.phase.patch_resultant_support_epsilon,
    )

    possibility = _weighted_block_mean_3d(state.possibility, weights, patch)
    empty_patch = _block_sum_2d(weights, patch) <= 0.0
    if np.any(empty_patch):
        possibility[empty_patch, :] = 0.0
        possibility[empty_patch, int(Action.REST)] = 1.0
    possibility = normalize_last_axis(possibility, epsilon=cfg.actions.epsilon).astype(
        DEFAULT_FLOAT_DTYPE
    )

    signal_pressure = _weighted_block_mean_3d(state.signal_reception, np.maximum(alive, 0.0), patch)
    signal_pressure = np.clip(signal_pressure, 0.0, 1.0).astype(DEFAULT_FLOAT_DTYPE)

    phase_blocks = block_view_2d(state.phase, patch)
    patch_phase_blocks = phase[:, :, None, None]
    phase_alignment = 0.5 + 0.5 * np.cos(phase_blocks - patch_phase_blocks)
    alive_blocks = block_view_2d(alive, patch)
    align_num = np.sum(phase_alignment * alive_blocks, axis=(2, 3), dtype=np.float64)
    align_den = np.sum(alive_blocks, axis=(2, 3), dtype=np.float64)
    coherence = np.zeros((ph, pw), dtype=DEFAULT_FLOAT_DTYPE)
    np.divide(align_num, align_den, out=coherence, where=align_den > 0.0)
    coherence = np.clip(coherence, 0.0, 1.0).astype(DEFAULT_FLOAT_DTYPE)

    # This patch-level metric measures whether cells in a patch
    # are internally aligned enough to be reliable parents for the next pass.
    cross_scale = np.clip(0.5 * synchrony + 0.5 * coherence, 0.0, 1.0).astype(DEFAULT_FLOAT_DTYPE)

    return PatchState(
        activation=activation,
        memory=memory,
        phase=phase,
        possibility=possibility,
        integration=integration,
        resource=resource,
        health=health,
        boundary=boundary,
        signal_pressure=signal_pressure,
        synchrony=synchrony,
        coherence=coherence,
        cross_scale=cross_scale,
        intention=np.zeros((ph, pw), dtype=DEFAULT_INT_DTYPE),
        policy_bias=np.zeros((ph, pw, actions), dtype=DEFAULT_FLOAT_DTYPE),
    )


def _base_aggregate_global(patches: PatchState, cfg: SimulationConfig) -> GlobalState:
    """Aggregate patch state into an apex/global observer window.

    Parameters
    ----------
    patches:
        Patch-level state. This function does not mutate it.
    cfg:
        Simulation coefficients.

    Returns
    -------
    GlobalState
        Apex summary. The global state is a diagnostic and weak-policy source;
        it must not overwrite cell readouts.
    """
    patch_shape = patches.integration.shape
    if len(patch_shape) != 2:
        raise ValueError(f"patches.integration must be 2D, got {patch_shape}")

    actions = len(Action)
    channels = cfg.communication.num_channels

    if patches.possibility.shape != (*patch_shape, actions):
        expected = (*patch_shape, actions)
        raise ValueError(
            f"patches.possibility must have shape {expected}, got {patches.possibility.shape}"
        )
    if patches.signal_pressure.shape != (*patch_shape, channels):
        expected = (*patch_shape, channels)
        raise ValueError(
            f"patches.signal_pressure must have shape {expected}, "
            f"got {patches.signal_pressure.shape}"
        )

    patch_integration = np.clip(patches.integration, 0.0, 1.0).astype(np.float32)
    alive_patch = np.clip(patches.health, 0.0, 1.0) > 0.0

    if alive_patch.any():
        weights = patch_integration * alive_patch.astype(np.float32)
        if np.sum(weights) <= 0.0:
            weights = alive_patch.astype(np.float32)
        weight_sum = float(np.sum(weights))
        integration = float(
            np.sum(patch_integration * weights) / max(weight_sum, cfg.actions.epsilon)
        )
        signal_pressure = np.sum(
            patches.signal_pressure * weights[..., None], axis=(0, 1), dtype=np.float64
        ) / max(weight_sum, cfg.actions.epsilon)
        global_possibility = np.sum(
            patches.possibility * weights[..., None], axis=(0, 1), dtype=np.float64
        ) / max(weight_sum, cfg.actions.epsilon)
    else:
        integration = 0.0
        signal_pressure = np.zeros((channels,), dtype=np.float32)
        global_possibility = np.zeros((actions,), dtype=np.float32)
        global_possibility[int(Action.REST)] = 1.0

    global_possibility = normalize_last_axis(
        np.asarray(global_possibility, dtype=np.float32), epsilon=cfg.actions.epsilon
    )
    readout = int(np.argmax(global_possibility))

    fragmentation = (
        float(np.var(patch_integration.astype(np.float64))) if patch_integration.size else 0.0
    )

    mean_patch_policy = normalize_last_axis(
        patches.possibility.mean(axis=(0, 1)), epsilon=cfg.actions.epsilon
    )
    positive = mean_patch_policy[mean_patch_policy > 0.0]
    if positive.size <= 1:
        diversity = 0.0
    else:
        diversity = float(
            np.clip(
                -np.sum(positive * np.log(positive + cfg.actions.epsilon)) / np.log(actions),
                0.0,
                1.0,
            )
        )

    # Complexity is high when integration is neither fragmented to zero nor
    # perfectly homogeneous; this remains an operational metric only.
    complexity = float(
        np.clip(integration * (1.0 - fragmentation) * (0.5 + 0.5 * diversity), 0.0, 1.0)
    )

    return GlobalState(
        integration=float(np.clip(integration, 0.0, 1.0)),
        readout=readout,
        intention=int(GlobalIntention.REST),
        fragmentation=float(np.clip(fragmentation, 0.0, 1.0)),
        diversity=diversity,
        complexity=complexity,
        signal_pressure=np.clip(signal_pressure, 0.0, 1.0).astype(DEFAULT_FLOAT_DTYPE),
        policy_bias=np.zeros((actions,), dtype=DEFAULT_FLOAT_DTYPE),
    )


def upsample_patch_field(field: np.ndarray, patch_size: int) -> np.ndarray:
    """Broadcast a patch field back to cell resolution.

    Parameters
    ----------
    field:
        Patch-level array with shape ``(patch_height, patch_width)`` or
        ``(patch_height, patch_width, depth)``.
    patch_size:
        Positive patch side length.

    Returns
    -------
    np.ndarray
        Repeated array with first two axes expanded by ``patch_size``.
    """
    patch = _validate_patch_size(patch_size)
    array = np.asarray(field)
    if array.ndim < 2:
        raise ValueError(f"field must have at least two patch axes, got shape {array.shape}")
    upsampled = np.repeat(np.repeat(array, patch, axis=0), patch, axis=1)
    return upsampled.astype(array.dtype, copy=False)


def upsample_patch_bias(policy_bias: np.ndarray, patch_size: int) -> np.ndarray:
    """Broadcast patch action-policy bias to cell resolution.

    Parameters
    ----------
    policy_bias:
        Patch-level action bias with shape ``(patch_height, patch_width, len(Action))``.
    patch_size:
        Positive patch side length.

    Returns
    -------
    np.ndarray
        Cell-level bias with shape
        ``(patch_height * patch_size, patch_width * patch_size, len(Action))``.
    """
    array = np.asarray(policy_bias)
    if array.ndim != 3:
        raise ValueError(f"policy_bias must be 3D, got shape {array.shape}")
    if array.shape[-1] != len(Action):
        raise ValueError(
            f"policy_bias last axis must equal len(Action)={len(Action)}, got {array.shape[-1]}"
        )
    return upsample_patch_field(array, patch_size).astype(array.dtype, copy=False)


# --- Advanced build overrides ------------------------------------------------
_mvp_aggregate_patches = _base_aggregate_patches
_mvp_aggregate_global = _base_aggregate_global


def _advanced_aggregate_patches(state: WorldState, cfg: SimulationConfig) -> PatchState:
    """Aggregate patches with optional dynamic parent-id centroids/errors."""
    patches = _mvp_aggregate_patches(state, cfg)
    if not getattr(cfg.hierarchy, "dynamic_patches", False):
        return patches

    from owl.core.advanced import ensure_advanced_fields

    ensure_advanced_fields(state, cfg)
    ph, pw = patches.integration.shape
    indices = np.indices(state.health.shape)
    yy = indices[0]
    xx = indices[1]
    cy = np.zeros((ph, pw), dtype=np.float32)
    cx = np.zeros((ph, pw), dtype=np.float32)
    pe = np.zeros((ph, pw), dtype=np.float32)
    old_cy = state.patches.centroid_y if isinstance(state.patches.centroid_y, np.ndarray) else cy
    old_cx = state.patches.centroid_x if isinstance(state.patches.centroid_x, np.ndarray) else cx
    smoothing = np.float32(cfg.hierarchy.centroid_smoothing)
    for pid in range(ph * pw):
        py = pid // pw
        px = pid % pw
        mask = (state.parent_id == pid) & (state.health > 0.0) & (~state.obstacle)
        if np.any(mask):
            raw_y = np.mean(yy[mask], dtype=np.float64)
            raw_x = np.mean(xx[mask], dtype=np.float64)
            cy[py, px] = (1.0 - smoothing) * old_cy[py, px] + smoothing * raw_y
            cx[py, px] = (1.0 - smoothing) * old_cx[py, px] + smoothing * raw_x
            pe[py, px] = np.float32(
                np.mean(np.abs(state.integration[mask] - patches.integration[py, px]))
            )
        else:
            cy[py, px] = old_cy[py, px]
            cx[py, px] = old_cx[py, px]
            pe[py, px] = 0.0
    patches.centroid_y = cy
    patches.centroid_x = cx
    patches.velocity_y = cy - old_cy
    patches.velocity_x = cx - old_cx
    patches.prediction_error = np.clip(pe, 0.0, 1.0)
    return patches


def _advanced_aggregate_global(patches: PatchState, cfg: SimulationConfig) -> GlobalState:
    """Aggregate global state; dynamic hierarchy includes predictive error."""
    g = _mvp_aggregate_global(patches, cfg)
    if getattr(cfg.hierarchy, "predictive_topdown", False) and isinstance(
        patches.prediction_error, np.ndarray
    ):
        err = float(np.mean(np.clip(patches.prediction_error, 0.0, 1.0)))
        g.complexity = float(np.clip(g.complexity * (1.0 - 0.25 * err), 0.0, 1.0))
        g.fragmentation = float(np.clip(max(g.fragmentation, err), 0.0, 1.0))
    return g


# --- Decision-homeostasis pass overrides -------------------------------------
def _normalized_entropy(probability: np.ndarray, epsilon: float) -> np.ndarray:
    """Return normalized entropy over the last axis."""
    p = np.clip(np.asarray(probability, dtype=np.float32), 0.0, 1.0)
    denom = np.maximum(np.sum(p, axis=-1, keepdims=True), epsilon)
    p = p / denom
    k = max(int(p.shape[-1]), 2)
    entropy = -np.sum(np.where(p > 0.0, p * np.log(np.maximum(p, epsilon)), 0.0), axis=-1) / np.log(
        float(k)
    )
    return cast(np.ndarray, np.clip(entropy, 0.0, 1.0).astype(np.float32))


def _decision_cell_noetic_components(
    state: WorldState, cfg: SimulationConfig
) -> tuple[np.ndarray, ...]:
    """Compute bounded cell-level noetic decomposition components.

    B=boundary/viability, M=memory, P=possibility entropy, C=same-scale coherence
    proxy, K=cross-scale parent alignment, Theta=threshold/carrying resistance,
    N=bounded integration diagnostic.
    """
    from owl.core.advanced import ensure_advanced_fields

    ensure_advanced_fields(state, cfg)
    assert state.parent_weight is not None
    assert state.prediction_error is not None
    resource = np.clip(
        state.resource / max(float(cfg.resources.max_resource), float(cfg.actions.epsilon)),
        0.0,
        1.0,
    )
    B = np.clip(0.35 * state.health + 0.35 * state.boundary + 0.30 * resource, 0.0, 1.0).astype(
        np.float32
    )
    M = np.clip(state.memory, 0.0, 1.0).astype(np.float32)
    P = _normalized_entropy(state.possibility, cfg.actions.epsilon)
    local = np.clip(
        0.5 * state.integration + 0.5 * np.clip(state.coupling_strength, 0.0, 1.0), 0.0, 1.0
    )
    C = local.astype(np.float32)
    parent = upsample_patch_field(
        np.clip(state.patches.integration, 0.0, 1.0), cfg.world.patch_size
    )
    if parent.shape != state.health.shape:
        parent = np.zeros_like(state.health, dtype=np.float32)
    K = np.clip(0.5 * parent + 0.5 * np.clip(state.parent_weight, 0.0, 1.0), 0.0, 1.0).astype(
        np.float32
    )
    patch_pressure = np.zeros_like(state.health, dtype=np.float32)
    if isinstance(state.patches.patch_carrying_pressure, np.ndarray):
        patch_pressure = upsample_patch_field(
            np.clip(state.patches.patch_carrying_pressure, 0.0, 1.0), cfg.world.patch_size
        )
    Theta = np.clip(0.5 * state.threshold + 0.5 * patch_pressure, 0.0, 1.0).astype(np.float32)
    crisis = np.zeros_like(state.health, dtype=np.float32)
    if isinstance(state.patches.patch_crisis, np.ndarray):
        crisis = upsample_patch_field(
            np.clip(state.patches.patch_crisis, 0.0, 1.0), cfg.world.patch_size
        )
    E = np.clip(0.5 * crisis + 0.5 * np.clip(state.prediction_error, 0.0, 1.0), 0.0, 1.0)
    N = np.clip(
        0.22 * B + 0.16 * M + 0.15 * P + 0.20 * C + 0.20 * K - 0.17 * Theta - 0.20 * E, 0.0, 1.0
    ).astype(np.float32)
    alive = (state.health > 0.0) & (~state.obstacle)
    for arr in (B, M, P, C, K, Theta, N):
        arr[~alive] = 0.0
    return B, M, P, C, K, Theta, N


def aggregate_patches(state: WorldState, cfg: SimulationConfig) -> PatchState:
    """Aggregate patches with lower-state homeostasis and noetic decomposition."""
    patches = _mvp_aggregate_patches(state, cfg)
    from owl.core.advanced import ensure_advanced_fields

    ensure_advanced_fields(state, cfg)
    assert state.noetic_B is not None
    assert state.noetic_M is not None
    assert state.noetic_P is not None
    assert state.noetic_C is not None
    assert state.noetic_K is not None
    assert state.noetic_Theta is not None
    assert state.noetic_N is not None
    patch = _validate_patch_size(cfg.world.patch_size)
    ph, pw = patches.integration.shape
    alive = ((state.health > 0.0) & (~state.obstacle)).astype(np.float32)
    resource = np.clip(
        state.resource / max(float(cfg.resources.max_resource), float(cfg.actions.epsilon)),
        0.0,
        1.0,
    )
    food = np.clip(state.food, 0.0, 1.0)
    starv = np.clip(
        state.starvation_debt if isinstance(state.starvation_debt, np.ndarray) else 1.0 - resource,
        0.0,
        1.0,
    )
    density = _block_mean_2d(alive, patch)
    food_mean = _block_mean_2d(food, patch)
    res_mean = _weighted_block_mean_2d(resource, alive, patch)
    health_mean = _weighted_block_mean_2d(np.clip(state.health, 0.0, 1.0), alive, patch)
    boundary_mean = _weighted_block_mean_2d(np.clip(state.boundary, 0.0, 1.0), alive, patch)
    starv_mean = _weighted_block_mean_2d(starv, alive, patch)

    readout = state.readout
    repro_frac = _weighted_block_mean_2d(
        (readout == int(Action.REPRODUCE)).astype(np.float32), alive, patch
    )
    move_mask = np.zeros_like(state.health, dtype=np.float32)
    for action in __import__("owl.core.actions", fromlist=["MOVE_DELTAS"]).MOVE_DELTAS:
        move_mask += (readout == int(action)).astype(np.float32)
    move_frac = _weighted_block_mean_2d(np.clip(move_mask, 0.0, 1.0), alive, patch)
    feed_frac = _weighted_block_mean_2d(
        (readout == int(Action.FEED)).astype(np.float32), alive, patch
    )
    death_pressure = _block_mean_2d(
        state.last_death_mask.astype(np.float32)
        if isinstance(state.last_death_mask, np.ndarray)
        else np.zeros_like(state.health, dtype=np.float32),
        patch,
    )

    if getattr(cfg.cross_scale_homeostasis, "enabled", False):
        carrying = np.clip(
            cfg.cross_scale_homeostasis.crowding_pressure_weight * density
            + cfg.cross_scale_homeostasis.food_deficit_weight * (1.0 - food_mean)
            + cfg.cross_scale_homeostasis.starvation_pressure_weight * starv_mean
            + cfg.cross_scale_homeostasis.reproduction_pressure_weight * repro_frac,
            0.0,
            1.0,
        ).astype(np.float32)
        crisis = np.clip(
            0.45 * starv_mean + 0.25 * (1.0 - res_mean) + 0.20 * carrying + 0.10 * death_pressure,
            0.0,
            1.0,
        ).astype(np.float32)
    else:
        carrying = np.zeros((ph, pw), dtype=np.float32)
        crisis = np.zeros((ph, pw), dtype=np.float32)

    Bc, Mc, Pc, Cc, Kc, Thetac, Nc = _decision_cell_noetic_components(state, cfg)
    state.noetic_B[...] = Bc
    state.noetic_M[...] = Mc
    state.noetic_P[...] = Pc
    state.noetic_C[...] = Cc
    state.noetic_K[...] = Kc
    state.noetic_Theta[...] = Thetac
    state.noetic_N[...] = Nc

    weights = np.maximum(alive, 0.0)
    nB = _weighted_block_mean_2d(Bc, weights, patch)
    nM = _weighted_block_mean_2d(Mc, weights, patch)
    nP = _weighted_block_mean_2d(Pc, weights, patch)
    nC = _weighted_block_mean_2d(Cc, weights, patch)
    nK = _weighted_block_mean_2d(Kc, weights, patch)
    nTheta = _weighted_block_mean_2d(Thetac, weights, patch)
    nN = _weighted_block_mean_2d(Nc, weights, patch)

    if getattr(cfg.cross_scale_homeostasis, "enabled", False):
        # Noetic patch integration is viability-weighted and explicitly reduced
        # by lower-level crisis/carrying pressure.
        patches.integration = np.clip(
            0.75 * nN + 0.25 * patches.integration - 0.30 * crisis, 0.0, 1.0
        ).astype(DEFAULT_FLOAT_DTYPE)
        patches.resource = res_mean.astype(DEFAULT_FLOAT_DTYPE)
        patches.health = health_mean.astype(DEFAULT_FLOAT_DTYPE)
        patches.boundary = boundary_mean.astype(DEFAULT_FLOAT_DTYPE)

    patches.alive_density = density.astype(DEFAULT_FLOAT_DTYPE)
    patches.food_mean = food_mean.astype(DEFAULT_FLOAT_DTYPE)
    patches.starvation_debt_mean = starv_mean.astype(DEFAULT_FLOAT_DTYPE)
    patches.reproduction_fraction = repro_frac.astype(DEFAULT_FLOAT_DTYPE)
    patches.movement_fraction = move_frac.astype(DEFAULT_FLOAT_DTYPE)
    patches.feed_fraction = feed_frac.astype(DEFAULT_FLOAT_DTYPE)
    patches.death_pressure = death_pressure.astype(DEFAULT_FLOAT_DTYPE)
    patches.patch_crisis = crisis.astype(DEFAULT_FLOAT_DTYPE)
    patches.patch_carrying_pressure = carrying.astype(DEFAULT_FLOAT_DTYPE)
    patches.noetic_B = nB.astype(DEFAULT_FLOAT_DTYPE)
    patches.noetic_M = nM.astype(DEFAULT_FLOAT_DTYPE)
    patches.noetic_P = nP.astype(DEFAULT_FLOAT_DTYPE)
    patches.noetic_C = nC.astype(DEFAULT_FLOAT_DTYPE)
    patches.noetic_K = nK.astype(DEFAULT_FLOAT_DTYPE)
    patches.noetic_Theta = nTheta.astype(DEFAULT_FLOAT_DTYPE)
    patches.noetic_N = nN.astype(DEFAULT_FLOAT_DTYPE)

    if getattr(cfg.hierarchy, "dynamic_patches", False):
        indices = np.indices(state.health.shape)
        yy = indices[0]
        xx = indices[1]
        cy = np.zeros((ph, pw), dtype=np.float32)
        cx = np.zeros((ph, pw), dtype=np.float32)
        pe = np.zeros((ph, pw), dtype=np.float32)
        old_cy = (
            state.patches.centroid_y if isinstance(state.patches.centroid_y, np.ndarray) else cy
        )
        old_cx = (
            state.patches.centroid_x if isinstance(state.patches.centroid_x, np.ndarray) else cx
        )
        smoothing = np.float32(cfg.hierarchy.centroid_smoothing)
        for pid in range(ph * pw):
            py = pid // pw
            px = pid % pw
            mask = (state.parent_id == pid) & (state.health > 0.0) & (~state.obstacle)
            if np.any(mask):
                raw_y = np.mean(yy[mask], dtype=np.float64)
                raw_x = np.mean(xx[mask], dtype=np.float64)
                cy[py, px] = (1.0 - smoothing) * old_cy[py, px] + smoothing * raw_y
                cx[py, px] = (1.0 - smoothing) * old_cx[py, px] + smoothing * raw_x
                pe[py, px] = np.float32(
                    np.mean(np.abs(state.integration[mask] - patches.integration[py, px]))
                )
            else:
                cy[py, px] = old_cy[py, px]
                cx[py, px] = old_cx[py, px]
                pe[py, px] = 0.0
        patches.centroid_y = cy
        patches.centroid_x = cx
        patches.velocity_y = cy - old_cy
        patches.velocity_x = cx - old_cx
        patches.prediction_error = np.clip(pe + 0.5 * crisis, 0.0, 1.0)

    return patches


def aggregate_global(patches: PatchState, cfg: SimulationConfig) -> GlobalState:
    """Aggregate global apex state with explicit bottom-up crisis pressure."""
    g = _mvp_aggregate_global(patches, cfg)
    patch_integration = np.clip(patches.integration, 0.0, 1.0)
    alive_patch = np.clip(patches.health, 0.0, 1.0) > 0.0
    weights = alive_patch.astype(np.float32) * np.maximum(patch_integration, 0.05)
    denom = float(np.sum(weights))
    if denom <= 0.0:
        g.crisis = 0.0
        g.carrying_pressure = 0.0
        g.starvation_pressure = 0.0
        g.food_deficit = 0.0
        return g

    def wmean(field: np.ndarray | None, default: float = 0.0) -> float:
        if not isinstance(field, np.ndarray):
            return default
        arr = np.clip(field.astype(np.float32), 0.0, 1.0)
        return float(
            np.sum(arr * weights, dtype=np.float64) / max(denom, float(cfg.actions.epsilon))
        )

    starv = wmean(getattr(patches, "starvation_debt_mean", None))
    food_deficit = 1.0 - wmean(getattr(patches, "food_mean", None), default=1.0)
    carrying = wmean(getattr(patches, "patch_carrying_pressure", None))
    crisis = float(np.clip(0.45 * starv + 0.35 * food_deficit + 0.20 * carrying, 0.0, 1.0))
    g.crisis = crisis
    g.carrying_pressure = carrying
    g.starvation_pressure = starv
    g.food_deficit = food_deficit
    g.integration = float(np.clip(g.integration * (1.0 - 0.55 * crisis), 0.0, 1.0))
    g.complexity = float(np.clip(g.complexity * (1.0 - 0.35 * crisis), 0.0, 1.0))
    g.fragmentation = float(np.clip(max(g.fragmentation, crisis), 0.0, 1.0))
    return g
