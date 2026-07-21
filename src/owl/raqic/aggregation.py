from __future__ import annotations

import numpy as np

from owl.core.actions import Action
from owl.core.config import SimulationConfig
from owl.core.constants import DEFAULT_FLOAT_DTYPE
from owl.core.state import WorldState, field_shape
from owl.engine.aggregation import block_view_2d
from owl.raqic.precision import raqic_numpy_dtype
from owl_raqic.math.aggregation import bottom_up_weights
from owl_raqic.math.intentions import normalize_intention


def aggregate_raqic_records_to_patches(state: WorldState, cfg: SimulationConfig) -> np.ndarray:
    assert state.raqic_patch_record_aggregate is not None
    assert state.raqic_patch_confidence is not None
    h, w = field_shape(state)
    patch = int(cfg.world.patch_size)
    ph, pw = h // patch, w // patch
    actions = len(Action)
    decision_dtype = raqic_numpy_dtype(cfg)
    alive = ((state.health > 0.0) & (~state.obstacle)).astype(decision_dtype)
    probs = np.asarray(
        getattr(state, "raqic_probabilities", state.possibility), dtype=decision_dtype
    )
    aggregate = np.zeros((ph, pw, actions), dtype=decision_dtype)
    confidence = np.zeros((ph, pw), dtype=DEFAULT_FLOAT_DTYPE)
    Bblk = block_view_2d(np.clip(state.boundary, 0, 1), patch)
    Rblk = block_view_2d(
        np.clip(state.resource / max(cfg.resources.max_resource, cfg.actions.epsilon), 0, 1), patch
    )
    Csrc = getattr(state, "noetic_C", state.integration)
    Cblk = block_view_2d(np.clip(Csrc, 0, 1), patch)
    Ablk = block_view_2d(alive, patch)
    yy, xx = np.indices((patch, patch))
    dist = np.sqrt(
        (yy.reshape(-1) - (patch - 1) / 2) ** 2 + (xx.reshape(-1) - (patch - 1) / 2) ** 2
    ) / max(float(patch), 1.0)
    for py in range(ph):
        for px in range(pw):
            mask = Ablk[py, px].reshape(-1) > 0
            if not np.any(mask):
                aggregate[py, px, int(Action.REST)] = 1.0
                continue
            W = bottom_up_weights(
                Bblk[py, px].reshape(-1)[mask],
                Cblk[py, px].reshape(-1)[mask],
                Rblk[py, px].reshape(-1)[mask],
                dist[mask],
                eta=1.0,
            )
            cells = probs[py * patch : (py + 1) * patch, px * patch : (px + 1) * patch, :].reshape(
                -1, actions
            )[mask]
            aggregate[py, px, :] = normalize_intention(W @ cells, actions).astype(
                decision_dtype, copy=False
            )
            confidence[py, px] = np.float32(
                np.clip(np.sum(W * Cblk[py, px].reshape(-1)[mask]), 0, 1)
            )
    state.raqic_patch_record_aggregate[...] = aggregate.astype(
        state.raqic_patch_record_aggregate.dtype, copy=False
    )
    state.raqic_patch_confidence[...] = confidence.astype(
        state.raqic_patch_confidence.dtype, copy=False
    )
    return aggregate


def aggregate_raqic_patches_to_global(state: WorldState, cfg: SimulationConfig) -> np.ndarray:
    assert state.raqic_patch_record_aggregate is not None
    assert state.raqic_patch_confidence is not None
    assert state.raqic_global_record_aggregate is not None
    rec = np.asarray(state.raqic_patch_record_aggregate, dtype=np.float32)
    ph, pw, actions = rec.shape
    weights = np.clip(state.raqic_patch_confidence, 0, 1)
    raw = np.sum(rec * weights[..., None], axis=(0, 1))
    if float(np.sum(raw)) <= cfg.actions.epsilon:
        raw = np.zeros((actions,), dtype=np.float32)
        raw[int(Action.REST)] = 1.0
    out = normalize_intention(raw, actions).astype(DEFAULT_FLOAT_DTYPE)
    state.raqic_global_record_aggregate[...] = out
    return out


def update_raqic_patch_intention(state: WorldState, cfg: SimulationConfig) -> np.ndarray:
    assert state.raqic_patch_intention is not None
    assert state.raqic_patch_record_aggregate is not None
    eta = float(cfg.raqic.parent_intention_eta)
    updated = (1 - eta) * state.raqic_patch_intention + eta * state.raqic_patch_record_aggregate
    sums = np.sum(updated, axis=-1, keepdims=True)
    updated = np.divide(updated, sums, out=np.zeros_like(updated), where=sums > cfg.actions.epsilon)
    bad = sums[..., 0] <= cfg.actions.epsilon
    if np.any(bad):
        updated[bad, :] = 0
        updated[bad, int(Action.REST)] = 1
    state.raqic_patch_intention[...] = updated.astype(DEFAULT_FLOAT_DTYPE)
    return state.raqic_patch_intention


def update_raqic_global_intention(state: WorldState, cfg: SimulationConfig) -> np.ndarray:
    assert state.raqic_global_intention is not None
    agg = aggregate_raqic_patches_to_global(state, cfg)
    eta = float(cfg.raqic.parent_intention_eta)
    out = normalize_intention(
        (1 - eta) * state.raqic_global_intention + eta * agg, len(Action)
    ).astype(DEFAULT_FLOAT_DTYPE)
    state.raqic_global_intention[...] = out
    return out
