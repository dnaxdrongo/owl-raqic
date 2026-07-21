from __future__ import annotations

import numpy as np

from owl.core.actions import Action
from owl.core.config import SimulationConfig
from owl.core.constants import DEFAULT_FLOAT_DTYPE
from owl.core.state import WorldState, field_shape
from owl.engine.aggregation import upsample_patch_bias


def dispatch_raqic_intention_to_cells(state: WorldState, cfg: SimulationConfig) -> np.ndarray:
    assert state.raqic_patch_intention is not None
    assert state.raqic_global_intention is not None
    assert state.raqic_parent_intention is not None
    h, w = field_shape(state)
    actions = len(Action)
    cell = upsample_patch_bias(
        np.asarray(state.raqic_patch_intention, dtype=np.float32), cfg.world.patch_size
    )
    if cell.shape != (h, w, actions):
        raise ValueError(
            f"upsampled RAQIC intention has shape {cell.shape}, expected {(h, w, actions)}"
        )
    glob = np.asarray(state.raqic_global_intention, dtype=np.float32)
    mixed = 0.75 * cell + 0.25 * glob[None, None, :]
    sums = np.sum(mixed, axis=-1, keepdims=True)
    mixed = np.divide(mixed, sums, out=np.zeros_like(mixed), where=sums > cfg.actions.epsilon)
    bad = sums[..., 0] <= cfg.actions.epsilon
    if np.any(bad):
        mixed[bad, :] = 0
        mixed[bad, int(Action.REST)] = 1
    dead = (state.health <= 0.0) | state.obstacle
    if np.any(dead):
        mixed[dead, :] = 0
        mixed[dead, int(Action.REST)] = 1
    state.raqic_parent_intention[...] = mixed.astype(DEFAULT_FLOAT_DTYPE)
    return state.raqic_parent_intention
