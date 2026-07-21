from __future__ import annotations

import copy
from collections.abc import Iterable
from dataclasses import fields
from typing import Any

import numpy as np

from owl.core.advanced import ensure_advanced_fields
from owl.core.init import initialize_world
from owl.core.state import WorldState
from owl.raqic.state import ensure_raqic_fields

from .partition import SpatialShard

_EXCLUDED = {"patches", "global_state", "event_queue", "mobile_ows", "tick"}


def _cell_array_names(state: WorldState, shape: tuple[int, int]) -> tuple[str, ...]:
    names = []
    for spec in fields(WorldState):
        if spec.name in _EXCLUDED:
            continue
        value = getattr(state, spec.name)
        if isinstance(value, np.ndarray) and value.shape[:2] == shape:
            names.append(spec.name)
    return tuple(names)


def create_local_state(
    global_state: WorldState,
    cfg: Any,
    shard: SpatialShard,
) -> Any:
    """Create a halo-padded local CPU state from a deterministic global state."""
    global_shape = global_state.health.shape
    if global_shape != (shard.world_height, shard.world_width):
        raise ValueError(
            f"global state shape {global_shape} does not match shard world "
            f"{(shard.world_height, shard.world_width)}"
        )
    local_cfg = cfg.model_copy(deep=True)
    local_cfg.world.height = int(shard.local_height)
    local_cfg.world.width = int(shard.world_width)
    # halo_width defaults to patch_size, so divisibility is retained.
    if local_cfg.world.height % int(local_cfg.world.patch_size):
        raise ValueError(
            "local halo-padded height must be divisible by patch_size; choose "
            "halo_width as a multiple of patch_size"
        )
    rng = np.random.default_rng(int(cfg.world.seed))
    local = initialize_world(local_cfg, rng)
    ensure_advanced_fields(local, local_cfg)
    if getattr(local_cfg.raqic, "enabled", False):
        ensure_raqic_fields(local, local_cfg)

    rows = np.asarray(shard.global_row_indices(), dtype=np.int64)
    for name in _cell_array_names(global_state, global_shape):
        source = getattr(global_state, name)
        target = getattr(local, name, None)
        sliced = np.asarray(source)[rows, ...]
        if isinstance(target, np.ndarray) and target.shape == sliced.shape:
            target[...] = sliced
        else:
            setattr(local, name, np.array(sliced, copy=True))
    local.tick = int(global_state.tick)
    local.next_ow_id = int(global_state.next_ow_id)
    local.event_queue = []
    local.mobile_ows = {}
    return local, local_cfg


def merge_local_states(
    base_state: WorldState,
    local_states: Iterable[tuple[SpatialShard, WorldState]],
    cfg: Any,
) -> WorldState:
    """Merge owned interiors from rank checkpoints into one global state."""
    out = copy.deepcopy(base_state)
    global_shape = out.health.shape
    rank_states = list(local_states)
    if not rank_states:
        raise ValueError("at least one local state is required")
    names = _cell_array_names(out, global_shape)
    ticks = set()
    for shard, local in rank_states:
        ticks.add(int(local.tick))
        interior = shard.interior_rows
        owned = shard.owned_rows
        for name in names:
            source = getattr(local, name, None)
            target = getattr(out, name, None)
            if not isinstance(source, np.ndarray) or not isinstance(target, np.ndarray):
                continue
            if source.shape[:2] != (shard.local_height, shard.world_width):
                continue
            target[owned, ...] = source[interior, ...]
        out.next_ow_id = max(int(out.next_ow_id), int(local.next_ow_id))
    if len(ticks) != 1:
        raise ValueError(f"rank checkpoints disagree on tick: {sorted(ticks)}")
    out.tick = ticks.pop()

    # Recompute global summaries from the merged dense state.
    from owl.engine.loop import _post_state_refresh

    _post_state_refresh(out, cfg)
    return out
