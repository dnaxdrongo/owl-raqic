from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from owl.core.actions import Action
from owl.gpu.array_write import write_array
from owl.gpu.stages.movement_gpu import _action_deltas
from owl.gpu.stages.reproduction_gpu import NEIGHBORS
from owl.gpu.stages.topology_gpu import (
    TOPOLOGY_EXPEL,
    TOPOLOGY_MERGE,
    TOPOLOGY_SPLIT,
    TopologyEventBuffer,
)


@dataclass
class FixedConflictBuffers:
    source: Any
    target: Any
    active: Any
    priority: Any
    key: Any
    order: Any
    sorted_target: Any
    first: Any
    accepted: Any
    winner_source: Any


@dataclass
class GraphStaticActionBuffers:
    movement: FixedConflictBuffers
    reproduction: FixedConflictBuffers
    merge: FixedConflictBuffers
    split: FixedConflictBuffers
    prefix: Any
    arange: Any
    next_ow_id: Any
    topology: TopologyEventBuffer


def _empty_conflict(xp: Any, n: int) -> FixedConflictBuffers:
    return FixedConflictBuffers(
        source=xp.arange(n, dtype=xp.int64),
        target=xp.full((n,), -1, dtype=xp.int64),
        active=xp.zeros((n,), dtype=bool),
        priority=xp.zeros((n,), dtype=xp.int64),
        key=xp.empty((n,), dtype=xp.int64),
        order=xp.empty((n,), dtype=xp.int64),
        sorted_target=xp.empty((n,), dtype=xp.int64),
        first=xp.zeros((n,), dtype=bool),
        accepted=xp.zeros((n,), dtype=bool),
        winner_source=xp.full((n,), -1, dtype=xp.int64),
    )


def _topology_buffer(xp: Any, n: int) -> TopologyEventBuffer:
    capacity = 3 * n
    return TopologyEventBuffer(
        capacity=capacity,
        event_type=xp.zeros((capacity,), dtype=xp.int32),
        source_y=xp.zeros((capacity,), dtype=xp.int32),
        source_x=xp.zeros((capacity,), dtype=xp.int32),
        target_y=xp.full((capacity,), -1, dtype=xp.int32),
        target_x=xp.full((capacity,), -1, dtype=xp.int32),
        priority=xp.zeros((capacity,), dtype=xp.float64),
        active=xp.zeros((capacity,), dtype=bool),
        accepted=xp.zeros((capacity,), dtype=bool),
        ttl=xp.zeros((capacity,), dtype=xp.int16),
        payload0=xp.zeros((capacity,), dtype=xp.float64),
        payload1=xp.zeros((capacity,), dtype=xp.float64),
        device_count=xp.zeros((1,), dtype=xp.int32),
        overflow_flag=xp.zeros((1,), dtype=xp.int32),
        count=capacity,
        overflow=0,
    )


def ensure_graph_static_action_buffers(ds: Any, cfg: Any) -> GraphStaticActionBuffers:
    """Allocate graph-stable candidate, prefix, ID, and event buffers once."""
    n = int(np.prod(ds.health.shape))
    existing = ds.metadata.get("graph_static_action_buffers")
    if existing is not None and int(existing.arange.size) == n:
        return cast(GraphStaticActionBuffers, existing)
    required_capacity = 3 * n
    configured = int(getattr(cfg.raqic, "full_gpu_sparse_event_capacity", 4096))
    if configured < required_capacity:
        raise MemoryError(
            "full-tick graph topology requires "
            "full_gpu_sparse_event_capacity >= 3*H*W "
            f"({required_capacity}), got {configured}"
        )
    xp = ds.xp
    next_id = xp.asarray([int(ds.scalars.get("next_ow_id", 1))], dtype=xp.int64)
    buffers = GraphStaticActionBuffers(
        movement=_empty_conflict(xp, n),
        reproduction=_empty_conflict(xp, n),
        merge=_empty_conflict(xp, n),
        split=_empty_conflict(xp, n),
        prefix=xp.zeros((n,), dtype=xp.int64),
        arange=xp.arange(n, dtype=xp.int64),
        next_ow_id=next_id,
        topology=_topology_buffer(xp, n),
    )
    ds.metadata["graph_static_action_buffers"] = buffers
    write_array(ds, "_next_ow_id_device", next_id)
    ds.arrays.setdefault("_graph_moved_count", xp.zeros((1,), dtype=xp.int64))
    ds.arrays.setdefault("_graph_children_count", xp.zeros((1,), dtype=xp.int64))
    ds.arrays.setdefault("_graph_topology_count", xp.zeros((1,), dtype=xp.int64))
    return buffers


def _resolve_fixed(buffers: FixedConflictBuffers, *, n: int, max_priority: int, xp: Any) -> None:
    """Deterministic one-winner-per-target resolution without compaction."""
    sentinel = np.iinfo(np.int64).max
    stride = np.int64(n + 1)
    target_safe = xp.where(buffers.active, buffers.target, 0).astype(xp.int64)
    priority = xp.clip(buffers.priority, 0, max_priority).astype(xp.int64)
    buffers.key[...] = xp.where(
        buffers.active,
        target_safe * np.int64((max_priority + 1) * (n + 1))
        + (np.int64(max_priority) - priority) * stride
        + buffers.source,
        np.int64(sentinel),
    )
    buffers.order[...] = xp.argsort(buffers.key)
    sorted_source = buffers.source[buffers.order]
    buffers.sorted_target[...] = buffers.target[buffers.order]
    sorted_active = buffers.active[buffers.order]
    buffers.first.fill(False)
    if n:
        buffers.first[0] = sorted_active[0]
    if n > 1:
        buffers.first[1:] = sorted_active[1:] & (
            ~sorted_active[:-1] | (buffers.sorted_target[1:] != buffers.sorted_target[:-1])
        )
    buffers.accepted.fill(False)
    buffers.accepted[buffers.order] = buffers.first
    buffers.winner_source.fill(-1)
    safe_target = xp.where(buffers.first, buffers.sorted_target, 0)
    source_value = xp.where(buffers.first, sorted_source, -1)
    xp.maximum.at(buffers.winner_source, safe_target, source_value)


def _copy_slab_by_winner(ds: Any, winner_source: Any, *, vacated: Any | None = None) -> None:
    manager = ds.metadata.get("slab_manager")
    if manager is None:
        raise RuntimeError("full-tick graph action execution requires full_gpu_fuse_scatter=true")
    xp = ds.xp
    n = int(winner_source.size)
    destination = xp.arange(n, dtype=xp.int64)
    source_index = xp.where(winner_source >= 0, winner_source, destination)
    winner_mask = winner_source >= 0
    for slab in manager.slabs.values():
        trailing = int(np.prod(slab.shape[3:])) if slab.ndim > 3 else 1
        view = slab.reshape(slab.shape[0], n, trailing)
        gathered = view[:, source_index, :]
        out = xp.where(winner_mask[None, :, None], gathered, view)
        if vacated is not None:
            out = xp.where(vacated[None, :, None], xp.zeros_like(out), out)
        view[...] = out


def _movement_candidates(ds: Any, cfg: Any, buffers: FixedConflictBuffers) -> None:
    xp = ds.xp
    h, w = map(int, ds.health.shape)
    n = h * w
    actions = int(ds.possibility.shape[-1])
    dy_tab, dx_tab = _action_deltas(xp, actions)
    readout = ds.readout.reshape(-1).astype(xp.int32)
    dy = dy_tab[readout]
    dx = dx_tab[readout]
    y = buffers.source // w
    x = buffers.source % w
    ty = y + dy
    tx = x + dx
    if str(cfg.world.boundary_mode) == "toroidal":
        ty %= h
        tx %= w
        in_bounds = xp.ones((n,), dtype=bool)
    else:
        in_bounds = (ty >= 0) & (ty < h) & (tx >= 0) & (tx < w)
        ty = xp.clip(ty, 0, h - 1)
        tx = xp.clip(tx, 0, w - 1)
    move_table = xp.zeros((actions,), dtype=bool)
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
        if int(action) < actions:
            move_table[int(action)] = True
    health = ds.health.reshape(-1)
    obstacle = ds.obstacle.reshape(-1)
    target = (ty * w + tx).astype(xp.int64)
    active = (
        (health > 0)
        & ~obstacle
        & move_table[readout]
        & in_bounds
        & ~obstacle[target]
        & (health[target] <= 0)
    )
    probability = ds.possibility.reshape(n, actions)[buffers.source, readout]
    buffers.target[...] = xp.where(active, target, -1)
    buffers.active[...] = active
    buffers.priority[...] = xp.rint(probability * 1000000).astype(xp.int64)


def _legacy_apply_movement_graph_static(ds: Any, cfg: Any) -> dict[str, Any]:
    buffers = ensure_graph_static_action_buffers(ds, cfg).movement
    xp = ds.xp
    n = int(buffers.source.size)
    _movement_candidates(ds, cfg, buffers)
    _resolve_fixed(buffers, n=n, max_priority=1000000, xp=xp)
    _copy_slab_by_winner(ds, buffers.winner_source, vacated=buffers.accepted)
    if "occupancy" in ds.arrays:
        old = ds.occupancy.reshape(-1)
        source_index = xp.where(buffers.winner_source >= 0, buffers.winner_source, buffers.source)
        out = xp.where(buffers.winner_source >= 0, old[source_index], old)
        out = xp.where(buffers.accepted, -1, out)
        old[...] = out.astype(old.dtype)
    ds.arrays["last_movement_action"][...] = ds.readout.astype(xp.int32)
    moved = xp.sum(buffers.accepted, dtype=xp.int64)
    ds.arrays["_graph_moved_count"][0] = moved
    return {"moved_device": ds.arrays["_graph_moved_count"]}


def _first_empty_target(ds: Any, cfg: Any, active: bool, source: Any) -> Any:
    xp = ds.xp
    h, w = map(int, ds.health.shape)
    y = source // w
    x = source % w
    target = source.copy()
    found = xp.zeros_like(active)
    health = ds.health.reshape(-1)
    obstacle = ds.obstacle.reshape(-1)
    for dy, dx in NEIGHBORS:
        cy = y + int(dy)
        cx = x + int(dx)
        if str(cfg.world.boundary_mode) == "toroidal":
            cy %= h
            cx %= w
            in_bounds = xp.ones_like(active)
        else:
            in_bounds = (cy >= 0) & (cy < h) & (cx >= 0) & (cx < w)
            cy = xp.clip(cy, 0, h - 1)
            cx = xp.clip(cx, 0, w - 1)
        candidate = (cy * w + cx).astype(xp.int64)
        choose = active & ~found & in_bounds & (health[candidate] <= 0) & ~obstacle[candidate]
        target = xp.where(choose, candidate, target)
        found |= choose
    return (target, found)


def _copy_slab_to_children(ds: Any, winner_source: Any) -> None:
    _copy_slab_by_winner(ds, winner_source, vacated=None)


def _assign_child_ids(
    ds: Any, cfg: Any, buffers: FixedConflictBuffers, static: GraphStaticActionBuffers
) -> None:
    xp = ds.xp
    accepted_i64 = buffers.accepted.astype(xp.int64)
    xp.cumsum(accepted_i64, out=static.prefix)
    child_count = static.prefix[-1] if static.prefix.size else xp.asarray(0, dtype=xp.int64)
    winner = buffers.winner_source
    child_mask = winner >= 0
    parent_prefix = static.prefix[xp.where(child_mask, winner, 0)]
    ids = static.next_ow_id[0] + parent_prefix - 1
    if "occupancy" in ds.arrays:
        occupancy = ds.occupancy.reshape(-1)
        occupancy[...] = xp.where(child_mask, ids, occupancy).astype(occupancy.dtype)
    static.next_ow_id[...] = static.next_ow_id + child_count


def _legacy_apply_reproduction_graph_static(ds: Any, cfg: Any) -> dict[str, Any]:
    xp = ds.xp
    static = ensure_graph_static_action_buffers(ds, cfg)
    buffers = static.reproduction
    n = int(buffers.source.size)
    if not bool(cfg.reproduction.enabled):
        buffers.active.fill(False)
        buffers.accepted.fill(False)
        ds.arrays["_graph_children_count"].fill(0)
        return {"children_device": ds.arrays["_graph_children_count"]}
    viable = (
        (ds.readout.reshape(-1) == int(Action.REPRODUCE))
        & (ds.resource.reshape(-1) >= float(cfg.reproduction.min_resource))
        & (ds.health.reshape(-1) >= float(cfg.reproduction.min_health))
        & (ds.boundary.reshape(-1) >= float(cfg.reproduction.min_boundary))
        & (ds.integration.reshape(-1) >= float(cfg.reproduction.min_integration))
    )
    target, found = _first_empty_target(ds, cfg, viable, buffers.source)
    buffers.target[...] = xp.where(found, target, -1)
    buffers.active[...] = found
    buffers.priority[...] = np.int64(n) - buffers.source
    _resolve_fixed(buffers, n=n, max_priority=n, xp=xp)
    parent_resource = ds.resource.reshape(-1).copy()
    _copy_slab_to_children(ds, buffers.winner_source)
    child_mask = buffers.winner_source >= 0
    source_for_child = xp.where(child_mask, buffers.winner_source, 0)
    fraction = float(cfg.reproduction.child_resource_fraction)
    resource = ds.resource.reshape(-1)
    resource[...] = xp.where(buffers.accepted, parent_resource * (1.0 - fraction), resource)
    resource[...] = xp.where(child_mask, parent_resource[source_for_child] * fraction, resource)
    health = ds.health.reshape(-1)
    boundary = ds.boundary.reshape(-1)
    age = ds.age.reshape(-1)
    health[...] = xp.where(child_mask, float(cfg.reproduction.initial_child_health), health)
    boundary[...] = xp.where(child_mask, float(cfg.reproduction.initial_child_boundary), boundary)
    age[...] = xp.where(child_mask, 0, age)
    _assign_child_ids(ds, cfg, buffers, static)
    count = static.prefix[-1] if n else xp.asarray(0, dtype=xp.int64)
    ds.arrays["_graph_children_count"][0] = count
    return {"children_device": ds.arrays["_graph_children_count"]}


def _best_merge_target(ds: Any, cfg: Any, active: bool, source: Any) -> Any:
    xp = ds.xp
    h, w = map(int, ds.health.shape)
    y = source // w
    x = source % w
    best_target = source.copy()
    best_score = xp.full((source.size,), -xp.inf, dtype=xp.float64)
    found = xp.zeros_like(active)
    health = ds.health.reshape(-1)
    obstacle = ds.obstacle.reshape(-1)
    integration = ds.integration.reshape(-1).astype(xp.float64)
    for dy, dx in ((-1, 0), (1, 0), (0, 1), (0, -1)):
        cy = y + dy
        cx = x + dx
        if str(cfg.world.boundary_mode) == "toroidal":
            cy %= h
            cx %= w
            in_bounds = xp.ones_like(active)
        else:
            in_bounds = (cy >= 0) & (cy < h) & (cx >= 0) & (cx < w)
            cy = xp.clip(cy, 0, h - 1)
            cx = xp.clip(cx, 0, w - 1)
        candidate = (cy * w + cx).astype(xp.int64)
        legal = active & in_bounds & (health[candidate] > 0) & ~obstacle[candidate]
        score = xp.where(legal, integration[candidate], -xp.inf)
        better = score > best_score
        best_score = xp.where(better, score, best_score)
        best_target = xp.where(better, candidate, best_target)
        found |= legal
    return (best_target, found, best_score)


def _fill_topology_event_buffer(ds: Any, static: GraphStaticActionBuffers) -> None:
    xp = ds.xp
    n = int(static.arange.size)
    topo = static.topology
    h, w = map(int, ds.health.shape)
    y = (static.arange // w).astype(xp.int32)
    x = (static.arange % w).astype(xp.int32)
    topo.event_type[:n] = TOPOLOGY_MERGE
    topo.event_type[n : 2 * n] = TOPOLOGY_SPLIT
    topo.event_type[2 * n : 3 * n] = TOPOLOGY_EXPEL
    for offset, buffers in ((0, static.merge), (n, static.split)):
        sl = slice(offset, offset + n)
        topo.source_y[sl] = y
        topo.source_x[sl] = x
        safe_target = xp.where(buffers.target >= 0, buffers.target, 0)
        topo.target_y[sl] = (safe_target // w).astype(xp.int32)
        topo.target_x[sl] = (safe_target % w).astype(xp.int32)
        topo.priority[sl] = buffers.priority.astype(xp.float64)
        topo.active[sl] = buffers.active
        topo.accepted[sl] = buffers.accepted
        topo.ttl[sl] = xp.where(buffers.accepted, 3, 0).astype(xp.int16)
    expel = ds.readout.reshape(-1) == int(Action.EXPEL)
    sl = slice(2 * n, 3 * n)
    topo.source_y[sl] = y
    topo.source_x[sl] = x
    topo.target_y[sl] = y
    topo.target_x[sl] = x
    topo.priority[sl] = ds.integration.reshape(-1).astype(xp.float64)
    topo.active[sl] = expel
    topo.accepted[sl] = expel
    topo.ttl[sl] = xp.where(expel, 3, 0).astype(xp.int16)
    topo.device_count[...] = xp.sum(topo.accepted, dtype=xp.int32)
    topo.overflow_flag.fill(0)


def _legacy_apply_topology_graph_static(ds: Any, cfg: Any) -> dict[str, Any]:
    xp = ds.xp
    static = ensure_graph_static_action_buffers(ds, cfg)
    n = int(static.arange.size)
    merge_active = (
        (ds.readout.reshape(-1) == int(Action.MERGE))
        & (ds.health.reshape(-1) > 0)
        & ~ds.obstacle.reshape(-1)
    )
    target, found, score = _best_merge_target(ds, cfg, merge_active, static.merge.source)
    static.merge.target[...] = xp.where(found, target, -1)
    static.merge.active[...] = found
    static.merge.priority[...] = xp.rint(xp.clip(score, 0.0, 1.0) * 1000000).astype(xp.int64)
    _resolve_fixed(static.merge, n=n, max_priority=1000000, xp=xp)
    split_active = (
        (ds.readout.reshape(-1) == int(Action.SPLIT))
        & (ds.health.reshape(-1) > 0)
        & (ds.resource.reshape(-1) > 0.05)
        & ~ds.obstacle.reshape(-1)
    )
    split_target, split_found = _first_empty_target(ds, cfg, split_active, static.split.source)
    static.split.target[...] = xp.where(split_found, split_target, -1)
    static.split.active[...] = split_found
    static.split.priority[...] = xp.rint(
        xp.clip(ds.resource.reshape(-1), 0.0, 1.0) * 1000000
    ).astype(xp.int64)
    _resolve_fixed(static.split, n=n, max_priority=1000000, xp=xp)
    merge_source = static.merge.winner_source
    merge_target = static.merge.target
    source_active = static.merge.accepted
    target_active = merge_source >= 0
    safe_target = xp.where(source_active, merge_target, static.arange)
    safe_source = xp.where(target_active, merge_source, static.arange)
    old_health = ds.health.reshape(-1).copy()
    old_boundary = ds.boundary.reshape(-1).copy()
    old_integration = ds.integration.reshape(-1).copy()
    old_resource = ds.resource.reshape(-1).copy()
    for array, old in (
        (ds.health.reshape(-1), old_health),
        (ds.boundary.reshape(-1), old_boundary),
        (ds.integration.reshape(-1), old_integration),
    ):
        source_avg = 0.5 * (old + old[safe_target])
        target_avg = 0.5 * (old + old[safe_source])
        array[...] = xp.where(source_active, source_avg, old)
        array[...] = xp.where(target_active, target_avg, array)
    pooled_source = xp.minimum(old_resource + old_resource[safe_target], 1.0) * 0.5
    pooled_target = xp.minimum(old_resource + old_resource[safe_source], 1.0) * 0.5
    resource = ds.resource.reshape(-1)
    resource[...] = xp.where(source_active, pooled_source, old_resource)
    resource[...] = xp.where(target_active, pooled_target, resource)
    if "parent_id" in ds.arrays:
        occupancy = ds.occupancy.reshape(-1)
        parent = ds.parent_id.reshape(-1)
        source_parent = xp.minimum(
            xp.where(occupancy >= 0, occupancy, static.arange),
            xp.where(occupancy[safe_target] >= 0, occupancy[safe_target], safe_target),
        )
        target_parent = xp.minimum(
            xp.where(occupancy >= 0, occupancy, static.arange),
            xp.where(occupancy[safe_source] >= 0, occupancy[safe_source], safe_source),
        )
        parent[...] = xp.where(source_active, source_parent, parent)
        parent[...] = xp.where(target_active, target_parent, parent)
    split_parent_resource = ds.resource.reshape(-1).copy()
    _copy_slab_to_children(ds, static.split.winner_source)
    split_child = static.split.winner_source >= 0
    split_source = xp.where(split_child, static.split.winner_source, 0)
    resource = ds.resource.reshape(-1)
    resource[...] = xp.where(static.split.accepted, 0.5 * split_parent_resource, resource)
    resource[...] = xp.where(split_child, 0.5 * split_parent_resource[split_source], resource)
    ds.health.reshape(-1)[...] = xp.where(
        split_child,
        xp.maximum(0.25, 0.75 * ds.health.reshape(-1)[split_source]),
        ds.health.reshape(-1),
    )
    ds.boundary.reshape(-1)[...] = xp.where(
        split_child,
        xp.maximum(0.25, 0.75 * ds.boundary.reshape(-1)[split_source]),
        ds.boundary.reshape(-1),
    )
    ds.age.reshape(-1)[...] = xp.where(split_child, 0, ds.age.reshape(-1))
    _assign_child_ids(ds, cfg, static.split, static)
    if "parent_id" in ds.arrays:
        expel = ds.readout.reshape(-1) == int(Action.EXPEL)
        ds.parent_id.reshape(-1)[...] = xp.where(expel, 0, ds.parent_id.reshape(-1))
    _fill_topology_event_buffer(ds, static)
    ds.metadata["last_topology_events"] = static.topology
    return {
        "event_count_device": static.topology.device_count,
        "overflow_device": static.topology.overflow_flag,
    }


# Scientific reference adapters -------------------------------------------

# Graph-static actions delegate to the matching eager scientific transition.
# scientific transition. Until capture-safe workspace kernels are certified
# on target CUDA hardware, graph-selected actions reuse the exact eager
# implementations. This closes the second scientific law while the execution
# plan remains fail-closed for strict full-tick capture (see execution_plan.py).
def apply_movement_graph_static(ds: Any, cfg: Any) -> dict[str, Any]:
    from owl.gpu.stages.movement_gpu import apply_movement_gpu

    return apply_movement_gpu(ds, cfg)


def apply_reproduction_graph_static(ds: Any, cfg: Any) -> dict[str, Any]:
    from owl.gpu.stages.reproduction_gpu import apply_reproduction_gpu

    return apply_reproduction_gpu(ds, cfg)


def apply_topology_graph_static(ds: Any, cfg: Any) -> dict[str, Any]:
    from owl.gpu.stages.topology_gpu import apply_topology_events_gpu, detect_topology_events_gpu

    events = detect_topology_events_gpu(ds, cfg)
    result = apply_topology_events_gpu(ds, cfg, events)
    if not isinstance(result, dict):
        result = {"events": int(events.count)}
    result.setdefault("events", int(events.count))
    result["_cadc_events"] = events
    return result
