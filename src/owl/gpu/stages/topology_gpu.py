from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from owl.core.actions import Action

TOPOLOGY_MERGE = 1
TOPOLOGY_SPLIT = 2
TOPOLOGY_EXPEL = 3
_CARDINAL_NEIGHBORS = ((-1, 0), (1, 0), (0, 1), (0, -1))
_SPLIT_NEIGHBORS = ((-1, 0), (1, 0), (0, 1), (0, -1), (-1, 1), (-1, -1), (1, 1), (1, -1))


@dataclass
class TopologyEventBuffer:
    """Fixed-capacity dense topology event buffer.

    The buffer is device-array backed when ``ds`` uses CuPy and NumPy-backed in
    CPU fallback tests.  It represents bounded merge/split/expel events without
    Python object lists in the execution path.
    """

    capacity: int
    event_type: Any
    source_y: Any
    source_x: Any
    target_y: Any
    target_x: Any
    priority: Any
    active: Any
    accepted: Any
    ttl: Any
    payload0: Any
    payload1: Any
    device_count: Any
    overflow_flag: Any
    count: int = 0
    overflow: int = 0

    def as_dict(self, backend: Any, *, include_event_types: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {
            "capacity": int(self.capacity),
            "count": int(self.count),
            "overflow": int(self.overflow),
        }
        if include_event_types and self.count:
            out["event_types"] = backend.asnumpy(self.event_type[: self.count]).astype(int).tolist()
        else:
            out["event_types"] = []
        return out


def _boundary_coords(xp: Any, y: Any, x: Any, h: int, w: int, mode: str) -> Any:
    if mode == "toroidal":
        return (y % h, x % w, xp.ones_like(y, dtype=bool))
    ok = (y >= 0) & (y < h) & (x >= 0) & (x < w)
    return (xp.clip(y, 0, h - 1), xp.clip(x, 0, w - 1), ok)


def _best_merge_targets(ds: Any, cfg: Any, source_mask: Any) -> Any:
    """Choose highest-integration cardinal live neighbor for each merge source."""
    xp = ds.xp
    h, w = ds.health.shape
    sy, sx = xp.nonzero(source_mask)
    n = int(sy.shape[0])
    ty = sy.copy()
    tx = sx.copy()
    best = xp.full((n,), -1e30, dtype=ds.integration.dtype)
    ok = xp.zeros((n,), dtype=bool)
    mode = str(cfg.world.boundary_mode)
    live = (ds.health > 0) & ~ds.obstacle
    for dy, dx in _CARDINAL_NEIGHBORS:
        cy, cx, inside = _boundary_coords(xp, sy + dy, sx + dx, h, w, mode)
        cand = inside & live[cy, cx]
        score = xp.where(cand, ds.integration[cy, cx], -1e30)
        take = score > best
        ty = xp.where(take, cy, ty)
        tx = xp.where(take, cx, tx)
        best = xp.where(take, score, best)
        ok = ok | cand
    return (sy, sx, ty, tx, ok, best)


def _first_empty_targets(ds: Any, cfg: Any, source_mask: Any) -> Any:
    """Choose first legal empty neighbor for split child placement."""
    xp = ds.xp
    h, w = ds.health.shape
    sy, sx = xp.nonzero(source_mask)
    n = int(sy.shape[0])
    ty = sy.copy()
    tx = sx.copy()
    ok = xp.zeros((n,), dtype=bool)
    mode = str(cfg.world.boundary_mode)
    empty = (ds.health <= 0) & ~ds.obstacle
    for dy, dx in _SPLIT_NEIGHBORS:
        cy, cx, inside = _boundary_coords(xp, sy + dy, sx + dx, h, w, mode)
        cand = inside & empty[cy, cx] & ~ok
        ty = xp.where(cand, cy, ty)
        tx = xp.where(cand, cx, tx)
        ok = ok | cand
    return (sy, sx, ty, tx, ok)


def _make_buffer(
    ds: Any, cfg: Any, event_type: Any, sy: Any, sx: Any, ty: Any, tx: Any, priority: Any
) -> TopologyEventBuffer:
    xp = ds.xp
    capacity = int(getattr(cfg.raqic, "full_gpu_sparse_event_capacity", 4096))
    total = int(event_type.shape[0])
    overflow = max(0, total - capacity)
    if (
        overflow
        and bool(getattr(cfg.raqic, "full_gpu_no_silent_fallback", True))
        and bool(getattr(cfg.raqic, "full_gpu_strict", True))
    ):
        raise RuntimeError(
            f"GPU topology event capacity exceeded: {total} events > capacity {capacity}"
        )
    keep = min(total, capacity)
    et = xp.zeros((capacity,), dtype=xp.int32)
    sy_out = xp.full((capacity,), -1, dtype=xp.int32)
    sx_out = xp.full((capacity,), -1, dtype=xp.int32)
    ty_out = xp.full((capacity,), -1, dtype=xp.int32)
    tx_out = xp.full((capacity,), -1, dtype=xp.int32)
    pr_out = xp.zeros((capacity,), dtype=xp.float64)
    active = xp.zeros((capacity,), dtype=bool)
    accepted = xp.zeros((capacity,), dtype=bool)
    ttl = xp.zeros((capacity,), dtype=xp.int16)
    payload0 = xp.zeros((capacity,), dtype=xp.float64)
    payload1 = xp.zeros((capacity,), dtype=xp.float64)
    if keep:
        et[:keep] = event_type[:keep].astype(xp.int32)
        sy_out[:keep] = sy[:keep].astype(xp.int32)
        sx_out[:keep] = sx[:keep].astype(xp.int32)
        ty_out[:keep] = ty[:keep].astype(xp.int32)
        tx_out[:keep] = tx[:keep].astype(xp.int32)
        pr_out[:keep] = priority[:keep].astype(xp.float64)
        active[:keep] = True
        accepted[:keep] = True
        ttl[:keep] = 3
    device_count = xp.asarray([keep], dtype=xp.int32)
    overflow_flag = xp.asarray([overflow], dtype=xp.int32)
    return TopologyEventBuffer(
        capacity,
        et,
        sy_out,
        sx_out,
        ty_out,
        tx_out,
        pr_out,
        active,
        accepted,
        ttl,
        payload0,
        payload1,
        device_count,
        overflow_flag,
        keep,
        overflow,
    )


def _deduplicate_targets(
    ds: Any, event_type: Any, sy: Any, sx: Any, ty: Any, tx: Any, priority: Any
) -> Any:
    """Accept one event per target using deterministic priority/source ordering."""
    xp = ds.xp
    n = int(sy.shape[0])
    if n <= 1:
        return (event_type, sy, sx, ty, tx, priority)
    h, w = ds.health.shape
    source_flat = sy.astype(xp.int64) * int(w) + sx.astype(xp.int64)
    target_flat = ty.astype(xp.int64) * int(w) + tx.astype(xp.int64)
    rank = xp.rint(priority * 1000000).astype(xp.int64)
    max_rank = xp.int64(1000000)
    type_rank = event_type.astype(xp.int64)
    key = (
        type_rank * xp.int64((h * w + 1) * (max_rank + 1) * (h * w + 1))
        + target_flat * xp.int64((max_rank + 1) * (h * w + 1))
        + (max_rank - rank) * xp.int64(h * w + 1)
        + source_flat
    )
    order = xp.argsort(key)
    et, sy, sx, ty, tx, priority = (
        event_type[order],
        sy[order],
        sx[order],
        ty[order],
        tx[order],
        priority[order],
    )
    target_sorted = target_flat[order] + et.astype(xp.int64) * xp.int64(h * w + 1)
    first = xp.ones_like(target_sorted, dtype=bool)
    if n > 1:
        first[1:] = target_sorted[1:] != target_sorted[:-1]
    return (et[first], sy[first], sx[first], ty[first], tx[first], priority[first])


def detect_topology_events_gpu(ds: Any, cfg: Any) -> TopologyEventBuffer:
    """Detect dense merge/split/expel topology events on device arrays.

    MERGE chooses the highest-integration live cardinal neighbor.  SPLIT creates
    a child in the first legal empty neighbor.  EXPEL clears a dense parent link.
    All events are represented in a bounded device buffer; strict mode raises on
    capacity overflow.
    """
    xp = ds.xp
    live = (ds.health > 0) & ~ds.obstacle
    readout = ds.readout.astype(xp.int32)
    merge_mask = live & (readout == int(Action.MERGE))
    msy, msx, mty, mtx, mok, mscore = _best_merge_targets(ds, cfg, merge_mask)
    msy, msx, mty, mtx, mscore = (msy[mok], msx[mok], mty[mok], mtx[mok], mscore[mok])
    met = xp.full((int(msy.shape[0]),), TOPOLOGY_MERGE, dtype=xp.int32)
    split_mask = live & (readout == int(Action.SPLIT)) & (ds.resource > 0.05)
    ssy, ssx, sty, stx, sok = _first_empty_targets(ds, cfg, split_mask)
    ssy, ssx, sty, stx = (ssy[sok], ssx[sok], sty[sok], stx[sok])
    setype = xp.full((int(ssy.shape[0]),), TOPOLOGY_SPLIT, dtype=xp.int32)
    spriority = (
        ds.resource[ssy, ssx] if int(ssy.shape[0]) else xp.zeros((0,), dtype=ds.resource.dtype)
    )
    expel_mask = live & (readout == int(Action.EXPEL))
    esy, esx = xp.nonzero(expel_mask)
    ecount = int(esy.shape[0])
    etype = xp.full((ecount,), TOPOLOGY_EXPEL, dtype=xp.int32)
    ety = esy.copy()
    etx = esx.copy()
    epriority = ds.integration[esy, esx] if ecount else xp.zeros((0,), dtype=ds.integration.dtype)
    event_type = xp.concatenate([met, setype, etype])
    sy = xp.concatenate([msy, ssy, esy]).astype(xp.int32)
    sx = xp.concatenate([msx, ssx, esx]).astype(xp.int32)
    ty = xp.concatenate([mty, sty, ety]).astype(xp.int32)
    tx = xp.concatenate([mtx, stx, etx]).astype(xp.int32)
    priority = xp.concatenate([mscore, spriority, epriority]).astype(xp.float64)
    event_type, sy, sx, ty, tx, priority = _deduplicate_targets(
        ds, event_type, sy, sx, ty, tx, priority
    )
    return _make_buffer(ds, cfg, event_type, sy, sx, ty, tx, priority)


def _apply_merge(ds: Any, events: TopologyEventBuffer, idx: int) -> Any:
    xp = ds.xp
    sy, sx = (events.source_y[idx], events.source_x[idx])
    ty, tx = (events.target_y[idx], events.target_x[idx])
    if int(sy.shape[0]) == 0:
        return 0
    avg_health = 0.5 * (ds.health[sy, sx] + ds.health[ty, tx])
    avg_boundary = 0.5 * (ds.boundary[sy, sx] + ds.boundary[ty, tx])
    avg_integration = 0.5 * (ds.integration[sy, sx] + ds.integration[ty, tx])
    pooled = xp.minimum(ds.resource[sy, sx] + ds.resource[ty, tx], 1.0)
    ds.arrays["health"][sy, sx] = avg_health
    ds.arrays["health"][ty, tx] = avg_health
    ds.arrays["boundary"][sy, sx] = avg_boundary
    ds.arrays["boundary"][ty, tx] = avg_boundary
    ds.arrays["integration"][sy, sx] = avg_integration
    ds.arrays["integration"][ty, tx] = avg_integration
    ds.arrays["resource"][sy, sx] = 0.5 * pooled
    ds.arrays["resource"][ty, tx] = 0.5 * pooled
    if "parent_id" in ds.arrays:
        parent = xp.minimum(
            xp.where(ds.occupancy[sy, sx] >= 0, ds.occupancy[sy, sx], sy * ds.health.shape[1] + sx),
            xp.where(ds.occupancy[ty, tx] >= 0, ds.occupancy[ty, tx], ty * ds.health.shape[1] + tx),
        ).astype(ds.parent_id.dtype)
        ds.arrays["parent_id"][sy, sx] = parent
        ds.arrays["parent_id"][ty, tx] = parent
    return int(sy.shape[0])


def _apply_split(ds: Any, events: TopologyEventBuffer, idx: int) -> Any:
    xp = ds.xp
    sy, sx = (events.source_y[idx], events.source_x[idx])
    ty, tx = (events.target_y[idx], events.target_x[idx])
    n = int(sy.shape[0])
    if n == 0:
        return 0
    from owl.gpu.kernels.scatter_kernels import FieldSlabManager

    # One registry-generated transform is authoritative for every execution
    # tier. The manager uses persistent slabs when attached and a typed
    # reference pack/unpack path otherwise; no Python per-field mutation loop
    # remains in the topology tick.
    FieldSlabManager(ds).copy_all_registered(sy, sx, ty, tx)
    child_resource = 0.5 * ds.resource[sy, sx]
    ds.arrays["resource"][ty, tx] = child_resource
    ds.arrays["resource"][sy, sx] = child_resource
    ds.arrays["health"][ty, tx] = xp.maximum(0.25, 0.75 * ds.health[sy, sx])
    ds.arrays["boundary"][ty, tx] = xp.maximum(0.25, 0.75 * ds.boundary[sy, sx])
    ds.arrays["age"][ty, tx] = 0
    if "occupancy" in ds.arrays:
        start = int(ds.scalars.get("next_ow_id", 1))
        ids = xp.arange(start, start + n, dtype=ds.occupancy.dtype)
        ds.arrays["occupancy"][ty, tx] = ids
        ds.scalars["next_ow_id"] = start + n
    if "parent_id" in ds.arrays:
        parent = xp.where(
            ds.occupancy[sy, sx] >= 0, ds.occupancy[sy, sx], sy * ds.health.shape[1] + sx
        ).astype(ds.parent_id.dtype)
        ds.arrays["parent_id"][sy, sx] = parent
        ds.arrays["parent_id"][ty, tx] = parent
    return n


def _apply_expel(ds: Any, events: TopologyEventBuffer, idx: int) -> Any:
    sy, sx = (events.source_y[idx], events.source_x[idx])
    n = int(sy.shape[0])
    if n == 0:
        return 0
    if "parent_id" in ds.arrays:
        ds.arrays["parent_id"][sy, sx] = 0
    return n


def apply_topology_events_gpu(
    ds: Any, cfg: Any, events: TopologyEventBuffer | None = None
) -> dict[str, Any]:
    """Apply bounded dense topology events on device arrays.

    The function implements deterministic merge, split, and expel semantics.
    MERGE creates or updates a shared parent complex over adjacent cells. SPLIT
    places a child copy in a legal empty neighbor with shared resources. EXPEL
    clears a dense parent link. Sparse ``mobile_ows`` records remain CPU-only,
    while strict ``gpu_full`` execution uses the dense device representation.
    """
    if events is None:
        events = detect_topology_events_gpu(ds, cfg)
    defer = bool(ds.metadata.get("defer_host_metrics", False))
    ds.scalars["topology_overflow"] = int(events.overflow)
    if events.count == 0:
        return events.as_dict(ds.backend, include_event_types=not defer) | {
            "merged": 0,
            "split": 0,
            "expelled": 0,
        }
    et = events.event_type
    active = events.active & events.accepted
    merged = _apply_merge(ds, events, active & (et == TOPOLOGY_MERGE))
    split = _apply_split(ds, events, active & (et == TOPOLOGY_SPLIT))
    expelled = _apply_expel(ds, events, active & (et == TOPOLOGY_EXPEL))
    return events.as_dict(ds.backend, include_event_types=not defer) | {
        "merged": merged,
        "split": split,
        "expelled": expelled,
    }
