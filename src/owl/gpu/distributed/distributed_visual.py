from __future__ import annotations

from typing import Any

import numpy as np

from owl.gpu.transfer_ledger import TransferLedger
from owl.viz.event_bus import VisualEvent, VisualEventBuffer, VisualEventType
from owl.viz.frame_model import VisualFrame
from owl.viz.gpu_compositor import compose_frame_device


def gather_rgba_frame(
    ds: Any, shard: Any, shards: Any, transport: Any, stream: Any, *, tick: int
) -> Any:
    """Gather padded rank-local RGBA interiors to rank zero."""
    xp = ds.xp
    local = compose_frame_device(ds)[shard.interior_rows, :, :]
    max_height = max(item.owned_height for item in shards)
    padded = xp.zeros((max_height, shard.world_width, 4), dtype=local.dtype)
    padded[: shard.owned_height, :, :] = local
    gathered = xp.empty(
        (len(shards), max_height, shard.world_width, 4),
        dtype=padded.dtype,
    )
    transport.all_gather(padded, gathered, stream=stream, tick=tick)
    if shard.rank != 0:
        return None
    host = ds.backend.asnumpy(gathered)
    transfer_ledger = ds.metadata.get("transfer_ledger")
    if isinstance(transfer_ledger, TransferLedger):
        transfer_ledger.record_d2h(
            int(gathered.nbytes),
            kind="visual",
            tick=int(tick),
            source_stream="distributed-visual",
            synchronization="stream",
            scheduled=True,
            graph_compatible=False,
            reason="rank-zero distributed RGBA gather at render cadence",
        )
    global_frame = np.zeros((shard.world_height, shard.world_width, 4), dtype=host.dtype)
    for item in shards:
        global_frame[item.owned_rows, :, :] = host[item.rank, : item.owned_height, :, :]
    return global_frame


def gather_visual_events(
    controller: Any,
    ds: Any,
    shard: Any,
    shards: Any,
    transport: Any,
    stream: Any,
    *,
    tick: int,
    per_rank_capacity: int,
) -> Any:
    """Gather fixed-capacity event rows and convert coordinates to global rows."""
    xp = ds.xp
    local = controller.collect_events(ds)
    rows = local.to_numpy()
    cap = int(per_rank_capacity)
    padded_host = np.zeros((cap, 14), dtype=np.float32)
    keep = min(cap, rows.shape[0])
    if keep:
        padded_host[:keep] = rows[:keep]
        # Local coordinates include the north halo.
        padded_host[:keep, 2] += float(shard.owned_start - shard.halo_width)
        valid_target = padded_host[:keep, 4] >= 0
        padded_host[:keep, 4][valid_target] += float(shard.owned_start - shard.halo_width)
        padded_host[:keep, 2] %= float(shard.world_height)
        padded_host[:keep, 4][valid_target] %= float(shard.world_height)
    padded = xp.asarray(padded_host)
    transfer_ledger = ds.metadata.get("transfer_ledger")
    if isinstance(transfer_ledger, TransferLedger):
        transfer_ledger.record_h2d(
            int(padded_host.nbytes),
            kind="visual",
            tick=int(tick),
            source_stream="distributed-visual",
            synchronization="stream",
            scheduled=True,
            graph_compatible=False,
            reason="bounded sparse visual-event payload at render cadence",
        )
    gathered = xp.empty((len(shards), cap, 14), dtype=xp.float32)
    transport.all_gather(padded, gathered, stream=stream, tick=tick)
    if shard.rank != 0:
        return None
    host = ds.backend.asnumpy(gathered).reshape(-1, 14)
    if isinstance(transfer_ledger, TransferLedger):
        transfer_ledger.record_d2h(
            int(gathered.nbytes),
            kind="visual",
            tick=int(tick),
            source_stream="distributed-visual",
            synchronization="stream",
            scheduled=True,
            graph_compatible=False,
            reason="rank-zero distributed sparse visual-event gather",
        )
    out = VisualEventBuffer(capacity=controller.event_bus.capacity)
    for row in host:
        event_type = int(row[1])
        ttl = int(row[8])
        if event_type <= 0 or ttl <= 0:
            continue
        out.add(
            VisualEvent(
                tick=int(row[0]),
                event_type=VisualEventType(event_type),
                y=int(row[2]),
                x=int(row[3]),
                target_y=int(row[4]),
                target_x=int(row[5]),
                action=int(row[6]),
                intensity=float(row[7]),
                ttl=ttl,
                source_id=int(row[9]),
                channel=int(row[10]),
                payload0=float(row[11]),
                payload1=float(row[12]),
                priority=int(row[13]),
            ),
            replace_lower_priority=True,
        )
    out.sort_for_render()
    return out


def make_global_visual_frame(controller: Any, rgba: Any, events: Any, tick: int) -> VisualFrame:
    lod = "glyphs"
    markers, colors, sizes, lines, line_colors, arrows, selected = controller._frame_layers(
        events,
        rgba.shape[:2],
        lod,
    )
    return VisualFrame(
        rgba=rgba,
        markers=markers,
        marker_colors=colors,
        marker_sizes=sizes,
        lines=lines,
        line_colors=line_colors,
        arrows=arrows,
        events=selected,
        metadata={"tick": int(tick), "distributed": True, "lod": lod},
    )
