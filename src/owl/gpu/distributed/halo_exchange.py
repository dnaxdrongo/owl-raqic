from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .halo_protocol import generate_halo_protocol
from .partition import SpatialShard


@dataclass
class HaloExchangeStats:
    fields: int = 0
    elements_sent: int = 0
    elements_received: int = 0
    protocol: dict[str, Any] | None = None

    def to_dict(self) -> Any:
        return {
            "fields": self.fields,
            "elements_sent": self.elements_sent,
            "elements_received": self.elements_received,
            "protocol": self.protocol,
        }


def _contiguous_buffer(
    ds: Any, cache: dict[tuple[Any, ...], Any], key: tuple[Any, ...], shape: Any, dtype: Any
) -> Any:
    xp = ds.xp
    value = cache.get(key)
    if value is None or value.shape != shape or value.dtype != dtype:
        value = xp.empty(shape, dtype=dtype)
        cache[key] = value
    return value


def exchange_halos(
    ds: Any,
    shard: SpatialShard,
    transport: Any,
    stream: Any,
    *,
    field_names: Any | None = None,
    tick: int = -1,
    phase: str = "predecision",
) -> HaloExchangeStats:
    """Exchange generated neighbor dependencies through persistent buffers."""
    protocol = generate_halo_protocol(ds, phase=phase)
    if field_names is None:
        field_names = protocol.fields
    h = max(int(shard.halo_width), int(protocol.halo_width))
    owned = shard.interior_rows
    top_owned = slice(owned.start, owned.start + h)
    bottom_owned = slice(owned.stop - h, owned.stop)
    top_halo = slice(0, h)
    bottom_halo = slice(owned.stop, owned.stop + h)
    stats = HaloExchangeStats(protocol=protocol.to_dict())
    cache = ds.metadata.setdefault("_distributed_halo_buffers_v2", {})

    pending_commits = []
    transport.group_start()
    try:
        for name in field_names:
            arr = ds.arrays.get(name)
            if (
                arr is None
                or len(getattr(arr, "shape", ())) < 2
                or arr.shape[0] != shard.local_height
            ):
                continue
            stats.fields += 1
            if shard.north_rank is not None:
                source = arr[top_owned, ...]
                send = _contiguous_buffer(
                    ds, cache, (phase, name, "north", "send"), source.shape, arr.dtype
                )
                recv = _contiguous_buffer(
                    ds, cache, (phase, name, "north", "recv"), arr[top_halo, ...].shape, arr.dtype
                )
                send[...] = source
                transport.send(
                    send,
                    peer=shard.north_rank,
                    stream=stream,
                    tick=tick,
                    phase=phase,
                    field_group=name,
                )
                transport.recv(
                    recv,
                    peer=shard.north_rank,
                    stream=stream,
                    tick=tick,
                    phase=phase,
                    field_group=name,
                )
                pending_commits.append((arr, top_halo, recv))
                stats.elements_sent += int(send.size)
                stats.elements_received += int(recv.size)
            if shard.south_rank is not None:
                source = arr[bottom_owned, ...]
                send = _contiguous_buffer(
                    ds, cache, (phase, name, "south", "send"), source.shape, arr.dtype
                )
                recv = _contiguous_buffer(
                    ds,
                    cache,
                    (phase, name, "south", "recv"),
                    arr[bottom_halo, ...].shape,
                    arr.dtype,
                )
                send[...] = source
                transport.send(
                    send,
                    peer=shard.south_rank,
                    stream=stream,
                    tick=tick,
                    phase=phase,
                    field_group=name,
                )
                transport.recv(
                    recv,
                    peer=shard.south_rank,
                    stream=stream,
                    tick=tick,
                    phase=phase,
                    field_group=name,
                )
                pending_commits.append((arr, bottom_halo, recv))
                stats.elements_sent += int(send.size)
                stats.elements_received += int(recv.size)
    finally:
        transport.group_end()

    # NCCL grouped P2P calls are only enqueued when group_end returns. Queue
    # the halo commits after that point on the same CUDA stream so each copy is
    # ordered after the corresponding receive without a host synchronization.
    for arr, target_slice, recv in pending_commits:
        arr[target_slice, ...] = recv
    return stats
