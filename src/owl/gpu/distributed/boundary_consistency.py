from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from owl.gpu.transfer_ledger import TransferLedger

from .partition import SpatialShard


@dataclass
class BoundaryConsistencyReport:
    checked_fields: int
    compared_elements: int
    local_mismatch_fields: tuple[str, ...]
    global_mismatch: bool
    max_abs_residual: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "checked_fields": int(self.checked_fields),
            "compared_elements": int(self.compared_elements),
            "local_mismatch_fields": list(self.local_mismatch_fields),
            "global_mismatch": bool(self.global_mismatch),
            "max_abs_residual": float(self.max_abs_residual),
        }


def _candidate_fields(ds: Any, shard: SpatialShard, field_names: Any | None = None) -> Any:
    names = sorted(ds.arrays) if field_names is None else tuple(field_names)
    for name in names:
        arr = ds.arrays.get(name)
        if arr is None or not hasattr(arr, "shape") or not hasattr(arr, "dtype"):
            continue
        if len(arr.shape) < 2 or int(arr.shape[0]) != int(shard.local_height):
            continue
        if str(arr.dtype) in {"object", "complex128", "complex64"}:
            # Complex diagnostic arrays are not currently present in device
            # state; reject instead of silently communicating unsupported NCCL
            # types should one be introduced.
            continue
        yield name, arr


def verify_and_commit_boundaries(
    ds: Any,
    shard: SpatialShard,
    transport: Any,
    stream: Any,
    *,
    tick: int,
    field_names: Any | None = None,
    tolerance: float = 1e-8,
    strict: bool = True,
) -> BoundaryConsistencyReport:
    """Verify redundant overlap before committing target-owned boundary data.

    Every rank computes its owned rows and halo overlap.  Neighbor-owned rows
    are then received into temporary buffers, compared with the local redundant
    overlap, and committed only after all ranks participate in a global mismatch
    reduction.  This provides deterministic movement/reproduction/topology
    boundary semantics without silently accepting divergent rank outcomes.
    """

    xp = ds.xp
    h = int(shard.halo_width)
    owned = shard.interior_rows
    top_owned = slice(owned.start, owned.start + h)
    bottom_owned = slice(owned.stop - h, owned.stop)
    top_halo = slice(0, h)
    bottom_halo = slice(owned.stop, owned.stop + h)

    cache = ds.metadata.setdefault("_distributed_boundary_buffers", {})
    exchanges = []
    transport.group_start()
    try:
        for name, arr in _candidate_fields(ds, shard, field_names):
            entry = cache.setdefault(name, {})
            if shard.north_rank is not None:
                send = entry.get("north_send")
                source = arr[top_owned, ...]
                if send is None or send.shape != source.shape or send.dtype != arr.dtype:
                    send = xp.empty(source.shape, dtype=arr.dtype)
                    entry["north_send"] = send
                send[...] = source
                recv = entry.get("north")
                expected_shape = arr[top_halo, ...].shape
                if recv is None or recv.shape != expected_shape or recv.dtype != arr.dtype:
                    recv = xp.empty(expected_shape, dtype=arr.dtype)
                    entry["north"] = recv
                transport.send(
                    send,
                    peer=shard.north_rank,
                    stream=stream,
                    tick=tick,
                    phase="boundary_consistency",
                    field_group=name,
                )
                transport.recv(
                    recv,
                    peer=shard.north_rank,
                    stream=stream,
                    tick=tick,
                    phase="boundary_consistency",
                    field_group=name,
                )
                exchanges.append((name, "north", arr, top_halo, recv))
            if shard.south_rank is not None:
                send = entry.get("south_send")
                source = arr[bottom_owned, ...]
                if send is None or send.shape != source.shape or send.dtype != arr.dtype:
                    send = xp.empty(source.shape, dtype=arr.dtype)
                    entry["south_send"] = send
                send[...] = source
                recv = entry.get("south")
                expected_shape = arr[bottom_halo, ...].shape
                if recv is None or recv.shape != expected_shape or recv.dtype != arr.dtype:
                    recv = xp.empty(expected_shape, dtype=arr.dtype)
                    entry["south"] = recv
                transport.send(
                    send,
                    peer=shard.south_rank,
                    stream=stream,
                    tick=tick,
                    phase="boundary_consistency",
                    field_group=name,
                )
                transport.recv(
                    recv,
                    peer=shard.south_rank,
                    stream=stream,
                    tick=tick,
                    phase="boundary_consistency",
                    field_group=name,
                )
                exchanges.append((name, "south", arr, bottom_halo, recv))
    finally:
        transport.group_end()

    status_cache = ds.metadata.setdefault("_distributed_boundary_status", {})
    exchange_count = len(exchanges)
    residuals_device = status_cache.get("residuals")
    mismatch_flags_device = status_cache.get("mismatch_flags")
    if residuals_device is None or int(residuals_device.shape[0]) != exchange_count:
        residuals_device = xp.zeros((exchange_count,), dtype=xp.float64)
        mismatch_flags_device = xp.zeros((exchange_count,), dtype=xp.int32)
        status_cache["residuals"] = residuals_device
        status_cache["mismatch_flags"] = mismatch_flags_device
    else:
        residuals_device[...] = 0.0
        mismatch_flags_device[...] = 0

    compared = 0
    for index, (_name, _direction, arr, local_slice, received) in enumerate(exchanges):
        local = arr[local_slice, ...]
        compared += int(local.size)
        if arr.dtype.kind in {"f", "c"}:
            residuals_device[index] = xp.max(xp.abs(local - received))
            mismatch_flags_device[index] = residuals_device[index] > float(tolerance)
        else:
            mismatch_flags_device[index] = xp.any(local != received)

    local_flag = status_cache.get("local_flag")
    global_flag = status_cache.get("global_flag")
    if local_flag is None:
        local_flag = xp.zeros((1,), dtype=xp.int32)
        global_flag = xp.zeros_like(local_flag)
        status_cache["local_flag"] = local_flag
        status_cache["global_flag"] = global_flag
    local_flag[0] = xp.max(mismatch_flags_device) if exchange_count else 0
    global_flag[...] = 0
    transport.all_reduce(
        local_flag,
        global_flag,
        op="max",
        stream=stream,
        tick=tick,
        phase="boundary_consistency",
        field_group="mismatch_flag",
    )
    stream.synchronize()
    status_device = xp.concatenate(
        (
            residuals_device,
            mismatch_flags_device.astype(xp.float64),
            global_flag.astype(xp.float64),
        )
    )
    status_host = ds.backend.asnumpy(status_device)
    transfer_ledger = ds.metadata.get("transfer_ledger")
    if isinstance(transfer_ledger, TransferLedger):
        transfer_ledger.record_d2h(
            int(status_device.nbytes),
            kind="distributed_verify",
            tick=int(tick),
            source_stream="distributed-boundary",
            synchronization="stream",
            scheduled=True,
            graph_compatible=False,
            reason="one compact distributed boundary verification status record",
        )
    residual_values = status_host[:exchange_count]
    mismatch_values = status_host[exchange_count : 2 * exchange_count]
    global_mismatch = bool(int(status_host[-1]))
    mismatch_names = {
        name
        for (name, _direction, _arr, _local_slice, _received), flag in zip(
            exchanges, mismatch_values, strict=True
        )
        if bool(flag)
    }
    max_residual = float(max(residual_values, default=0.0))
    if global_mismatch and strict:
        raise RuntimeError(
            "distributed boundary verification failed at tick "
            f"{tick}; local mismatches={sorted(mismatch_names)}, "
            f"max_abs_residual={max_residual}"
        )

    # Neighbor ownership is authoritative for halo data.
    if not global_mismatch:
        for _name, _direction, arr, local_slice, received in exchanges:
            arr[local_slice, ...] = received

    report = BoundaryConsistencyReport(
        checked_fields=len({name for name, *_ in exchanges}),
        compared_elements=compared,
        local_mismatch_fields=tuple(sorted(mismatch_names)),
        global_mismatch=global_mismatch,
        max_abs_residual=max_residual,
    )
    ds.metadata["last_boundary_consistency"] = report.to_dict()
    return report
