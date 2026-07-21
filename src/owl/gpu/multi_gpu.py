"""Provide single-process sharding helpers for reference validation.

Distributed execution is implemented in :mod:`owl.gpu.distributed` with one
process per GPU, collective communication, overlap verification, and certificate
evidence. This module does not certify distributed runs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

DEPRECATED_PROTOTYPE = True


@dataclass(frozen=True)
class SpatialShard:
    device_id: int
    row_start: int
    row_stop: int
    halo: int = 1

    @property
    def rows(self) -> int:
        return self.row_stop - self.row_start


def partition_rows(height: int, device_ids: list[int], *, halo: int = 1) -> list[SpatialShard]:
    if height < 0 or not device_ids:
        raise ValueError("height must be nonnegative and at least one device is required")
    q, r = divmod(int(height), len(device_ids))
    shards = []
    start = 0
    for i, device in enumerate(device_ids):
        stop = start + q + (1 if i < r else 0)
        shards.append(SpatialShard(int(device), start, stop, int(halo)))
        start = stop
    return shards


def available_cuda_devices() -> list[int]:
    try:
        import cupy as cp

        return list(range(int(cp.cuda.runtime.getDeviceCount())))
    except Exception:
        return []


class MultiGPUCellExecutor:
    """Single-node deterministic cell-shard executor.

    This is suitable for independent RAQIC batch kernels and other
    embarrassingly parallel cell functions. Spatial stencils require explicit
    halo exchange and are intentionally not hidden behind this class.
    """

    def __init__(self, device_ids: list[int] | None = None) -> None:
        self.device_ids = list(available_cuda_devices() if device_ids is None else device_ids)
        if not self.device_ids:
            raise RuntimeError("no CUDA devices available")

    def map_rows(self, array: Any, fn: Callable[[Any, SpatialShard], Any], *, halo: int = 0) -> Any:
        import cupy as cp

        shards = partition_rows(array.shape[0], self.device_ids, halo=halo)
        outputs = []
        for shard in shards:
            with cp.cuda.Device(shard.device_id):
                local = cp.asarray(array[shard.row_start : shard.row_stop])
                outputs.append((shard, fn(local, shard)))
        return outputs

    @staticmethod
    def gather_numpy(outputs: Any) -> Any:
        import cupy as cp

        ordered = sorted(outputs, key=lambda x: x[0].row_start)
        return __import__("numpy").concatenate([cp.asnumpy(value) for _, value in ordered], axis=0)


def exchange_row_halos(shard_arrays: list[Any], *, halo: int = 1) -> list[Any]:
    """Return arrays padded with neighbor rows for a single-node prototype.

    CuPy performs peer/device copies when supported. This function is explicit
    and synchronized; it is an audit prototype, not an NCCL performance claim.
    """
    if halo <= 0:
        return shard_arrays
    import cupy as cp

    out = []
    for i, arr in enumerate(shard_arrays):
        with cp.cuda.Device(int(arr.device.id)):
            top = shard_arrays[i - 1][-halo:] if i > 0 else arr[:halo]
            bottom = shard_arrays[i + 1][:halo] if i + 1 < len(shard_arrays) else arr[-halo:]
            out.append(cp.concatenate([cp.asarray(top), arr, cp.asarray(bottom)], axis=0))
    cp.cuda.Device().synchronize()
    return out
