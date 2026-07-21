"""Provide CPU reference partitioning helpers for focused validation.

Distributed production execution is implemented in
:mod:`owl.gpu.distributed` with one process per GPU. This module does not
certify distributed runs.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

DEPRECATED_PROTOTYPE = True


@dataclass(frozen=True)
class SpatialShard:
    device_id: int
    row_start: int
    row_stop: int
    halo_top: int
    halo_bottom: int

    @property
    def owned_rows(self) -> int:
        return self.row_stop - self.row_start


def partition_rows(height: int, device_ids: Iterable[int], halo: int = 1) -> list[SpatialShard]:
    ids = [int(x) for x in device_ids]
    if height <= 0:
        raise ValueError("height must be positive")
    if not ids:
        raise ValueError("at least one device id is required")
    if halo < 0:
        raise ValueError("halo must be nonnegative")
    if len(ids) > height:
        raise ValueError("cannot assign more devices than grid rows")
    base, rem = divmod(height, len(ids))
    out: list[SpatialShard] = []
    start = 0
    for i, dev in enumerate(ids):
        stop = start + base + (1 if i < rem else 0)
        out.append(
            SpatialShard(
                device_id=dev,
                row_start=start,
                row_stop=stop,
                halo_top=min(halo, start),
                halo_bottom=min(halo, height - stop),
            )
        )
        start = stop
    assert start == height
    return out


def split_live_indices(indices: np.ndarray, device_ids: Iterable[int]) -> dict[int, np.ndarray]:
    """Deterministically shard flattened live-cell indices by contiguous chunks."""
    ids = [int(x) for x in device_ids]
    if not ids:
        raise ValueError("at least one device id is required")
    arr = np.asarray(indices, dtype=np.int64).reshape(-1)
    chunks = np.array_split(arr, len(ids))
    return {dev: chunk.copy() for dev, chunk in zip(ids, chunks, strict=True)}


def extract_halo(field: np.ndarray, shard: SpatialShard) -> np.ndarray:
    """Return owned rows plus bounded halo rows for CPU/reference testing."""
    arr = np.asarray(field)
    lo = shard.row_start - shard.halo_top
    hi = shard.row_stop + shard.halo_bottom
    return arr[lo:hi].copy()


def validate_partition(height: int, shards: list[SpatialShard]) -> None:
    if not shards:
        raise ValueError("empty shard list")
    cursor = 0
    seen: set[int] = set()
    for shard in shards:
        if shard.device_id in seen:
            raise ValueError("duplicate device id")
        seen.add(shard.device_id)
        if shard.row_start != cursor or shard.row_stop <= shard.row_start:
            raise ValueError("partition is not contiguous/nonempty")
        cursor = shard.row_stop
    if cursor != height:
        raise ValueError("partition does not cover the full grid")
