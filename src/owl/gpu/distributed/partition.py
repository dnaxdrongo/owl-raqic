from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SpatialShard:
    rank: int
    world_size: int
    world_height: int
    world_width: int
    owned_start: int
    owned_stop: int
    halo_width: int
    boundary_mode: str

    @property
    def owned_rows(self) -> slice:
        return slice(self.owned_start, self.owned_stop)

    @property
    def owned_height(self) -> int:
        return self.owned_stop - self.owned_start

    @property
    def local_height(self) -> int:
        return self.owned_height + 2 * self.halo_width

    @property
    def interior_rows(self) -> slice:
        return slice(self.halo_width, self.halo_width + self.owned_height)

    @property
    def north_rank(self) -> int | None:
        if self.rank > 0:
            return self.rank - 1
        return self.world_size - 1 if self.boundary_mode == "toroidal" else None

    @property
    def south_rank(self) -> int | None:
        if self.rank + 1 < self.world_size:
            return self.rank + 1
        return 0 if self.boundary_mode == "toroidal" else None

    def global_row_indices(self) -> tuple[int, ...]:
        rows = []
        for local in range(-self.halo_width, self.owned_height + self.halo_width):
            global_row = self.owned_start + local
            if self.boundary_mode == "toroidal":
                global_row %= self.world_height
            else:
                global_row = min(max(global_row, 0), self.world_height - 1)
            rows.append(global_row)
        return tuple(rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "world_height": self.world_height,
            "world_width": self.world_width,
            "owned_start": self.owned_start,
            "owned_stop": self.owned_stop,
            "halo_width": self.halo_width,
            "north_rank": self.north_rank,
            "south_rank": self.south_rank,
        }


def partition_rows(
    height: int,
    width: int,
    world_size: int,
    patch_size: int,
    *,
    boundary_mode: str = "toroidal",
    halo_width: int | None = None,
) -> tuple[SpatialShard, ...]:
    """Partition rows deterministically while preserving complete patch rows."""
    height, width, world_size, patch_size = map(int, (height, width, world_size, patch_size))
    if height <= 0 or width <= 0 or world_size <= 0 or patch_size <= 0:
        raise ValueError("height, width, world_size, and patch_size must be positive")
    if height % patch_size:
        raise ValueError("world height must be divisible by patch_size")
    patch_rows = height // patch_size
    if world_size > patch_rows:
        raise ValueError(
            "world_size cannot exceed the number of patch rows; each rank must own "
            "at least one complete patch row"
        )
    base, remainder = divmod(patch_rows, world_size)
    halo = max(patch_size, 2) if halo_width is None else int(halo_width)
    if halo < 1:
        raise ValueError("halo_width must be positive")
    shards = []
    patch_cursor = 0
    for rank in range(world_size):
        count = base + (1 if rank < remainder else 0)
        start = patch_cursor * patch_size
        stop = (patch_cursor + count) * patch_size
        shards.append(
            SpatialShard(
                rank=rank,
                world_size=world_size,
                world_height=height,
                world_width=width,
                owned_start=start,
                owned_stop=stop,
                halo_width=halo,
                boundary_mode=str(boundary_mode),
            )
        )
        patch_cursor += count
    assert shards[-1].owned_stop == height
    return tuple(shards)
