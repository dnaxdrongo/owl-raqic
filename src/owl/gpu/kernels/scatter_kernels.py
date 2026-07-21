from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from owl.gpu.field_registry import FIELD_REGISTRY


@dataclass
class FieldSlab:
    names: tuple[str, ...]
    slab: Any
    tail_shape: tuple[int, ...]
    dtype: str


class FieldSlabManager:
    """Registry-driven scatter/gather with persistent-slab fast path.

    When ``owl.gpu.slabs.FieldSlabManager`` is attached to the device state,
    registered arrays are views into persistent dtype/trailing-shape slabs and
    no stack/unpack allocation occurs in the tick loop. The stack path remains
    as a reference for stage-once and compatibility modes.
    """

    def __init__(self, ds: Any) -> None:
        self.ds = ds
        self.persistent = ds.metadata.get("slab_manager")

    def grouped_names(
        self,
        *,
        moves_with_cell: bool | None = None,
        clears_on_death: bool | None = None,
        copy_on_reproduction: bool | None = None,
    ) -> dict[tuple[str, tuple[int, ...]], tuple[str, ...]]:
        groups: dict[tuple[str, tuple[int, ...]], list[str]] = defaultdict(list)
        h, w = self.ds.health.shape
        for name, spec in FIELD_REGISTRY.items():
            if moves_with_cell is not None and spec.moves_with_cell != moves_with_cell:
                continue
            if clears_on_death is not None and spec.clears_on_death != clears_on_death:
                continue
            if (
                copy_on_reproduction is not None
                and spec.copy_on_reproduction != copy_on_reproduction
            ):
                continue
            arr = self.ds.arrays.get(name)
            if arr is None or arr.shape[:2] != (h, w):
                continue
            key = (str(arr.dtype), tuple(arr.shape[2:]))
            groups[key].append(name)
        return {key: tuple(names) for key, names in groups.items()}

    def group_names(self, *, moves_with_cell: bool = True, ndim: int = 2) -> tuple[str, ...]:
        names: list[str] = []
        for grouped in self.grouped_names(moves_with_cell=moves_with_cell).values():
            names.extend(name for name in grouped if self.ds.arrays[name].ndim == ndim)
        return tuple(names)

    def pack(self, names: tuple[str, ...]) -> FieldSlab:
        if not names:
            raise ValueError("cannot pack empty field group")
        arrays = [self.ds.arrays[name] for name in names]
        first = arrays[0]
        if any(arr.dtype != first.dtype or arr.shape[2:] != first.shape[2:] for arr in arrays):
            raise TypeError("field slab requires identical dtype and trailing shape")
        slab = self.ds.xp.stack(arrays, axis=0)
        return FieldSlab(names, slab, tuple(first.shape[2:]), str(first.dtype))

    def unpack(self, field_slab: FieldSlab) -> None:
        for i, name in enumerate(field_slab.names):
            self.ds.arrays[name] = field_slab.slab[i]

    def _persistent_group_selection(self, names: tuple[str, ...]) -> Any:
        if self.persistent is None:
            return []
        wanted = set(names)
        selections = []
        for group, group_names in self.persistent.field_names.items():
            indices = [i for i, name in enumerate(group_names) if name in wanted]
            if indices:
                selections.append((self.persistent.slabs[group], indices))
        return selections

    def _move_group(self, names: Any, sy: Any, sx: Any, dy: Any, dx: Any) -> None:
        persistent = self._persistent_group_selection(tuple(names))
        if persistent:
            for slab, indices in persistent:
                index = self.ds.xp.asarray(indices, dtype=self.ds.xp.int32)
                values = slab[index[:, None], sy[None, :], sx[None, :], ...].copy()
                slab[index[:, None], sy[None, :], sx[None, :], ...] = 0
                slab[index[:, None], dy[None, :], dx[None, :], ...] = values
            return
        field_slab = self.pack(tuple(names))
        out = field_slab.slab.copy()
        values = field_slab.slab[:, sy, sx, ...].copy()
        out[:, sy, sx, ...] = 0
        out[:, dy, dx, ...] = values
        self.unpack(FieldSlab(field_slab.names, out, field_slab.tail_shape, field_slab.dtype))

    def move_all_registered(self, sy: Any, sx: Any, dy: Any, dx: Any) -> None:
        for names in self.grouped_names(moves_with_cell=True).values():
            self._move_group(names, sy, sx, dy, dx)

    def copy_all_registered(self, sy: Any, sx: Any, ty: Any, tx: Any) -> None:
        for names in self.grouped_names(copy_on_reproduction=True).values():
            persistent = self._persistent_group_selection(names)
            if persistent:
                for slab, indices in persistent:
                    index = self.ds.xp.asarray(indices, dtype=self.ds.xp.int32)
                    values = slab[index[:, None], sy[None, :], sx[None, :], ...].copy()
                    slab[index[:, None], ty[None, :], tx[None, :], ...] = values
                continue
            field_slab = self.pack(names)
            out = field_slab.slab.copy()
            out[:, ty, tx, ...] = field_slab.slab[:, sy, sx, ...]
            self.unpack(FieldSlab(names, out, field_slab.tail_shape, field_slab.dtype))

    def clear_all_registered(self, dead_mask: Any) -> None:
        for names in self.grouped_names(clears_on_death=True).values():
            persistent = self._persistent_group_selection(names)
            persistent_names: set[str] = set()
            for slab, indices in persistent:
                group_names = self.persistent.field_names[
                    next(group for group, value in self.persistent.slabs.items() if value is slab)
                ]
                persistent_names.update(group_names[i] for i in indices)
                index = self.ds.xp.asarray(indices, dtype=self.ds.xp.int32)
                selected = slab[index, ...]
                mask = dead_mask[None, ...]
                while mask.ndim < selected.ndim:
                    mask = mask[..., None]
                slab[index, ...] = self.ds.xp.where(mask, 0, selected)
            remaining = tuple(name for name in names if name not in persistent_names)
            if remaining:
                field_slab = self.pack(remaining)
                mask = dead_mask[None, ...]
                while mask.ndim < field_slab.slab.ndim:
                    mask = mask[..., None]
                field_slab.slab = self.ds.xp.where(mask, 0, field_slab.slab)
                self.unpack(field_slab)

    def move_slab_2d(self, names: tuple[str, ...], sy: Any, sx: Any, dy: Any, dx: Any) -> None:
        if names:
            self._move_group(names, sy, sx, dy, dx)

    def clear_dead_slab_2d(self, names: tuple[str, ...], dead_mask: Any) -> None:
        if not names:
            return
        for group_names in self.grouped_names(clears_on_death=True).values():
            chosen = tuple(name for name in group_names if name in set(names))
            if chosen:
                self.clear_all_registered(dead_mask)
                return
