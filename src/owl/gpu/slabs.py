from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from owl.gpu.field_registry import FIELD_REGISTRY


@dataclass(frozen=True)
class SlabSlice:
    group: str
    index: int
    trailing_shape: tuple[int, ...]
    dtype: str


@dataclass
class FieldSlabManager:
    """Persistent slab layout for registered cell-resident arrays.

    Each group contains fields of identical dtype and trailing shape. Replacing
    ``device_state.arrays[name]`` with slab views keeps existing stage code
    readable while movement/reproduction can operate on contiguous groups.
    """

    backend: Any
    slabs: dict[str, Any] = field(default_factory=dict)
    slices: dict[str, SlabSlice] = field(default_factory=dict)
    field_names: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @classmethod
    def attach(cls, device_state: Any, *, include_nonmoving: bool = False) -> FieldSlabManager:
        manager = cls(device_state.backend)
        h = int(
            device_state.metadata.get("height", 0)
            or next(iter(device_state.arrays.values())).shape[0]
        )
        w = int(
            device_state.metadata.get("width", 0)
            or next(iter(device_state.arrays.values())).shape[1]
        )
        groups: dict[tuple[str, tuple[int, ...], str], list[str]] = {}
        for name, arr in device_state.arrays.items():
            spec = FIELD_REGISTRY.get(name)
            if spec is None or spec.owner != "cell_resident":
                continue
            if not include_nonmoving and not (spec.moves_with_cell or spec.copy_on_reproduction):
                continue
            if getattr(arr, "ndim", 0) < 2 or tuple(arr.shape[:2]) != (h, w):
                continue
            dtype = str(arr.dtype)
            trailing = tuple(int(x) for x in arr.shape[2:])
            group_key = spec.layout_group or "cell"
            groups.setdefault((dtype, trailing, group_key), []).append(name)

        xp = device_state.xp
        for n_group, ((dtype, trailing, layout), names) in enumerate(
            sorted(groups.items(), key=str)
        ):
            names = sorted(names)
            group = f"{layout}:{dtype}:{'x'.join(map(str, trailing)) or 'scalar'}:{n_group}"
            slab = xp.empty((len(names), h, w, *trailing), dtype=np.dtype(dtype))
            for index, name in enumerate(names):
                slab[index, ...] = device_state.arrays[name]
                device_state.arrays[name] = slab[index, ...]
                manager.slices[name] = SlabSlice(group, index, trailing, dtype)
            manager.slabs[group] = slab
            manager.field_names[group] = tuple(names)
        device_state.metadata["slab_groups"] = {
            group: {
                "fields": list(names),
                "shape": tuple(manager.slabs[group].shape),
                "dtype": str(manager.slabs[group].dtype),
            }
            for group, names in manager.field_names.items()
        }
        device_state.metadata["slab_manager"] = manager
        return manager

    def view(self, name: str) -> Any:
        sl = self.slices[name]
        return self.slabs[sl.group][sl.index]

    def field_group(self, name: str) -> tuple[Any, tuple[str, ...]]:
        sl = self.slices[name]
        return self.slabs[sl.group], self.field_names[sl.group]

    def groups_for(self, names: Any) -> list[tuple[Any, tuple[str, ...]]]:
        wanted = set(names)
        out = []
        for group, group_names in self.field_names.items():
            selected = tuple(n for n in group_names if n in wanted)
            if selected:
                out.append((self.slabs[group], selected))
        return out

    def assert_views_current(self, device_state: Any) -> None:
        for name, sl in self.slices.items():
            arr = device_state.arrays[name]
            if arr.shape != self.slabs[sl.group][sl.index].shape:
                raise AssertionError(f"slab view shape mismatch for {name}")
            # Device pointer comparison is backend-specific; object identity is
            # reliable for NumPy views, while shared-memory check handles CuPy.
            if self.backend.is_gpu:
                if int(arr.data.ptr) != int(self.slabs[sl.group][sl.index].data.ptr):
                    raise AssertionError(f"stale slab view for {name}")
            elif not np.shares_memory(arr, self.slabs[sl.group]):
                raise AssertionError(f"stale slab view for {name}")

    def layout(self) -> dict[str, Any]:
        return {
            "groups": {
                group: {
                    "fields": list(self.field_names[group]),
                    "shape": list(slab.shape),
                    "dtype": str(slab.dtype),
                    "bytes": int(slab.nbytes),
                }
                for group, slab in self.slabs.items()
            },
            "total_bytes": int(sum(s.nbytes for s in self.slabs.values())),
        }
