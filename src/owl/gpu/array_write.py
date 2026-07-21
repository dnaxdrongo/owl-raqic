from __future__ import annotations

from typing import Any


def _write(mapping: dict[str, Any], name: str, value: Any) -> Any:
    """Write a stage result without changing an existing allocation.

    Persistent slabs, CUDA graphs, and asynchronous consumers depend on stable
    addresses.  When a compatible destination already exists, this function
    updates it in place and preserves its canonical dtype.  New/scratch fields
    are allocated once and are subsequently updated in place.
    """
    current = mapping.get(name)
    if current is not None and getattr(current, "shape", None) == getattr(value, "shape", None):
        current[...] = value
        return current
    mapping[name] = value
    return value


def write_array(ds: Any, name: str, value: Any) -> Any:
    return _write(ds.arrays, name, value)


def write_patch_array(ds: Any, name: str, value: Any) -> Any:
    return _write(ds.patch_arrays, name, value)


def write_global_array(ds: Any, name: str, value: Any) -> Any:
    return _write(ds.global_arrays, name, value)
