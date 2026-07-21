"""Complete backend-native source capture and isolated branch cloning."""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from owl.gpu.device_state import OWLDeviceState

PHASE25_REQUIRED_ARRAYS: tuple[str, ...] = (
    "active_sense_food_memory",
    "active_sense_toxin_memory",
    "active_sense_alive_memory",
    "active_sense_ttl",
    "active_sense_new_cell_count",
    "active_sense_new_target_count",
    "action_target_y",
    "action_target_x",
    "action_target_ow_id",
    "action_target_kind",
    "action_target_source",
    "action_target_distance",
    "action_target_confidence",
    "action_direction_y",
    "action_direction_x",
    "action_direction_executable",
    "action_direction_score",
    "action_direction_distance_delta",
    "action_direction_hazard",
    "action_direction_opportunity",
    "flee_compiled_action",
    "pursue_compiled_action",
    "compiled_execution_action",
)

SCIENTIFIC_METADATA_KEYS: tuple[str, ...] = (
    "field_epochs",
    "event_queue",
    "cfg_mode",
    "defer_host_metrics",
    "precision_policy",
    "precision_promoted_fields",
    "raqic_real_dtype",
)


@dataclass(frozen=True)
class CloneField:
    group: str
    name: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int


@dataclass(frozen=True)
class PointerRange:
    owner: str
    group: str
    name: str
    start: int
    end: int


@dataclass(frozen=True)
class CloneManifest:
    fields: tuple[CloneField, ...]
    scalar_names: tuple[str, ...]
    metadata_names: tuple[str, ...]

    @property
    def total_array_bytes(self) -> int:
        return sum(item.nbytes for item in self.fields)


@dataclass
class CounterfactualSourceState:
    """Immutable-by-contract complete source world at the decision seam."""

    source_state_id: str
    backend: Any
    arrays: dict[str, Any]
    patch_arrays: dict[str, Any]
    global_arrays: dict[str, Any]
    scalars: dict[str, Any]
    metadata: dict[str, Any]
    manifest: CloneManifest
    source_root: str | None = None
    ready_event: Any | None = None

    @property
    def nbytes(self) -> int:
        return self.manifest.total_array_bytes

    def branch_clone(self) -> OWLDeviceState:
        """Create a disjoint mutable branch state on the same backend."""
        ds = OWLDeviceState(
            backend=self.backend,
            arrays=_copy_mapping(self.arrays),
            patch_arrays=_copy_mapping(self.patch_arrays),
            global_arrays=_copy_mapping(self.global_arrays),
            scalars=copy.deepcopy(self.scalars),
            metadata=copy.deepcopy(self.metadata),
        )
        assert_no_alias(self, ds)
        return ds


def _copy_mapping(values: Mapping[str, Any]) -> dict[str, Any]:
    return {name: _copy_array(value) for name, value in values.items()}


def _copy_array(value: Any) -> Any:
    """Copy while preserving NumPy SIMD alignment class when practical.

    NumPy may choose different floating reduction kernels for different pointer
    alignment. Matching the source address modulo 64 prevents an allocator-only
    last-bit difference from changing a later categorical decision. CuPy device
    allocations already provide a stable high alignment and use its native copy.
    """
    if not isinstance(value, np.ndarray):
        return value.copy()
    if not value.flags.c_contiguous:
        return value.copy(order="K")
    alignment = 64
    source_mod = int(value.__array_interface__["data"][0]) % alignment
    raw = np.empty(int(value.nbytes) + alignment, dtype=np.uint8)
    base_mod = int(raw.__array_interface__["data"][0]) % alignment
    offset = (source_mod - base_mod) % alignment
    copied = np.ndarray(value.shape, dtype=value.dtype, buffer=raw, offset=offset, order="C")
    copied[...] = value
    return copied


def build_clone_manifest(ds: OWLDeviceState, *, require_phase25: bool = True) -> CloneManifest:
    """Enumerate every authoritative array group dynamically."""
    if require_phase25:
        missing = sorted(set(PHASE25_REQUIRED_ARRAYS) - set(ds.arrays))
        if missing:
            raise RuntimeError(f"Phase 2.5 clone fields missing: {missing}")
    fields: list[CloneField] = []
    for group, mapping in ordered_array_groups(ds):
        for name, value in mapping.items():
            fields.append(
                CloneField(
                    group=group,
                    name=name,
                    dtype=np.dtype(value.dtype).str,
                    shape=tuple(int(item) for item in value.shape),
                    nbytes=int(value.nbytes),
                )
            )
    metadata_names = tuple(name for name in SCIENTIFIC_METADATA_KEYS if name in ds.metadata)
    return CloneManifest(
        fields=tuple(fields),
        scalar_names=tuple(sorted(ds.scalars)),
        metadata_names=metadata_names,
    )


def ordered_array_groups(ds: Any) -> tuple[tuple[str, Mapping[str, Any]], ...]:
    return (
        ("arrays", {name: ds.arrays[name] for name in sorted(ds.arrays)}),
        ("patch_arrays", {name: ds.patch_arrays[name] for name in sorted(ds.patch_arrays)}),
        ("global_arrays", {name: ds.global_arrays[name] for name in sorted(ds.global_arrays)}),
    )


def capture_source_state(
    ds: OWLDeviceState,
    source_state_id: str,
    *,
    stream: Any | None = None,
    ready_event: Any | None = None,
) -> CounterfactualSourceState:
    """Copy a complete source on the active backend without a host checkpoint."""
    manifest = build_clone_manifest(ds, require_phase25=True)
    context = stream if stream is not None else _NullContext()
    with context:
        arrays = _copy_mapping(ds.arrays)
        patch_arrays = _copy_mapping(ds.patch_arrays)
        global_arrays = _copy_mapping(ds.global_arrays)
    if ready_event is not None:
        ready_event.record(stream)
    source = CounterfactualSourceState(
        source_state_id=source_state_id,
        backend=ds.backend,
        arrays=arrays,
        patch_arrays=patch_arrays,
        global_arrays=global_arrays,
        scalars=copy.deepcopy(ds.scalars),
        metadata={name: copy.deepcopy(ds.metadata[name]) for name in manifest.metadata_names},
        manifest=manifest,
        ready_event=ready_event,
    )
    assert_no_alias(ds, source)
    return source


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Literal[False]:
        return False


def _pointer(value: Any) -> int:
    cuda = getattr(value, "__cuda_array_interface__", None)
    if cuda is not None:
        return int(cuda["data"][0])
    array = getattr(value, "__array_interface__", None)
    if array is not None:
        return int(array["data"][0])
    raise TypeError(f"array lacks pointer interface: {type(value).__name__}")


def pointer_ranges(owner: str, state: Any) -> tuple[PointerRange, ...]:
    ranges: list[PointerRange] = []
    for group, mapping in ordered_array_groups(state):
        for name, value in mapping.items():
            start = _pointer(value)
            ranges.append(PointerRange(owner, group, name, start, start + int(value.nbytes)))
    return tuple(ranges)


def _overlap(left: PointerRange, right: PointerRange) -> bool:
    return left.start < right.end and right.start < left.end


def assert_no_alias(*states: Any) -> None:
    """Fail when allocations owned by distinct state objects overlap."""
    groups = [pointer_ranges(f"state-{index}", state) for index, state in enumerate(states)]
    for left_index, left_ranges in enumerate(groups):
        for right_ranges in groups[left_index + 1 :]:
            for left in left_ranges:
                for right in right_ranges:
                    if _overlap(left, right):
                        raise RuntimeError(
                            "counterfactual state alias detected: "
                            f"{left.group}.{left.name} overlaps {right.group}.{right.name}"
                        )


def ranges_do_not_overlap(left: Iterable[PointerRange], right: Iterable[PointerRange]) -> bool:
    return not any(_overlap(a, b) for a in left for b in right)
