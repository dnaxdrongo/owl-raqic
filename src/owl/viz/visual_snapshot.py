from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from owl.viz.event_bus import VisualEvent

_VISUAL_FIELDS: tuple[str, ...] = (
    "health",
    "resource",
    "toxin",
    "food",
    "waste",
    "obstacle",
    "occupancy",
    "readout",
    "raqic_readout",
    "raqic_probabilities",
    "possibility",
    "last_utilities",
    "last_logits",
    "last_action_probabilities",
    "pre_utilities",
    "pre_authority",
    "raqic_parent_intention",
    "raqic_score",
    "raqic_phase",
    "raqic_pre_mixer_probabilities",
    "raqic_utility_innovation",
    "raqic_phase_alignment",
    "raqic_resonant_parent_intention",
    "raqic_shadow_probabilities",
    "raqic_utility_innovation_norm",
    "raqic_utility_projection_fraction",
    "raqic_utility_score_cosine",
    "raqic_utility_orthogonality_residual",
    "raqic_policy_kl",
    "raqic_interference_delta_l1",
    "raqic_interference_norm_error",
    "raqic_interference_illegal_mass",
    "raqic_patch_action_phase",
    "raqic_patch_action_coherence",
    "raqic_global_action_phase",
    "raqic_global_action_coherence",
    "raqic_parent_action_phase",
    "raqic_parent_action_coherence",
    "authority",
    "_authority_bool",
    "raqic_record_action",
    "raqic_record_readout",
    "raqic_record_confidence",
    "integration",
    "noetic_C",
    "phase",
    "boundary",
    "age",
    "ow_type",
    "lineage_id",
    "parent_id",
    "mobility",
    "metabolism",
    "predation",
    "grazing",
    "cooperation",
    "aggression",
    "curiosity",
    "reproduction_rate",
    "toxin_resistance",
    "memory_capacity",
    "coupling_strength",
    "emit_strength",
    "signal_precision",
    "honesty_bias",
    "deception_bias",
    "signal_emission",
    "starvation_debt",
    "development_stage",
    "last_death_mask",
    "genome",
)


@dataclass(frozen=True)
class VisualSnapshot:
    """Immutable host-side visual boundary.

    The snapshot owns every array it exposes.  It never retains a device array,
    mutable WorldState array, event queue, or scientific configuration object.
    """

    tick: int
    world_shape: tuple[int, int]
    boundary_mode: str
    arrays: Mapping[str, np.ndarray]
    events: tuple[VisualEvent, ...]
    id_to_position: Mapping[int, tuple[int, int]]
    metadata: Mapping[str, Any]

    def field(self, name: str) -> np.ndarray:
        try:
            return self.arrays[name]
        except KeyError as exc:
            raise KeyError(f"visual snapshot does not contain field {name!r}") from exc

    def position_of(self, ow_id: int) -> tuple[int, int] | None:
        return self.id_to_position.get(int(ow_id))


def freeze_array(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    """Return an owning, read-only NumPy array."""

    result: np.ndarray[Any, np.dtype[Any]] = np.array(value, dtype=dtype, copy=True, order="C")
    result.flags.writeable = False
    return result


def build_stable_id_index(
    occupancy: np.ndarray,
    health: np.ndarray,
) -> dict[int, tuple[int, int]]:
    occupied = np.asarray(occupancy)
    living = (np.asarray(health) > 0) & (occupied >= 0)
    width = int(living.shape[1])
    flat: np.ndarray[Any, np.dtype[Any]] = np.flatnonzero(living.reshape(-1, order="C"))
    ids: np.ndarray[Any, np.dtype[Any]] = occupied.reshape(-1, order="C")[flat]
    y: np.ndarray[Any, np.dtype[Any]] = flat // width
    x: np.ndarray[Any, np.dtype[Any]] = flat % width
    return {
        int(ow_id): (int(row), int(column)) for ow_id, row, column in zip(ids, y, x, strict=True)
    }


def _selection_fields(selection: Any | None) -> tuple[str, ...]:
    requested = getattr(selection, "fields", ()) if selection is not None else ()
    if requested:
        return tuple(dict.fromkeys((*_VISUAL_FIELDS, *(str(name) for name in requested))))
    return _VISUAL_FIELDS


def _boundary_mode_from(value: Any) -> str:
    cfg = getattr(value, "metadata", {}).get("cfg") if hasattr(value, "metadata") else None
    mode = getattr(getattr(cfg, "world", None), "boundary_mode", None)
    if mode is None:
        mode = getattr(value, "boundary_mode", "toroidal")
    return str(mode)


def _device_array_map(ds: Any) -> dict[str, Any]:
    """Return device arrays without shadowing authoritative world fields.

    DeviceState exposes world-scale arrays plus patch/global diagnostic arrays.
    Some patch/global maps can contain generic lower-resolution names such as
    ``health``.  Visual snapshots must use ``ds.arrays`` as authoritative for
    world-scale fields; patch/global maps are additive only when names do not
    collide.
    """

    arrays: dict[str, Any] = dict(getattr(ds, "arrays", {}))
    for map_name in ("patch_arrays", "global_arrays"):
        for name, value in dict(getattr(ds, map_name, {})).items():
            arrays.setdefault(name, value)
    return arrays


def _copy_device_array(
    ds: Any, name: str, arrays: Mapping[str, Any] | None = None
) -> np.ndarray | None:
    source = _device_array_map(ds) if arrays is None else arrays
    if name not in source:
        return None
    return freeze_array(ds.backend.asnumpy(source[name]))


def _copy_device_arrays(ds: Any, names: Sequence[str]) -> dict[str, np.ndarray]:
    """Copy requested device fields with one synchronization when CuPy is active.

    The pinned arrays become the immutable snapshot owners, eliminating the
    former synchronous copy followed by a second NumPy owning copy.  Other
    backends retain the established compatibility path.
    """

    source_arrays = _device_array_map(ds)
    available = [(name, source_arrays[name]) for name in names if name in source_arrays]
    xp_name = str(getattr(getattr(ds, "xp", None), "__name__", ""))
    if not bool(getattr(ds, "is_gpu", False)) or not xp_name.startswith("cupy"):
        return {
            name: value
            for name in names
            if (value := _copy_device_array(ds, name, source_arrays)) is not None
        }
    try:
        import cupy as cp
    except ImportError:
        return {
            name: value
            for name in names
            if (value := _copy_device_array(ds, name, source_arrays)) is not None
        }
    total_bytes = sum(int(array.nbytes) for _, array in available)
    max_pinned = int(os.environ.get("OWL_REPLAY_PINNED_SNAPSHOT_MAX_BYTES", str(2 * 1024**3)))
    if total_bytes > max_pinned:
        return {
            name: value
            for name in names
            if (value := _copy_device_array(ds, name, source_arrays)) is not None
        }
    stream = cp.cuda.Stream(non_blocking=True)
    copied: dict[str, np.ndarray] = {}
    with stream:
        for name, source in available:
            dtype = np.dtype(source.dtype)
            count = int(np.prod(source.shape, dtype=np.int64))
            owner = cp.cuda.alloc_pinned_memory(count * int(dtype.itemsize))
            host = np.frombuffer(owner, dtype=dtype, count=count).reshape(source.shape)
            cp.asnumpy(source, out=host, stream=stream, blocking=False)
            copied[name] = host
        ready = cp.cuda.Event()
        ready.record(stream)
    ready.synchronize()
    for array in copied.values():
        array.flags.writeable = False
    return copied


def _copy_world_array(state: Any, name: str) -> np.ndarray | None:
    value = getattr(state, name, None)
    if value is None or not hasattr(value, "shape"):
        return None
    return freeze_array(value)


def _snapshot(
    *,
    tick: int,
    boundary_mode: str,
    arrays: dict[str, np.ndarray],
    events: Sequence[VisualEvent],
    metadata: Mapping[str, Any] | None,
) -> VisualSnapshot:
    if "health" not in arrays:
        raise ValueError("visual snapshot requires health")
    shape = tuple(int(v) for v in arrays["health"].shape[:2])
    occupancy = arrays.get("occupancy", freeze_array(np.full(shape, -1, dtype=np.int64)))
    id_to_position = build_stable_id_index(occupancy, arrays["health"])
    frozen_arrays = MappingProxyType(dict(arrays))
    frozen_positions = MappingProxyType(id_to_position)
    frozen_metadata = MappingProxyType(dict(metadata or {}))
    return VisualSnapshot(
        tick=int(tick),
        world_shape=(shape[0], shape[1]),
        boundary_mode=str(boundary_mode),
        arrays=frozen_arrays,
        events=tuple(events),
        id_to_position=frozen_positions,
        metadata=frozen_metadata,
    )


def snapshot_from_device_state(
    ds: Any,
    selection: Any | None = None,
    *,
    events: Sequence[VisualEvent] = (),
    field_names: Sequence[str] | None = None,
) -> VisualSnapshot:
    requested = tuple(field_names) if field_names is not None else _selection_fields(selection)
    arrays = _copy_device_arrays(ds, requested)
    return _snapshot(
        tick=int(ds.tick),
        boundary_mode=_boundary_mode_from(ds),
        arrays=arrays,
        events=events,
        metadata={
            "source": "device",
            "backend": str(getattr(ds.backend, "name", "unknown")),
            "is_gpu": bool(getattr(ds, "is_gpu", False)),
        },
    )


def snapshot_from_world_state(
    state: Any,
    selection: Any | None = None,
    *,
    events: Sequence[VisualEvent] = (),
    boundary_mode: str | None = None,
) -> VisualSnapshot:
    arrays: dict[str, np.ndarray] = {}
    for name in _selection_fields(selection):
        value = _copy_world_array(state, name)
        if value is not None:
            arrays[name] = value
    return _snapshot(
        tick=int(getattr(state, "tick", 0)),
        boundary_mode=boundary_mode or _boundary_mode_from(state),
        arrays=arrays,
        events=events,
        metadata={"source": "world", "backend": "numpy", "is_gpu": False},
    )


def hash_snapshot_fields(snapshot: VisualSnapshot) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, value in sorted(snapshot.arrays.items()):
        array = np.ascontiguousarray(value)
        digest = hashlib.sha256()
        digest.update(str(array.shape).encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(array.view(np.uint8))
        hashes[name] = digest.hexdigest()
    return hashes


def snapshot_from_npz(
    path: str,
    selection: Any | None = None,
    *,
    tick: int = 0,
    boundary_mode: str = "toroidal",
    events: Sequence[VisualEvent] = (),
) -> VisualSnapshot:
    """Load a recorder NPZ into the immutable visual boundary.

    This is intended for visual replay, style validation, and offline rendering;
    it does not write back to scientific state.
    """

    loaded = np.load(path)
    wanted = set(_selection_fields(selection))
    arrays: dict[str, np.ndarray] = {}
    for key in loaded.files:
        if not key.startswith("world/"):
            continue
        name = key.split("/", 1)[1]
        if name in wanted:
            arrays[name] = freeze_array(loaded[key])
    return _snapshot(
        tick=int(tick),
        boundary_mode=boundary_mode,
        arrays=arrays,
        events=events,
        metadata={"source": "npz_replay", "path": str(path), "backend": "numpy"},
    )


def snapshot_from_arrays(
    *,
    tick: int,
    boundary_mode: str,
    arrays: Mapping[str, Any],
    events: Sequence[VisualEvent] = (),
    metadata: Mapping[str, Any] | None = None,
) -> VisualSnapshot:
    """Create a replay snapshot from host arrays using the immutable boundary."""

    copied = {str(name): freeze_array(value) for name, value in arrays.items()}
    return _snapshot(
        tick=int(tick),
        boundary_mode=str(boundary_mode),
        arrays=copied,
        events=events,
        metadata=metadata,
    )
