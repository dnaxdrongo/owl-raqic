"""Snapshot save/load interfaces.

Snapshots are compact single-tick NumPy ``.npz`` files containing all dense
state arrays plus JSON metadata for sparse events/mobile OW records. They are
intended for debugging and reproducible analysis, not as a replacement for
chunked time-series recording.
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any, cast

import numpy as np

from owl.core.state import EventRecord, GlobalState, OWRecord, PatchState, WorldState

_PATCH_ARRAY_FIELDS = tuple(field.name for field in fields(PatchState))
_GLOBAL_ARRAY_FIELDS = ("signal_pressure", "policy_bias")


def _world_array_field_names(state: WorldState) -> tuple[str, ...]:
    """Return dense array field names from ``WorldState`` excluding nested state."""
    names: list[str] = []
    for field in fields(WorldState):
        name = field.name
        if name in {"patches", "global_state", "event_queue", "mobile_ows", "tick"}:
            continue
        if isinstance(getattr(state, name), np.ndarray):
            names.append(name)
    return tuple(names)


def _event_to_json(event: EventRecord) -> dict[str, Any]:
    """Convert an event record to JSON-friendly data."""
    return {
        "kind": str(event.kind),
        "tick": int(event.tick),
        "source": None if event.source is None else [int(event.source[0]), int(event.source[1])],
        "target": None if event.target is None else [int(event.target[0]), int(event.target[1])],
        "payload": event.payload,
    }


def _event_from_json(data: dict[str, Any]) -> EventRecord:
    """Reconstruct an ``EventRecord`` from JSON data."""
    source = data.get("source")
    target = data.get("target")
    return EventRecord(
        kind=str(data.get("kind", "")),
        tick=int(data.get("tick", 0)),
        source=None if source is None else (int(source[0]), int(source[1])),
        target=None if target is None else (int(target[0]), int(target[1])),
        payload=dict(data.get("payload", {})),
    )


def _ow_to_json(record: OWRecord) -> dict[str, Any]:
    """Convert a mobile OW sparse record to JSON-friendly data."""
    return {
        "id": int(record.id),
        "type_id": int(record.type_id),
        "pos_y": int(record.pos_y),
        "pos_x": int(record.pos_x),
        "occupied_cells": [[int(y), int(x)] for y, x in record.occupied_cells],
        "parent_id": None if record.parent_id is None else int(record.parent_id),
        "children": [int(child) for child in record.children],
        "traits": np.asarray(record.traits, dtype=np.float32).tolist(),
        "alive": bool(record.alive),
        "genome": None
        if record.genome is None
        else np.asarray(record.genome, dtype=np.float32).tolist(),
        "resource": float(getattr(record, "resource", 0.0)),
        "health": float(getattr(record, "health", 1.0)),
        "boundary": float(getattr(record, "boundary", 1.0)),
    }


def _ow_from_json(data: dict[str, Any]) -> OWRecord:
    """Reconstruct an ``OWRecord`` from JSON data."""
    return OWRecord(
        id=int(data["id"]),
        type_id=int(data.get("type_id", 0)),
        pos_y=int(data.get("pos_y", 0)),
        pos_x=int(data.get("pos_x", 0)),
        occupied_cells=[(int(y), int(x)) for y, x in data.get("occupied_cells", [])],
        parent_id=None if data.get("parent_id") is None else int(data["parent_id"]),
        children=[int(child) for child in data.get("children", [])],
        traits=np.asarray(data.get("traits", []), dtype=np.float32),
        alive=bool(data.get("alive", False)),
    )


def save_snapshot(state: WorldState, path: str) -> None:
    """Save a compact single-tick state snapshot.

    Parameters
    ----------
    state:
        Runtime dense state to serialize. This function does not mutate state.
    path:
        Destination ``.npz`` file path. Parent directories are created
        automatically. Existing files are overwritten by NumPy.
    """
    out_path = Path(path)
    if out_path.exists() and out_path.is_dir():
        raise ValueError(f"snapshot path points to a directory: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {}
    world_fields = _world_array_field_names(state)
    for name in world_fields:
        arrays[f"world/{name}"] = np.asarray(getattr(state, name))

    patch_fields: list[str] = []
    for name in _PATCH_ARRAY_FIELDS:
        value = getattr(state.patches, name)
        if isinstance(value, np.ndarray):
            arrays[f"patches/{name}"] = np.asarray(value)
            patch_fields.append(name)

    for name in _GLOBAL_ARRAY_FIELDS:
        arrays[f"global/{name}"] = np.asarray(getattr(state.global_state, name))

    metadata = {
        "format": "observer-window-life-snapshot-v1",
        "tick": int(state.tick),
        "next_ow_id": int(getattr(state, "next_ow_id", 1)),
        "global_crisis": float(getattr(state, "global_crisis", 0.0)),
        "global_carrying_pressure": float(getattr(state, "global_carrying_pressure", 0.0)),
        "world_fields": list(world_fields),
        "patch_fields": patch_fields,
        "global_scalar_fields": {
            "integration": float(state.global_state.integration),
            "readout": int(state.global_state.readout),
            "intention": int(state.global_state.intention),
            "fragmentation": float(state.global_state.fragmentation),
            "diversity": float(state.global_state.diversity),
            "complexity": float(state.global_state.complexity),
            "crisis": float(getattr(state.global_state, "crisis", 0.0)),
            "carrying_pressure": float(getattr(state.global_state, "carrying_pressure", 0.0)),
            "starvation_pressure": float(getattr(state.global_state, "starvation_pressure", 0.0)),
            "food_deficit": float(getattr(state.global_state, "food_deficit", 0.0)),
        },
        "event_queue": [_event_to_json(event) for event in state.event_queue],
        "mobile_ows": [_ow_to_json(record) for record in state.mobile_ows.values()],
    }
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_)

    np.savez_compressed(out_path, **cast(dict[str, Any], arrays))


def load_snapshot(path: str) -> WorldState:
    """Load a compact single-tick state snapshot.

    Parameters
    ----------
    path:
        Snapshot path written by :func:`save_snapshot`.

    Returns
    -------
    WorldState
        Reconstructed state with dense arrays, patch/global summaries, sparse
        event queue, mobile OW records, and tick value restored.
    """
    in_path = Path(path)
    if not in_path.exists():
        raise FileNotFoundError(f"snapshot not found: {in_path}")
    if in_path.is_dir():
        raise ValueError(f"snapshot path points to a directory: {in_path}")

    with np.load(in_path, allow_pickle=False) as data:
        if "metadata_json" not in data.files:
            raise ValueError(f"snapshot {in_path} is missing metadata_json")
        metadata = json.loads(str(data["metadata_json"].item()))
        if metadata.get("format") != "observer-window-life-snapshot-v1":
            raise ValueError(f"unsupported snapshot format: {metadata.get('format')!r}")

        world_kwargs: dict[str, Any] = {}
        for name in metadata["world_fields"]:
            key = f"world/{name}"
            if key not in data.files:
                raise ValueError(f"snapshot missing array {key!r}")
            world_kwargs[name] = np.array(data[key], copy=True)

        patch_kwargs: dict[str, Any] = {}
        for name in metadata["patch_fields"]:
            key = f"patches/{name}"
            if key not in data.files:
                raise ValueError(f"snapshot missing array {key!r}")
            patch_kwargs[name] = np.array(data[key], copy=True)

        global_arrays: dict[str, Any] = {}
        for name in _GLOBAL_ARRAY_FIELDS:
            key = f"global/{name}"
            if key not in data.files:
                raise ValueError(f"snapshot missing array {key!r}")
            global_arrays[name] = np.array(data[key], copy=True)

    global_scalars = metadata["global_scalar_fields"]
    patches = PatchState(**patch_kwargs)
    global_state = GlobalState(
        integration=float(global_scalars["integration"]),
        readout=int(global_scalars["readout"]),
        intention=int(global_scalars["intention"]),
        fragmentation=float(global_scalars["fragmentation"]),
        diversity=float(global_scalars["diversity"]),
        complexity=float(global_scalars["complexity"]),
        signal_pressure=global_arrays["signal_pressure"],
        policy_bias=global_arrays["policy_bias"],
        crisis=float(global_scalars.get("crisis", 0.0)),
        carrying_pressure=float(global_scalars.get("carrying_pressure", 0.0)),
        starvation_pressure=float(global_scalars.get("starvation_pressure", 0.0)),
        food_deficit=float(global_scalars.get("food_deficit", 0.0)),
    )

    world_kwargs.setdefault("next_ow_id", int(metadata.get("next_ow_id", 1)))
    world_kwargs.setdefault(
        "global_crisis",
        float(metadata.get("global_crisis", global_scalars.get("crisis", 0.0))),
    )
    world_kwargs.setdefault(
        "global_carrying_pressure",
        float(
            metadata.get("global_carrying_pressure", global_scalars.get("carrying_pressure", 0.0))
        ),
    )
    state = WorldState(
        **world_kwargs,
        patches=patches,
        global_state=global_state,
        event_queue=[_event_from_json(item) for item in metadata.get("event_queue", [])],
        mobile_ows={
            int(item["id"]): _ow_from_json(item) for item in metadata.get("mobile_ows", [])
        },
        tick=int(metadata["tick"]),
    )
    return state
