from __future__ import annotations

from typing import Any

import numpy as np

from owl.core.actions import Action
from owl.viz.sprite_state import (
    SpriteDescriptor,
    SpriteState,
    SpriteStatus,
    build_sprite_state,
    descriptor_from_traits,
)
from owl.viz.trait_color import TraitVector
from owl.viz.visual_snapshot import VisualSnapshot, snapshot_from_world_state


def probability_entropy(probabilities: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.asarray(probabilities, dtype=float)
    return -np.sum(np.where(p > 0, p * np.log(np.maximum(p, eps)), 0.0), axis=-1)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _field(snapshot: VisualSnapshot, name: str, default: Any = 0.0) -> np.ndarray:
    value = snapshot.arrays.get(name)
    if value is None:
        return np.full(snapshot.world_shape, default)
    return np.asarray(value)


def _probability_fields(snapshot: VisualSnapshot) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    readout = snapshot.arrays.get("raqic_readout", snapshot.arrays.get("readout"))
    if readout is None:
        readout = np.zeros(snapshot.world_shape, dtype=np.int16)
    probabilities = snapshot.arrays.get("raqic_probabilities")
    if probabilities is None:
        possibilities = snapshot.arrays.get("possibility")
        if possibilities is None:
            probabilities = np.eye(len(Action), dtype=np.float32)[np.asarray(readout, dtype=int)]
        else:
            probabilities = possibilities
    probs = np.asarray(probabilities)
    return np.asarray(readout), np.max(probs, axis=-1), probability_entropy(probs)


def _normalized_field(snapshot: VisualSnapshot, name: str) -> np.ndarray:
    values = np.asarray(_field(snapshot, name, 0.0), dtype=np.float32)
    return np.clip(np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def _trait_group_arrays(snapshot: VisualSnapshot) -> tuple[np.ndarray, ...]:
    aggression = _normalized_field(snapshot, "aggression")
    predation = _normalized_field(snapshot, "predation")
    metabolism = _normalized_field(snapshot, "metabolism")
    reproduction = _normalized_field(snapshot, "reproduction_rate")
    cooperation = _normalized_field(snapshot, "cooperation")
    grazing = _normalized_field(snapshot, "grazing")
    toxin_resistance = _normalized_field(snapshot, "toxin_resistance")
    boundary = _normalized_field(snapshot, "boundary")
    curiosity = _normalized_field(snapshot, "curiosity")
    memory_capacity = _normalized_field(snapshot, "memory_capacity")
    mobility = _normalized_field(snapshot, "mobility")
    coupling = _normalized_field(snapshot, "coupling_strength")
    emission = _normalized_field(snapshot, "emit_strength")
    signal_precision = _normalized_field(snapshot, "signal_precision")
    return (
        (aggression + predation) * np.float32(0.5),
        (metabolism + reproduction) * np.float32(0.5),
        (cooperation + grazing) * np.float32(0.5),
        (toxin_resistance + boundary) * np.float32(0.5),
        (curiosity + memory_capacity + mobility) / np.float32(3.0),
        (coupling + emission + signal_precision) / np.float32(3.0),
    )


def _visible_coordinates(
    snapshot: VisualSnapshot,
    visible_mask: np.ndarray | None,
    max_cells: int | None,
) -> np.ndarray:
    health = np.asarray(_field(snapshot, "health", 0.0), dtype=np.float32)
    obstacle = np.asarray(_field(snapshot, "obstacle", False), dtype=bool)
    mask = (health > 0) & (~obstacle)
    if visible_mask is not None:
        mask &= np.asarray(visible_mask, dtype=bool)
    coords = np.argwhere(mask)
    if max_cells is not None:
        coords = coords[: int(max_cells)]
    return coords


def _descriptor_status_arrays(snapshot: VisualSnapshot) -> dict[str, np.ndarray]:
    health = np.asarray(_field(snapshot, "health", 0.0), dtype=np.float32)
    resource = np.asarray(_field(snapshot, "resource", 0.0), dtype=np.float32)
    probabilities = _probability_fields(snapshot)[1:]
    confidence, entropy = probabilities
    age = np.asarray(_field(snapshot, "age", 0.0), dtype=np.float32)
    parent = snapshot.arrays.get("raqic_parent_intention")
    parent_pressure = (
        np.max(np.asarray(parent), axis=-1)
        if parent is not None
        else np.zeros_like(health, dtype=np.float32)
    )
    living_resource = resource[health > 0]
    threshold = float(np.quantile(living_resource, 0.75)) if living_resource.size else 1.0
    return {
        "health": health,
        "resource": resource,
        "toxin": np.asarray(_field(snapshot, "toxin", 0.0), dtype=np.float32),
        "starvation": np.asarray(_field(snapshot, "starvation_debt", 0.0), dtype=np.float32),
        "integration": np.asarray(_field(snapshot, "noetic_C", 0.0), dtype=np.float32),
        "phase": np.asarray(_field(snapshot, "phase", 0.0), dtype=np.float32),
        "age": age,
        "parent_pressure": np.asarray(parent_pressure, dtype=np.float32),
        "confidence": np.asarray(confidence, dtype=np.float32),
        "entropy": np.asarray(entropy, dtype=np.float32),
        "max_age": np.asarray(max(float(np.max(age)), 1.0), dtype=np.float32),
        "resource_threshold": np.asarray(threshold, dtype=np.float32),
    }


def sprite_descriptors_from_snapshot(
    snapshot: VisualSnapshot,
    visible_mask: np.ndarray,
    selected_id: int | None = None,
) -> tuple[SpriteDescriptor, ...]:
    del selected_id
    coords = np.argwhere(np.asarray(visible_mask, dtype=bool))
    occupancy = np.asarray(_field(snapshot, "occupancy", -1))
    lineage = np.asarray(_field(snapshot, "lineage_id", -1))
    stage = np.asarray(_field(snapshot, "development_stage", 0.0), dtype=np.float32)
    trait_groups = _trait_group_arrays(snapshot)
    descriptors: list[SpriteDescriptor] = []
    for y_value, x_value in coords:
        y, x = int(y_value), int(x_value)
        traits = TraitVector(*(float(values[y, x]) for values in trait_groups))
        descriptors.append(
            descriptor_from_traits(
                ow_id=int(occupancy[y, x]),
                traits=traits,
                developmental_stage=int(round(float(stage[y, x]) * 3.0)),
                lineage_marker=int(lineage[y, x]),
            )
        )
    return tuple(descriptors)


def sprite_statuses_from_snapshot(
    snapshot: VisualSnapshot,
    visible_mask: np.ndarray,
    selected_id: int | None = None,
) -> tuple[SpriteStatus, ...]:
    coords = np.argwhere(np.asarray(visible_mask, dtype=bool))
    arrays = _descriptor_status_arrays(snapshot)
    occupancy = np.asarray(_field(snapshot, "occupancy", -1))
    max_age = float(arrays["max_age"])
    threshold = float(arrays["resource_threshold"])
    statuses: list[SpriteStatus] = []
    for y_value, x_value in coords:
        y, x = int(y_value), int(x_value)
        statuses.append(
            SpriteStatus(
                health_fraction=_clamp01(arrays["health"][y, x]),
                resource_fraction=_clamp01(arrays["resource"][y, x]),
                toxin_fraction=_clamp01(arrays["toxin"][y, x]),
                starvation_fraction=_clamp01(arrays["starvation"][y, x]),
                integration=_clamp01(arrays["integration"][y, x]),
                phase=float(arrays["phase"][y, x]) % (2.0 * np.pi),
                confidence=_clamp01(arrays["confidence"][y, x]),
                entropy=max(0.0, float(arrays["entropy"][y, x])),
                selected=selected_id is not None and int(occupancy[y, x]) == int(selected_id),
                parent_pressure=_clamp01(arrays["parent_pressure"][y, x]),
                age_fraction=_clamp01(arrays["age"][y, x] / max_age),
                reproduction_ready=bool(
                    arrays["resource"][y, x] >= threshold and arrays["health"][y, x] > 0.5
                ),
            )
        )
    return tuple(statuses)


def sprite_states_from_snapshot(
    snapshot: VisualSnapshot,
    *,
    visible_mask: np.ndarray | None = None,
    max_cells: int | None = None,
    selected_id: int | None = None,
) -> tuple[tuple[tuple[int, int], SpriteState], ...]:
    coords = _visible_coordinates(snapshot, visible_mask, max_cells)
    occupancy = np.asarray(_field(snapshot, "occupancy", -1))
    lineage = np.asarray(_field(snapshot, "lineage_id", -1))
    stage = np.asarray(_field(snapshot, "development_stage", 0.0), dtype=np.float32)
    trait_groups = _trait_group_arrays(snapshot)
    arrays = _descriptor_status_arrays(snapshot)
    readout, _, _ = _probability_fields(snapshot)
    signal = snapshot.arrays.get("signal_emission")
    signal_values = None if signal is None else np.asarray(signal)
    max_age = float(arrays["max_age"])
    threshold = float(arrays["resource_threshold"])
    states: list[tuple[tuple[int, int], SpriteState]] = []
    for y_value, x_value in coords:
        y, x = int(y_value), int(x_value)
        traits = TraitVector(*(float(values[y, x]) for values in trait_groups))
        descriptor = descriptor_from_traits(
            ow_id=int(occupancy[y, x]),
            traits=traits,
            developmental_stage=int(round(float(stage[y, x]) * 3.0)),
            lineage_marker=int(lineage[y, x]),
        )
        status = SpriteStatus(
            health_fraction=_clamp01(arrays["health"][y, x]),
            resource_fraction=_clamp01(arrays["resource"][y, x]),
            toxin_fraction=_clamp01(arrays["toxin"][y, x]),
            starvation_fraction=_clamp01(arrays["starvation"][y, x]),
            integration=_clamp01(arrays["integration"][y, x]),
            phase=float(arrays["phase"][y, x]) % (2.0 * np.pi),
            confidence=_clamp01(arrays["confidence"][y, x]),
            entropy=max(0.0, float(arrays["entropy"][y, x])),
            selected=selected_id is not None and int(occupancy[y, x]) == int(selected_id),
            parent_pressure=_clamp01(arrays["parent_pressure"][y, x]),
            age_fraction=_clamp01(arrays["age"][y, x] / max_age),
            reproduction_ready=bool(
                arrays["resource"][y, x] >= threshold and arrays["health"][y, x] > 0.5
            ),
        )
        action_value = int(readout[y, x])
        try:
            action = Action(action_value)
            invalid = False
        except ValueError:
            action = Action.REST
            invalid = True
        channel = -1
        if signal_values is not None and action == Action.COMMUNICATE:
            channel = int(np.argmax(signal_values[y, x]))
        states.append(
            (
                (y, x),
                SpriteState(
                    descriptor=descriptor,
                    status=status,
                    action=action,
                    communication_channel=channel,
                    debug_marker="UNKNOWN_ACTION" if invalid else "",
                    invalid_action=invalid,
                ),
            )
        )
    return tuple(states)


def sprite_states_from_state(
    state: Any,
    *,
    max_cells: int | None = None,
    selected_id: int | None = None,
) -> tuple[tuple[tuple[int, int], SpriteState], ...]:
    snapshot = snapshot_from_world_state(state)
    return sprite_states_from_snapshot(
        snapshot,
        max_cells=max_cells,
        selected_id=selected_id,
    )


__all__ = [
    "build_sprite_state",
    "probability_entropy",
    "sprite_descriptors_from_snapshot",
    "sprite_states_from_snapshot",
    "sprite_states_from_state",
    "sprite_statuses_from_snapshot",
]
