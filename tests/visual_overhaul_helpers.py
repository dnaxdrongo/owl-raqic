from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from owl.core.actions import Action
from owl.viz.visual_snapshot import VisualSnapshot, snapshot_from_world_state


def synthetic_world(*, shape: tuple[int, int] = (6, 7), tick: int = 3) -> Any:
    height, width = shape
    health = np.zeros(shape, dtype=np.float32)
    health[1:5, 1:6] = 0.85
    obstacle = np.zeros(shape, dtype=bool)
    obstacle[0, 0] = True
    occupancy = np.full(shape, -1, dtype=np.int64)
    coords = np.argwhere(health > 0)
    for index, (y, x) in enumerate(coords, start=100):
        occupancy[y, x] = index
    readout = np.full(shape, int(Action.REST), dtype=np.int16)
    actions = list(Action)
    for index, (y, x) in enumerate(coords):
        readout[y, x] = int(actions[index % len(actions)])
    probabilities = np.zeros((*shape, len(Action)), dtype=np.float32)
    probabilities[..., 0] = 1.0
    for y, x in coords:
        probabilities[y, x] = 0.01
        probabilities[y, x, int(readout[y, x])] = 0.79
        probabilities[y, x] /= probabilities[y, x].sum()

    y_axis = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    x_axis = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    base = np.clip(0.25 + 0.45 * y_axis + 0.25 * x_axis, 0.0, 1.0).astype(np.float32)
    resource = np.where(health > 0, np.clip(base, 0.0, 1.0), 0.0).astype(np.float32)
    food = np.zeros(shape, dtype=np.float32)
    food[0, 2] = 0.9
    food[5, 5] = 0.7
    toxin = np.zeros(shape, dtype=np.float32)
    toxin[2, 6] = 0.8
    waste = np.zeros(shape, dtype=np.float32)
    waste[5, 1] = 0.65

    fields: dict[str, Any] = {
        "tick": tick,
        "health": health,
        "resource": resource,
        "toxin": toxin,
        "food": food,
        "waste": waste,
        "obstacle": obstacle,
        "occupancy": occupancy,
        "readout": readout,
        "raqic_readout": readout.copy(),
        "raqic_probabilities": probabilities,
        "raqic_record_confidence": np.max(probabilities, axis=-1),
        "integration": np.where(health > 0, base, 0.0).astype(np.float32),
        "noetic_C": np.where(health > 0, base, 0.0).astype(np.float32),
        "phase": (base * np.float32(2.0 * np.pi)).astype(np.float32),
        "boundary": np.where(health > 0, 0.55, 0.0).astype(np.float32),
        "age": np.where(health > 0, 0.3 + base * 10.0, 0.0).astype(np.float32),
        "ow_type": np.where(health > 0, 1, 0).astype(np.int16),
        "lineage_id": np.where(health > 0, occupancy // 4, -1).astype(np.int64),
        "parent_id": np.where(health > 0, occupancy - 1, -1).astype(np.int64),
        "development_stage": np.where(health > 0, base, 0.0).astype(np.float32),
        "starvation_debt": np.where(health > 0, 1.0 - resource, 0.0).astype(np.float32),
        "signal_emission": np.zeros((*shape, 8), dtype=np.float32),
        "raqic_parent_intention": probabilities.copy(),
    }
    trait_names = (
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
    )
    for index, name in enumerate(trait_names):
        value = np.mod(base + index * 0.071, 1.0).astype(np.float32)
        fields[name] = np.where(health > 0, value, 0.0).astype(np.float32)
    communicate = readout == int(Action.COMMUNICATE)
    fields["signal_emission"][communicate, 3] = 1.0
    return SimpleNamespace(**fields)


def synthetic_snapshot(*, shape: tuple[int, int] = (6, 7), tick: int = 3) -> VisualSnapshot:
    return snapshot_from_world_state(synthetic_world(shape=shape, tick=tick))
