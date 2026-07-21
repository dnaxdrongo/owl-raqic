"""Constants and field-name registries for Observer-Window Life.

These names are used by movement, death, copy, and invariant code to keep all
observer-window state fields synchronized. Environment fields are deliberately
separate: food, toxin, obstacle, and noise belong to the world, not to a moving
cell identity.
"""

from __future__ import annotations

import numpy as np

CELL_FIELDS_2D: tuple[str, ...] = (
    "activation",
    "memory",
    "phase",
    "threshold",
    "integration",
    "resource",
    "health",
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
    "emit_efficiency",
    "receive_sensitivity",
    "signal_precision",
    "honesty_bias",
    "deception_bias",
)

CELL_FIELDS_3D: tuple[str, ...] = (
    "possibility",
    "channel_receptivity",
    "channel_emission_bias",
    "channel_trust_local",
    "signal_memory",
)

COMMUNICATION_FIELDS_3D: tuple[str, ...] = (
    "signal",
    "signal_emission",
    "signal_reception",
    "signal_memory",
    "channel_receptivity",
    "channel_emission_bias",
    "channel_trust_local",
)

ENVIRONMENT_FIELDS: tuple[str, ...] = ("food", "toxin", "signal", "obstacle", "noise")

DEFAULT_FLOAT_DTYPE = np.float32
DEFAULT_INT_DTYPE = np.int32
DEFAULT_READOUT_DTYPE = np.int16
DEFAULT_BOOL_DTYPE = np.bool_

BOUNDED_CELL_FIELDS: tuple[str, ...] = (
    "activation",
    "memory",
    "integration",
    "resource",
    "health",
    "boundary",
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
    "emit_efficiency",
    "receive_sensitivity",
    "signal_precision",
    "honesty_bias",
    "deception_bias",
)
