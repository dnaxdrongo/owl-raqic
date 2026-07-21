"""Trait initialization and mutation utilities.

Communication is universal in Observer-Window Life: every cell has emission,
reception, trust, and channel-bias traits. Ecological roles are therefore
implemented as continuous trait presets, not as hard-coded behavior classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from owl.core.actions import SignalChannel
from owl.core.config import SimulationConfig
from owl.core.constants import BOUNDED_CELL_FIELDS
from owl.core.state import FloatArray, WorldState

TRAIT_FIELD_NAMES: tuple[str, ...] = (
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


@dataclass(slots=True)
class TraitPreset:
    """Named initial trait bundle for a cell population.

    Scalar fields are bounded in ``[0, 1]``. Channel arrays are optional; when
    omitted, neutral communication vectors are used. The channel vectors are
    not permissions. They only express communication style preferences.
    """

    name: str
    mobility: float
    metabolism: float
    predation: float
    grazing: float
    cooperation: float
    aggression: float
    curiosity: float
    reproduction_rate: float
    toxin_resistance: float
    emit_strength: float
    emit_efficiency: float
    receive_sensitivity: float
    honesty_bias: float
    deception_bias: float
    memory_capacity: float = 0.5
    coupling_strength: float = 0.5
    signal_precision: float = 0.7
    channel_emission_bias: tuple[float, ...] = field(default_factory=tuple)
    channel_receptivity: tuple[float, ...] = field(default_factory=tuple)


def _as_channel_vector(
    values: tuple[float, ...], num_channels: int, fill: float = 0.5
) -> np.ndarray:
    """Return a bounded float32 channel vector of length ``num_channels``."""
    if not values:
        out = np.full(num_channels, fill, dtype=np.float32)
    else:
        raw = np.asarray(values, dtype=np.float32)
        if raw.size < num_channels:
            out = np.full(num_channels, fill, dtype=np.float32)
            out[: raw.size] = raw
        else:
            out = raw[:num_channels].astype(np.float32, copy=True)
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def default_trait_presets() -> dict[str, TraitPreset]:
    """Return default ecological trait presets.

    Returns
    -------
    dict[str, TraitPreset]
        Presets keyed by stable names used in configuration. These roles seed
        trait distributions only. Later engines still use continuous utility and
        authority masks; no preset receives exclusive access to communication.
    """
    food = int(SignalChannel.FOOD)
    danger = int(SignalChannel.DANGER)
    threat = int(SignalChannel.THREAT)
    coordination = int(SignalChannel.COORDINATION)
    distress = int(SignalChannel.DISTRESS)
    territory = int(SignalChannel.TERRITORY)
    integration = int(SignalChannel.INTEGRATION)

    def channels(
        default: float = 0.45, updates: dict[int, float] | None = None
    ) -> tuple[float, ...]:
        values = np.full(8, default, dtype=np.float32)
        updates = {} if updates is None else updates
        for idx, val in updates.items():
            values[int(idx)] = val
        return tuple(float(v) for v in values)

    return {
        "grazer": TraitPreset(
            name="grazer",
            mobility=0.55,
            metabolism=0.45,
            predation=0.05,
            grazing=0.90,
            cooperation=0.45,
            aggression=0.10,
            curiosity=0.35,
            reproduction_rate=0.45,
            toxin_resistance=0.25,
            emit_strength=0.40,
            emit_efficiency=0.70,
            receive_sensitivity=0.65,
            honesty_bias=0.90,
            deception_bias=0.05,
            memory_capacity=0.45,
            coupling_strength=0.45,
            signal_precision=0.75,
            channel_emission_bias=channels(0.35, {food: 0.85, danger: 0.65, coordination: 0.55}),
            channel_receptivity=channels(0.45, {food: 0.80, danger: 0.75, threat: 0.70}),
        ),
        "cooperator": TraitPreset(
            name="cooperator",
            mobility=0.45,
            metabolism=0.50,
            predation=0.03,
            grazing=0.75,
            cooperation=0.85,
            aggression=0.08,
            curiosity=0.50,
            reproduction_rate=0.40,
            toxin_resistance=0.30,
            emit_strength=0.65,
            emit_efficiency=0.70,
            receive_sensitivity=0.80,
            honesty_bias=0.92,
            deception_bias=0.03,
            memory_capacity=0.65,
            coupling_strength=0.75,
            signal_precision=0.85,
            channel_emission_bias=channels(
                0.45,
                {food: 0.70, danger: 0.85, coordination: 0.90, distress: 0.70, integration: 0.75},
            ),
            channel_receptivity=channels(
                0.55, {food: 0.75, danger: 0.90, coordination: 0.90, integration: 0.80}
            ),
        ),
        "proto_carnivore": TraitPreset(
            name="proto_carnivore",
            mobility=0.75,
            metabolism=0.70,
            predation=0.65,
            grazing=0.25,
            cooperation=0.20,
            aggression=0.70,
            curiosity=0.55,
            reproduction_rate=0.30,
            toxin_resistance=0.35,
            emit_strength=0.45,
            emit_efficiency=0.55,
            receive_sensitivity=0.65,
            honesty_bias=0.45,
            deception_bias=0.45,
            memory_capacity=0.55,
            coupling_strength=0.35,
            signal_precision=0.60,
            channel_emission_bias=channels(
                0.35, {food: 0.60, threat: 0.85, distress: 0.45, territory: 0.70}
            ),
            channel_receptivity=channels(
                0.45, {food: 0.60, threat: 0.75, distress: 0.80, territory: 0.65}
            ),
        ),
        "scavenger": TraitPreset(
            name="scavenger",
            mobility=0.50,
            metabolism=0.45,
            predation=0.25,
            grazing=0.55,
            cooperation=0.25,
            aggression=0.30,
            curiosity=0.65,
            reproduction_rate=0.35,
            toxin_resistance=0.55,
            emit_strength=0.35,
            emit_efficiency=0.65,
            receive_sensitivity=0.70,
            honesty_bias=0.70,
            deception_bias=0.20,
            memory_capacity=0.60,
            coupling_strength=0.40,
            signal_precision=0.65,
            channel_emission_bias=channels(0.35, {food: 0.55, danger: 0.45, distress: 0.65}),
            channel_receptivity=channels(0.50, {food: 0.70, danger: 0.65, distress: 0.85}),
        ),
        "explorer": TraitPreset(
            name="explorer",
            mobility=0.80,
            metabolism=0.60,
            predation=0.10,
            grazing=0.60,
            cooperation=0.35,
            aggression=0.20,
            curiosity=0.90,
            reproduction_rate=0.30,
            toxin_resistance=0.40,
            emit_strength=0.35,
            emit_efficiency=0.60,
            receive_sensitivity=0.75,
            honesty_bias=0.80,
            deception_bias=0.10,
            memory_capacity=0.60,
            coupling_strength=0.45,
            signal_precision=0.70,
            channel_emission_bias=channels(0.40, {food: 0.65, danger: 0.70, coordination: 0.50}),
            channel_receptivity=channels(0.55, {food: 0.65, danger: 0.80, coordination: 0.60}),
        ),
    }


def _preset_names_for_config(cfg: SimulationConfig) -> list[str]:
    """Return stable preset names from config weights."""
    presets = default_trait_presets()
    missing = [name for name in cfg.initialization.type_weights if name not in presets]
    if missing:
        available = ", ".join(sorted(presets))
        raise ValueError(
            f"unknown initialization type_weights {missing}; available presets: {available}"
        )
    return list(cfg.initialization.type_weights)


def initialize_traits(state: WorldState, cfg: SimulationConfig, rng: np.random.Generator) -> None:
    """Populate trait arrays from configured population types.

    Mutates scalar trait fields, communication channel-bias fields, and local
    trust/receptivity fields. Dead cells receive zero behavioral traits. Living
    cells receive a preset value plus small clipped Gaussian noise.
    """
    presets = default_trait_presets()
    preset_names = _preset_names_for_config(cfg)
    num_channels = cfg.communication.num_channels
    living = state.health > 0.0
    sigma = cfg.initialization.trait_noise_sigma

    for field_name in TRAIT_FIELD_NAMES:
        getattr(state, field_name).fill(0.0)

    state.channel_receptivity.fill(0.0)
    state.channel_emission_bias.fill(0.0)
    state.channel_trust_local.fill(0.0)

    for type_id, preset_name in enumerate(preset_names):
        preset = presets[preset_name]
        mask = living & (state.ow_type == type_id)

        for field_name in TRAIT_FIELD_NAMES:
            base = float(getattr(preset, field_name))
            values = base + rng.normal(0.0, sigma, size=state.health.shape)
            field = getattr(state, field_name)
            field[mask] = np.clip(values[mask], 0.0, 1.0).astype(np.float32)

        emission = _as_channel_vector(preset.channel_emission_bias, num_channels, fill=0.45)
        reception = _as_channel_vector(preset.channel_receptivity, num_channels, fill=0.50)
        state.channel_emission_bias[mask, :] = emission
        state.channel_receptivity[mask, :] = reception
        state.channel_trust_local[mask, :] = cfg.initialization.initial_trust

    for field_name in BOUNDED_CELL_FIELDS:
        field = getattr(state, field_name)
        np.clip(field, 0.0, 1.0, out=field)

    np.clip(state.channel_receptivity, 0.0, 1.0, out=state.channel_receptivity)
    np.clip(state.channel_emission_bias, 0.0, 1.0, out=state.channel_emission_bias)
    np.clip(state.channel_trust_local, 0.0, 1.0, out=state.channel_trust_local)


def mutate_scalar_trait(value: float, sigma: float, rng: np.random.Generator) -> float:
    """Return a bounded scalar trait after Gaussian mutation.

    Parameters
    ----------
    value:
        Current trait value.
    sigma:
        Nonnegative mutation standard deviation.
    rng:
        Explicit NumPy random generator for deterministic tests.
    """
    if sigma < 0:
        raise ValueError("sigma must be nonnegative")
    return float(np.clip(float(value) + rng.normal(0.0, sigma), 0.0, 1.0))


def mutate_trait_vector(values: FloatArray, sigma: float, rng: np.random.Generator) -> FloatArray:
    """Return a bounded float32 vector after Gaussian mutation."""
    if sigma < 0:
        raise ValueError("sigma must be nonnegative")
    array = np.asarray(values, dtype=np.float32)
    return np.clip(array + rng.normal(0.0, sigma, size=array.shape), 0.0, 1.0).astype(np.float32)


def copy_traits_with_mutation(
    state: WorldState,
    source: tuple[int, int],
    target: tuple[int, int],
    cfg: SimulationConfig,
    rng: np.random.Generator,
) -> None:
    """Copy mutable traits from parent cell to child cell with mutation.

    Mutates only trait fields and communication channel trait fields at
    ``target``. Identity, resource, health, boundary, memory, and lineage are
    handled by later reproduction code.
    """
    sy, sx = source
    ty, tx = target
    h, w = state.health.shape
    if not (0 <= sy < h and 0 <= sx < w and 0 <= ty < h and 0 <= tx < w):
        raise IndexError("source and target coordinates must lie within the cell grid")

    scalar_sigma = cfg.reproduction.mutation_sigma
    channel_sigma = cfg.reproduction.channel_mutation_sigma

    for field_name in TRAIT_FIELD_NAMES:
        field = getattr(state, field_name)
        field[ty, tx] = mutate_scalar_trait(float(field[sy, sx]), scalar_sigma, rng)

    state.channel_emission_bias[ty, tx, :] = mutate_trait_vector(
        state.channel_emission_bias[sy, sx, :],
        channel_sigma,
        rng,
    )
    state.channel_receptivity[ty, tx, :] = mutate_trait_vector(
        state.channel_receptivity[sy, sx, :],
        channel_sigma,
        rng,
    )

    # Trust is inherited conservatively: offspring start close to the parent's
    # priors, with small mutation but no source-specific memory.
    state.channel_trust_local[ty, tx, :] = mutate_trait_vector(
        state.channel_trust_local[sy, sx, :],
        min(channel_sigma, 0.02),
        rng,
    )
