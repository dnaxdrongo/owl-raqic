"""World initialization utilities.

 creates deterministic, bounded :class:`WorldState` instances from
validated :class:`SimulationConfig` objects. No runtime update rules are
implemented here; this module only allocates arrays and establishes clean
initial conditions for later engine passes.
"""

from __future__ import annotations

import numpy as np

from owl.core.actions import Action
from owl.core.advanced import ensure_advanced_fields
from owl.core.config import SimulationConfig
from owl.core.constants import (
    DEFAULT_BOOL_DTYPE,
    DEFAULT_FLOAT_DTYPE,
    DEFAULT_INT_DTYPE,
    DEFAULT_READOUT_DTYPE,
)
from owl.core.state import (
    GlobalState,
    PatchState,
    WorldState,
    action_shape,
    channel_shape,
    field_shape,
)
from owl.core.traits import default_trait_presets, initialize_traits


def _zeros(shape: tuple[int, ...]) -> np.ndarray:
    """Return a zero float32 array with ``shape``."""
    return np.zeros(shape, dtype=DEFAULT_FLOAT_DTYPE)


def _bounded_normal(
    rng: np.random.Generator,
    shape: tuple[int, int],
    mean: float,
    sigma: float,
) -> np.ndarray:
    """Return clipped float32 normal samples in ``[0, 1]``."""
    if sigma == 0:
        return np.full(shape, mean, dtype=DEFAULT_FLOAT_DTYPE)
    return np.clip(rng.normal(mean, sigma, size=shape), 0.0, 1.0).astype(DEFAULT_FLOAT_DTYPE)


def _patch_shape(cfg: SimulationConfig) -> tuple[int, int]:
    """Return patch-grid shape after validating exact tiling."""
    h, w, patch = cfg.world.height, cfg.world.width, cfg.world.patch_size
    if h % patch or w % patch:
        raise ValueError("world.height and world.width must be divisible by world.patch_size")
    return h // patch, w // patch


def create_empty_patch_state(cfg: SimulationConfig) -> PatchState:
    """Create zero-filled patch-level state arrays.

    Returns
    -------
    PatchState
        Patch-level arrays with shape ``(height // patch_size, width //
        patch_size)`` for scalar fields, plus action/channel tensors.
    """
    ph, pw = _patch_shape(cfg)
    actions = len(Action)
    channels = cfg.communication.num_channels
    possibility = np.full((ph, pw, actions), 1.0 / actions, dtype=DEFAULT_FLOAT_DTYPE)

    return PatchState(
        activation=_zeros((ph, pw)),
        memory=_zeros((ph, pw)),
        phase=_zeros((ph, pw)),
        possibility=possibility,
        integration=_zeros((ph, pw)),
        resource=_zeros((ph, pw)),
        health=_zeros((ph, pw)),
        boundary=_zeros((ph, pw)),
        signal_pressure=_zeros((ph, pw, channels)),
        synchrony=_zeros((ph, pw)),
        coherence=_zeros((ph, pw)),
        cross_scale=_zeros((ph, pw)),
        intention=np.zeros((ph, pw), dtype=DEFAULT_INT_DTYPE),
        policy_bias=_zeros((ph, pw, actions)),
    )


def create_empty_global_state(cfg: SimulationConfig) -> GlobalState:
    """Create a neutral global/apex observer-window summary."""
    return GlobalState(
        integration=0.0,
        readout=int(Action.REST),
        intention=0,
        fragmentation=0.0,
        diversity=0.0,
        complexity=0.0,
        signal_pressure=np.zeros((cfg.communication.num_channels,), dtype=DEFAULT_FLOAT_DTYPE),
        policy_bias=np.zeros((len(Action),), dtype=DEFAULT_FLOAT_DTYPE),
    )


def initialize_world(cfg: SimulationConfig, rng: np.random.Generator | None = None) -> WorldState:
    """Allocate and initialize all dense state arrays.

    Parameters
    ----------
    cfg:
        Validated simulation configuration.
    rng:
        Optional explicit NumPy random generator. If omitted, a generator is
        created from ``cfg.world.seed``.

    Returns
    -------
    WorldState
        Fully allocated, bounded, deterministic world state.
    """
    rng = np.random.default_rng(cfg.world.seed) if rng is None else rng
    h, w = cfg.world.height, cfg.world.width
    actions = len(Action)
    channels = cfg.communication.num_channels
    cell_shape = (h, w)
    action_shape_ = (h, w, actions)
    channel_shape_ = (h, w, channels)

    patches = create_empty_patch_state(cfg)
    global_state = create_empty_global_state(cfg)

    state = WorldState(
        activation=_zeros(cell_shape),
        memory=_zeros(cell_shape),
        phase=_zeros(cell_shape),
        threshold=np.full(
            cell_shape, cfg.initialization.initial_threshold_mean, dtype=DEFAULT_FLOAT_DTYPE
        ),
        readout=np.full(cell_shape, int(Action.REST), dtype=DEFAULT_READOUT_DTYPE),
        integration=_zeros(cell_shape),
        resource=_zeros(cell_shape),
        health=_zeros(cell_shape),
        boundary=_zeros(cell_shape),
        age=np.zeros(cell_shape, dtype=DEFAULT_INT_DTYPE),
        ow_type=np.zeros(cell_shape, dtype=DEFAULT_INT_DTYPE),
        lineage_id=np.full(cell_shape, -1, dtype=DEFAULT_INT_DTYPE),
        parent_id=np.full(cell_shape, -1, dtype=DEFAULT_INT_DTYPE),
        possibility=np.zeros(action_shape_, dtype=DEFAULT_FLOAT_DTYPE),
        signal=_zeros(channel_shape_),
        signal_emission=_zeros(channel_shape_),
        signal_reception=_zeros(channel_shape_),
        signal_memory=_zeros(channel_shape_),
        channel_receptivity=_zeros(channel_shape_),
        channel_emission_bias=_zeros(channel_shape_),
        channel_trust_local=_zeros(channel_shape_),
        food=_zeros(cell_shape),
        toxin=_zeros(cell_shape),
        obstacle=np.zeros(cell_shape, dtype=DEFAULT_BOOL_DTYPE),
        occupancy=np.full(cell_shape, -1, dtype=DEFAULT_INT_DTYPE),
        noise=_zeros(cell_shape),
        mobility=_zeros(cell_shape),
        metabolism=_zeros(cell_shape),
        predation=_zeros(cell_shape),
        grazing=_zeros(cell_shape),
        cooperation=_zeros(cell_shape),
        aggression=_zeros(cell_shape),
        curiosity=_zeros(cell_shape),
        reproduction_rate=_zeros(cell_shape),
        toxin_resistance=_zeros(cell_shape),
        memory_capacity=_zeros(cell_shape),
        coupling_strength=_zeros(cell_shape),
        emit_strength=_zeros(cell_shape),
        emit_efficiency=_zeros(cell_shape),
        receive_sensitivity=_zeros(cell_shape),
        signal_precision=_zeros(cell_shape),
        honesty_bias=_zeros(cell_shape),
        deception_bias=_zeros(cell_shape),
        patches=patches,
        global_state=global_state,
    )

    initialize_food_patches(state, cfg, rng)
    initialize_population(state, cfg, rng)
    initialize_traits(state, cfg, rng)
    initialize_possibilities(state, cfg)
    ensure_advanced_fields(state, cfg)
    _initialize_advanced_genomes(state, cfg, rng)

    _validate_initialized_state(state, cfg)
    return state


def initialize_food_patches(
    state: WorldState, cfg: SimulationConfig, rng: np.random.Generator
) -> None:
    """Initialize environmental food and toxin fields.

    Mutates ``state.food``, ``state.toxin``, ``state.obstacle``, and
    ``state.noise``. Food/toxin patches use toroidal radial falloff so no edge
    is privileged in the baseline world.
    """
    h, w = field_shape(state)
    icfg = cfg.initialization

    state.food.fill(icfg.background_food)
    state.toxin.fill(0.0)
    state.obstacle.fill(False)

    yy, xx = np.indices((h, w))

    def add_patches(target: np.ndarray, count: int, radius: int, intensity: float) -> None:
        if count <= 0 or intensity <= 0:
            return
        for _ in range(count):
            cy = int(rng.integers(0, h))
            cx = int(rng.integers(0, w))
            dy = np.minimum(np.abs(yy - cy), h - np.abs(yy - cy))
            dx = np.minimum(np.abs(xx - cx), w - np.abs(xx - cx))
            distance = np.sqrt((dy.astype(np.float32) ** 2) + (dx.astype(np.float32) ** 2))
            falloff = np.clip(1.0 - distance / max(radius, 1), 0.0, 1.0)
            target += intensity * falloff.astype(DEFAULT_FLOAT_DTYPE)

    add_patches(
        state.food, icfg.food_patch_count, icfg.food_patch_radius, icfg.food_patch_intensity
    )
    add_patches(
        state.toxin, icfg.toxin_patch_count, icfg.toxin_patch_radius, icfg.toxin_patch_intensity
    )

    if icfg.obstacle_density > 0:
        state.obstacle[...] = rng.random((h, w)) < icfg.obstacle_density

    if icfg.initial_activation_sigma > 0:
        state.noise[...] = rng.normal(0.0, icfg.initial_activation_sigma, size=(h, w)).astype(
            DEFAULT_FLOAT_DTYPE
        )

    np.clip(state.food, 0.0, 1.0, out=state.food)
    np.clip(state.toxin, 0.0, 1.0, out=state.toxin)


def initialize_population(
    state: WorldState, cfg: SimulationConfig, rng: np.random.Generator
) -> None:
    """Initialize living cells, physical state, readouts, IDs, and types.

    Mutates cell-level physical and identity arrays. The initial state is
    deterministic when the supplied ``rng`` is deterministic.
    """
    h, w = field_shape(state)
    icfg = cfg.initialization
    living = rng.random((h, w)) < icfg.population_density
    living &= ~state.obstacle

    if icfg.population_density > 0.0 and not living.any():
        available = np.argwhere(~state.obstacle)
        if available.size == 0:
            raise ValueError("cannot initialize population: all cells are obstacles")
        y, x = available[int(rng.integers(0, len(available)))]
        living[int(y), int(x)] = True

    state.health.fill(0.0)
    state.resource.fill(0.0)
    state.boundary.fill(0.0)
    state.activation.fill(0.0)
    state.memory.fill(0.0)
    state.integration.fill(0.0)
    state.threshold.fill(0.0)
    state.phase.fill(0.0)
    state.age.fill(0)
    state.readout.fill(int(Action.REST))
    state.lineage_id.fill(-1)
    state.occupancy.fill(-1)

    state.activation[living] = _bounded_normal(
        rng, (h, w), icfg.initial_activation_mean, icfg.initial_activation_sigma
    )[living]
    state.memory[living] = _bounded_normal(
        rng, (h, w), icfg.initial_memory_mean, icfg.initial_memory_sigma
    )[living]
    state.integration[living] = _bounded_normal(
        rng, (h, w), icfg.initial_integration_mean, icfg.initial_integration_sigma
    )[living]
    state.resource[living] = _bounded_normal(
        rng, (h, w), icfg.initial_resource_mean, icfg.initial_resource_sigma
    )[living]
    state.health[living] = _bounded_normal(
        rng, (h, w), icfg.initial_health_mean, icfg.initial_health_sigma
    )[living]
    state.boundary[living] = _bounded_normal(
        rng, (h, w), icfg.initial_boundary_mean, icfg.initial_boundary_sigma
    )[living]
    state.threshold[living] = _bounded_normal(
        rng, (h, w), icfg.initial_threshold_mean, icfg.initial_threshold_sigma
    )[living]
    state.phase[living] = rng.uniform(0.0, 2.0 * np.pi, size=living.sum()).astype(
        DEFAULT_FLOAT_DTYPE
    )

    names = list(icfg.type_weights)
    presets = default_trait_presets()
    missing = [name for name in names if name not in presets]
    if missing:
        available_names = ", ".join(sorted(presets))
        raise ValueError(
            f"unknown initialization type_weights {missing}; available presets: {available_names}"
        )

    weights = np.asarray([icfg.type_weights[name] for name in names], dtype=np.float64)
    weights = weights / weights.sum()
    type_choices = rng.choice(np.arange(len(names), dtype=np.int32), size=living.sum(), p=weights)
    state.ow_type.fill(0)
    state.ow_type[living] = type_choices.astype(DEFAULT_INT_DTYPE)

    flat_ids = np.arange(h * w, dtype=DEFAULT_INT_DTYPE).reshape(h, w)
    state.lineage_id[living] = flat_ids[living]
    state.occupancy[living] = flat_ids[living]
    live_ids = state.occupancy[state.occupancy >= 0]
    max_id = int(np.max(live_ids)) if live_ids.size else 0
    state.next_ow_id = int(max_id + 1)

    patch_size = cfg.world.patch_size
    patch_w = w // patch_size
    yy, xx = np.indices((h, w))
    parent_id = (yy // patch_size) * patch_w + (xx // patch_size)
    state.parent_id[...] = parent_id.astype(DEFAULT_INT_DTYPE)
    state.parent_id[~living] = -1


def initialize_possibilities(state: WorldState, cfg: SimulationConfig) -> None:
    """Initialize normalized action possibility vectors.

    Mutates ``state.possibility``. Living cells receive a uniform distribution
    over all declared actions. Dead cells receive a one-hot REST distribution so
    they cannot act before later authority masks are implemented.
    """
    _, _, actions = action_shape(state)
    if actions != len(Action):
        raise ValueError(
            f"state.possibility action axis must equal len(Action)={len(Action)}, got {actions}"
        )

    state.possibility.fill(0.0)
    living = state.health > 0.0
    if living.any():
        state.possibility[living, :] = np.float32(1.0 / actions)
    state.possibility[~living, int(Action.REST)] = 1.0


def _initialize_advanced_genomes(
    state: WorldState, cfg: SimulationConfig, rng: np.random.Generator
) -> None:
    """Initialize optional advanced genome and growth arrays for living cells."""
    ensure_advanced_fields(state, cfg)
    living = state.health > 0.0
    if state.genome is not None and living.any():
        # Genome channels are normalized trait factors. Initializing around
        # Existing traits keep optional decoded behavior aligned with the baseline.
        base = np.stack(
            [
                state.mobility,
                state.metabolism,
                state.predation,
                state.grazing,
                state.cooperation,
                state.aggression,
                state.curiosity,
                state.reproduction_rate,
            ],
            axis=-1,
        )
        g = state.genome.shape[-1]
        if g <= base.shape[-1]:
            state.genome[..., :g] = base[..., :g]
        else:
            state.genome[..., : base.shape[-1]] = base
            state.genome[..., base.shape[-1] :] = rng.random(
                (state.health.shape[0], state.health.shape[1], g - base.shape[-1])
            ).astype(np.float32)
        state.genome[~living, :] = 0.0
        np.clip(state.genome, 0.0, 1.0, out=state.genome)
    if state.development_stage is not None:
        state.development_stage[living] = 0.25
        state.development_stage[~living] = 0.0


def _validate_initialized_state(state: WorldState, cfg: SimulationConfig) -> None:
    """Raise ``ValueError`` if initialization produced inconsistent arrays."""
    h, w = field_shape(state)
    ah, aw, actions = action_shape(state)
    ch, cw, channels = channel_shape(state)
    if (ah, aw) != (h, w):
        raise ValueError("possibility spatial shape must match health shape")
    if (ch, cw) != (h, w):
        raise ValueError("signal spatial shape must match health shape")
    if actions != len(Action):
        raise ValueError("possibility action axis must equal len(Action)")
    if channels != cfg.communication.num_channels:
        raise ValueError("signal channel axis must equal cfg.communication.num_channels")

    patch_h, patch_w = _patch_shape(cfg)
    if state.patches.integration.shape != (patch_h, patch_w):
        raise ValueError("patch state shape does not match configured patch tiling")

    for name in (
        "activation",
        "memory",
        "integration",
        "resource",
        "health",
        "boundary",
        "food",
        "toxin",
        "mobility",
        "metabolism",
        "predation",
        "grazing",
        "cooperation",
        "aggression",
        "curiosity",
        "reproduction_rate",
        "toxin_resistance",
        "emit_strength",
        "emit_efficiency",
        "receive_sensitivity",
        "signal_precision",
        "honesty_bias",
        "deception_bias",
    ):
        arr = getattr(state, name)
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values")
        if np.any(arr < 0.0) or np.any(arr > 1.0):
            raise ValueError(f"{name} must be bounded in [0, 1]")

    probs = state.possibility.sum(axis=-1)
    if not np.allclose(probs, 1.0, atol=1e-6):
        raise ValueError("state.possibility must sum to 1 along the action axis")
