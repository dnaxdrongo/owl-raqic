"""Physical environment updates for Observer-Window Life.

This module implements the passive physical substrate for the simulation:
food/nutrient fields, toxin/damage fields, and multichannel communication
signals. These functions mutate only environment and communication-signal
fields. They do not choose actions, update health, render, record, or aggregate
patches.
"""

from __future__ import annotations

import numpy as np

from owl.core.config import SimulationConfig
from owl.core.state import WorldState
from owl.kernels.numpy_kernels import laplacian_wrap


def _require_field_shapes(state: WorldState, cfg: SimulationConfig) -> tuple[int, int, int]:
    """Validate cell and channel field compatibility for environment updates."""
    height, width = state.health.shape
    expected_2d = (height, width)
    for name in ("food", "toxin", "noise", "obstacle"):
        shape = getattr(state, name).shape
        if shape != expected_2d:
            raise ValueError(f"state.{name} must have shape {expected_2d}, got {shape}")

    expected_channels = (height, width, cfg.communication.num_channels)
    for name in ("signal", "signal_emission", "signal_reception", "signal_memory"):
        shape = getattr(state, name).shape
        if shape != expected_channels:
            raise ValueError(f"state.{name} must have shape {expected_channels}, got {shape}")
    return height, width, cfg.communication.num_channels


def update_environment(state: WorldState, cfg: SimulationConfig) -> None:
    """Update passive environmental fields in place.

    Mutates
    -------
    state.food:
        Adds configured growth, applies toroidal diffusion, applies decay, and
        clips to ``[0, 1]``.
    state.toxin:
        Applies toroidal diffusion and decay, then clips to ``[0, 1]``.
    state.signal:
        Adds current emissions, applies channel-wise diffusion/decay, clips to
        ``[0, 1]``, and clears ``state.signal_emission``.
    environment obstacle cells:
        ``apply_obstacle_mask`` zeros environmental fields at obstacles after
        the updates.
    """
    _require_field_shapes(state, cfg)
    update_food_field(state, cfg)
    update_toxin_field(state, cfg)
    update_signal_fields(state, cfg)
    apply_obstacle_mask(state)


def _base_update_food_field(state: WorldState, cfg: SimulationConfig) -> None:
    """Update the nutrient/food field in place.

    The update is a bounded finite-difference rule:
    growth plus toroidal diffusion minus linear decay. Consumption is not
    handled here; feeding will subtract food in a later engine pass.

    Mutates ``state.food`` and clips it to ``[0, 1]``.
    """
    _height, _width, _channels = _require_field_shapes(state, cfg)
    state.food += np.float32(cfg.resources.food_growth)
    if cfg.resources.food_diffusion:
        state.food += np.float32(cfg.resources.food_diffusion) * laplacian_wrap(state.food).astype(
            state.food.dtype, copy=False
        )
    if cfg.resources.food_decay:
        state.food -= np.float32(cfg.resources.food_decay) * state.food
    np.clip(state.food, 0.0, 1.0, out=state.food)


def update_toxin_field(state: WorldState, cfg: SimulationConfig) -> None:
    """Update the toxin/damage field in place.

    Toxins diffuse toroidally and decay linearly. Source injection is intentionally
    not handled here; initialization or later experiment modules may seed toxin
    patches before this function runs.

    Mutates ``state.toxin`` and clips it to ``[0, 1]``.
    """
    _height, _width, _channels = _require_field_shapes(state, cfg)
    if cfg.resources.toxin_diffusion:
        state.toxin += np.float32(cfg.resources.toxin_diffusion) * laplacian_wrap(
            state.toxin
        ).astype(state.toxin.dtype, copy=False)
    if cfg.resources.toxin_decay:
        state.toxin -= np.float32(cfg.resources.toxin_decay) * state.toxin
    np.clip(state.toxin, 0.0, 1.0, out=state.toxin)


def _base_update_signal_fields(state: WorldState, cfg: SimulationConfig) -> None:
    """Update multichannel communication signal fields in place.

    Each channel obeys a bounded passive field update:

    ``signal_next = signal + diffusion[channel] * laplacian(signal)
                   - decay[channel] * signal + emission[channel]``

    Mutates ``state.signal`` and clears ``state.signal_emission``. If
    communication is disabled, all signal and emission fields are zeroed.
    """
    _height, _width, channels = _require_field_shapes(state, cfg)

    if not cfg.communication.enabled:
        state.signal.fill(0.0)
        state.signal_emission.fill(0.0)
        state.signal_reception.fill(0.0)
        return

    for channel in range(channels):
        field = state.signal[..., channel]
        diffusion = np.float32(cfg.communication.diffusion[channel])
        decay = np.float32(cfg.communication.decay[channel])
        if diffusion:
            field += diffusion * laplacian_wrap(field).astype(field.dtype, copy=False)
        if decay:
            field -= decay * field
        field += state.signal_emission[..., channel]
        np.clip(field, 0.0, 1.0, out=field)

    state.signal_emission.fill(0.0)


def apply_obstacle_mask(state: WorldState) -> None:
    """Zero passive environmental fields at obstacle cells.

    Mutates only environment/signal fields and ``state.occupancy`` at obstacle
    cells. It does not clear living cell state; initialization and movement are
    responsible for avoiding occupancy on obstacles.
    """
    obstacle = state.obstacle
    if obstacle.shape != state.food.shape:
        raise ValueError(
            f"state.obstacle shape {obstacle.shape} must match state.food shape {state.food.shape}"
        )
    if not np.any(obstacle):
        return

    state.food[obstacle] = 0.0
    state.toxin[obstacle] = 0.0
    state.noise[obstacle] = 0.0
    state.signal[obstacle, :] = 0.0
    state.signal_emission[obstacle, :] = 0.0
    state.signal_reception[obstacle, :] = 0.0
    state.signal_memory[obstacle, :] = 0.0
    state.occupancy[obstacle] = -1


# --- Advanced build overrides ------------------------------------------------
_mvp_update_food_field = _base_update_food_field
_mvp_update_signal_fields = _base_update_signal_fields


def update_food_field(state: WorldState, cfg: SimulationConfig) -> None:
    """Update food with optional logistic regrowth and waste recycling."""
    if not getattr(cfg.ecology, "advanced_enabled", False):
        _mvp_update_food_field(state, cfg)
        return

    from owl.core.advanced import ensure_advanced_fields

    _height, _width, _channels = _require_field_shapes(state, cfg)
    ensure_advanced_fields(state, cfg)
    assert state.waste is not None
    food = state.food
    if cfg.resources.food_diffusion:
        food += np.float32(cfg.resources.food_diffusion) * laplacian_wrap(food).astype(
            food.dtype, copy=False
        )
    carrying = np.float32(cfg.ecology.food_carrying_capacity)
    growth = np.float32(cfg.ecology.food_regrowth_rate) * food * (1.0 - food / carrying)
    recycle = np.float32(cfg.ecology.waste_recycle_rate) * state.waste
    food += growth.astype(food.dtype, copy=False) + recycle.astype(food.dtype, copy=False)
    if cfg.resources.food_decay:
        food -= np.float32(cfg.resources.food_decay) * food
    state.waste *= np.float32(1.0 - cfg.ecology.waste_decay)
    np.clip(food, 0.0, 1.0, out=food)
    np.clip(state.waste, 0.0, 1.0, out=state.waste)


def update_signal_fields(state: WorldState, cfg: SimulationConfig) -> None:
    """Update signal fields and optionally track strongest source identity."""
    from owl.core.advanced import ensure_advanced_fields, moore_directions

    if not getattr(cfg.communication, "source_tracking_enabled", False):
        _mvp_update_signal_fields(state, cfg)
        return

    _height, _width, channels = _require_field_shapes(state, cfg)
    ensure_advanced_fields(state, cfg)
    assert state.signal_source_id is not None
    assert state.neighbor_trust is not None

    # Winner-take-strongest immediate emission source before passive diffusion.
    state.signal_source_id.fill(-1)
    if cfg.communication.enabled:
        for channel in range(channels):
            local = state.signal_emission[..., channel]
            best_value = local.copy()
            best_source = state.occupancy.copy()
            for d, (dy, dx) in enumerate(moore_directions()):
                shifted = np.roll(np.roll(local, int(dy), axis=0), int(dx), axis=1)
                shifted_source = np.roll(np.roll(state.occupancy, int(dy), axis=0), int(dx), axis=1)
                trusted = shifted * state.neighbor_trust[..., d, channel]
                replace = trusted > best_value
                best_value[replace] = trusted[replace]
                best_source[replace] = shifted_source[replace]
            state.signal_source_id[..., channel] = np.where(best_value > 0, best_source, -1).astype(
                state.signal_source_id.dtype
            )

    _mvp_update_signal_fields(state, cfg)
    if state.signal_source_id is not None:
        state.signal_source_id[state.signal <= 0.0] = -1
