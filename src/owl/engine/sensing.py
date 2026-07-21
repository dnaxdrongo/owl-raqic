"""Local sensing and passive perception functions.

Sensing turns physical and communication fields into bounded cell-level inputs.
Except for ``compute_signal_reception``, functions in this module are pure array
computations and do not mutate :class:`owl.core.state.WorldState`.
"""

from __future__ import annotations

from typing import cast

import numpy as np

from owl.core.config import SimulationConfig
from owl.core.state import WorldState
from owl.kernels.numpy_kernels import neighbor_mean_wrap


def _alive_mask(state: WorldState) -> np.ndarray:
    """Return a float32 mask for living, non-obstacle cells."""
    return ((state.health > 0.0) & (~state.obstacle)).astype(np.float32)


def _validate_channel_shapes(state: WorldState, cfg: SimulationConfig) -> None:
    """Raise a clear error when communication tensors do not match config."""
    expected = (*state.health.shape, cfg.communication.num_channels)
    for name in (
        "signal",
        "signal_emission",
        "signal_reception",
        "signal_memory",
        "channel_receptivity",
        "channel_emission_bias",
        "channel_trust_local",
    ):
        shape = getattr(state, name).shape
        if shape != expected:
            raise ValueError(f"state.{name} must have shape {expected}, got {shape}")


def compute_local_food_pressure(state: WorldState, cfg: SimulationConfig) -> np.ndarray:
    """Compute local food availability sensed by each cell.

    Parameters
    ----------
    state:
        Runtime dense state. This function does not mutate state.
    cfg:
        Simulation coefficients. Present for API symmetry with other sensing
        functions.

    Returns
    -------
    np.ndarray
        Cell-level ``float32`` field with shape ``(height, width)`` and values
        in ``[0, 1]``. The value combines direct food with the Moore-neighborhood
        mean.
    """
    del cfg
    pressure = 0.5 * state.food + 0.5 * neighbor_mean_wrap(state.food)
    pressure = np.asarray(pressure, dtype=np.float32)
    pressure[state.obstacle] = 0.0
    return np.clip(pressure, 0.0, 1.0)


def compute_local_toxin_pressure(state: WorldState, cfg: SimulationConfig) -> np.ndarray:
    """Compute local toxin/danger sensed by each cell.

    Returns a bounded ``(height, width)`` field combining direct toxin with
    neighboring toxin pressure. This function does not mutate state.
    """
    del cfg
    pressure = 0.5 * state.toxin + 0.5 * neighbor_mean_wrap(state.toxin)
    pressure = np.asarray(pressure, dtype=np.float32)
    pressure[state.obstacle] = 0.0
    return np.clip(pressure, 0.0, 1.0)


def _base_compute_signal_reception(state: WorldState, cfg: SimulationConfig) -> None:
    """Compute effective signal reception for every cell and channel.

    Mutates ``state.signal_reception`` only. Reception combines local signal
    pressure, receive sensitivity, channel receptivity, trust, and boundary
    openness. Dead cells and obstacle cells receive zero signal.
    """
    _validate_channel_shapes(state, cfg)

    if not cfg.communication.enabled:
        state.signal_reception.fill(0.0)
        return

    local_signal = 0.5 * state.signal + 0.5 * neighbor_mean_wrap(state.signal)
    boundary_openness = 0.25 + 0.75 * np.clip(state.boundary, 0.0, 1.0)
    alive = _alive_mask(state)

    state.signal_reception[...] = (
        local_signal
        * np.clip(state.receive_sensitivity, 0.0, 1.0)[..., None]
        * np.clip(state.channel_receptivity, 0.0, 1.0)
        * np.clip(state.channel_trust_local, 0.0, 1.0)
        * boundary_openness[..., None]
        * alive[..., None]
    )
    np.clip(state.signal_reception, 0.0, 1.0, out=state.signal_reception)


def compute_crowding(state: WorldState) -> np.ndarray:
    """Compute local living-neighbor density.

    Returns
    -------
    np.ndarray
        Cell-level ``float32`` field with shape ``(height, width)`` in
        ``[0, 1]``. The center cell is not included; the value is the mean of
        the eight neighboring living-cell indicators.
    """
    crowding = neighbor_mean_wrap(_alive_mask(state)).astype(np.float32, copy=False)
    crowding[state.obstacle] = 0.0
    return np.clip(crowding, 0.0, 1.0)


def compute_novelty(state: WorldState, cfg: SimulationConfig) -> np.ndarray:
    """Compute a bounded novelty/curiosity pressure field.

    The baseline novelty proxy combines mismatch between current received signals and
    remembered signals with local food/toxin contrast. Later utility code can use
    this field as a curiosity term without requiring action choice here.

    Returns a ``(height, width)`` field in ``[0, 1]`` and does not mutate state.
    """
    _validate_channel_shapes(state, cfg)
    signal_delta = np.mean(np.abs(state.signal_reception - state.signal_memory), axis=-1)
    food_contrast = np.abs(state.food - neighbor_mean_wrap(state.food))
    toxin_contrast = np.abs(state.toxin - neighbor_mean_wrap(state.toxin))
    novelty = 0.60 * signal_delta + 0.25 * food_contrast + 0.15 * toxin_contrast
    novelty = novelty.astype(np.float32, copy=False)
    novelty[state.obstacle] = 0.0
    return cast(np.ndarray, np.clip(novelty, 0.0, 1.0))


# --- Advanced build overrides ------------------------------------------------
_mvp_compute_signal_reception = _base_compute_signal_reception


def compute_signal_reception(state: WorldState, cfg: SimulationConfig) -> None:
    """Compute reception with optional source/deception modulation."""
    if not getattr(cfg.communication, "source_tracking_enabled", False):
        _mvp_compute_signal_reception(state, cfg)
        return

    from owl.core.advanced import ensure_advanced_fields

    ensure_advanced_fields(state, cfg)
    assert state.deception_memory is not None
    assert state.source_confidence is not None
    _mvp_compute_signal_reception(state, cfg)
    state.signal_reception *= (1.0 - 0.5 * np.clip(state.deception_memory, 0.0, 1.0)).astype(
        state.signal_reception.dtype, copy=False
    )
    state.signal_reception *= (0.50 + 0.50 * np.clip(state.source_confidence, 0.0, 1.0)).astype(
        state.signal_reception.dtype, copy=False
    )
    np.clip(state.signal_reception, 0.0, 1.0, out=state.signal_reception)
