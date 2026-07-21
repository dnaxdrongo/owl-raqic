"""Memory, identity, and signal-trace updates.

The scalar memory field is an baseline approximation of temporal persistence. It is
updated from recent physical viability, integration, action readouts, and signal
pressure while remaining a bounded cell-level array.
"""

from __future__ import annotations

from typing import cast

import numpy as np

from owl.core.actions import Action, SignalChannel
from owl.core.config import SimulationConfig
from owl.core.constants import DEFAULT_FLOAT_DTYPE
from owl.core.state import WorldState, field_shape


def _channel_or_zero(
    state: WorldState, channel: SignalChannel, cfg: SimulationConfig
) -> np.ndarray:
    """Return a signal-reception channel or zeros if the config omits it."""
    shape = field_shape(state)
    idx = int(channel)
    n = min(cfg.communication.num_channels, state.signal_reception.shape[-1])
    if idx >= n:
        return np.zeros(shape, dtype=DEFAULT_FLOAT_DTYPE)
    return np.clip(state.signal_reception[..., idx], 0.0, 1.0).astype(
        DEFAULT_FLOAT_DTYPE, copy=False
    )


def encode_experience(state: WorldState, cfg: SimulationConfig) -> np.ndarray:
    """Encode current physical/behavioral state into a bounded memory target.

    Parameters
    ----------
    state:
        Runtime dense state. This function does not mutate state.
    cfg:
        Simulation coefficients.

    Returns
    -------
    np.ndarray
        Cell-level ``float32`` field with shape ``(height, width)`` and values
        in ``[0, 1]``. Dead and obstacle cells return zero experience.
    """
    shape = field_shape(state)
    for name in ("resource", "health", "boundary", "integration", "readout", "memory_capacity"):
        if getattr(state, name).shape != shape:
            raise ValueError(
                f"state.{name} must have shape {shape}, got {getattr(state, name).shape}"
            )

    alive = ((state.health > 0.0) & (~state.obstacle)).astype(np.float32)
    resource_norm = np.clip(
        state.resource / max(float(cfg.resources.max_resource), cfg.actions.epsilon), 0.0, 1.0
    )
    health = np.clip(state.health, 0.0, 1.0)
    boundary = np.clip(state.boundary, 0.0, 1.0)
    integration = np.clip(state.integration, 0.0, 1.0)

    action_trace = np.zeros(shape, dtype=np.float32)
    action_trace += 0.40 * (state.readout == int(Action.FEED))
    action_trace += 0.30 * (state.readout == int(Action.REPAIR))
    action_trace += 0.35 * (state.readout == int(Action.INTEGRATE))
    action_trace += 0.20 * (state.readout == int(Action.COMMUNICATE))
    action_trace = np.clip(action_trace, 0.0, 1.0)

    signal_trace = np.maximum.reduce(
        [
            _channel_or_zero(state, SignalChannel.FOOD, cfg),
            _channel_or_zero(state, SignalChannel.DANGER, cfg),
            _channel_or_zero(state, SignalChannel.COORDINATION, cfg),
            _channel_or_zero(state, SignalChannel.INTEGRATION, cfg),
        ]
    )

    experience = (
        0.22 * resource_norm
        + 0.22 * health
        + 0.18 * boundary
        + 0.20 * integration
        + 0.10 * action_trace
        + 0.08 * signal_trace
    )
    experience *= np.clip(state.memory_capacity, 0.0, 1.0)
    experience *= alive
    return cast(np.ndarray, np.clip(experience, 0.0, 1.0).astype(DEFAULT_FLOAT_DTYPE, copy=False))


def update_memory(state: WorldState, cfg: SimulationConfig) -> None:
    """Update scalar memory persistence in place.

    Mutates ``state.memory`` using an exponential moving average of
    :func:`encode_experience`. Dead and obstacle cells hold zero memory in the
    baseline implementation.
    """
    shape = field_shape(state)
    if state.memory.shape != shape:
        raise ValueError(f"state.memory must have shape {shape}, got {state.memory.shape}")

    # Memory settings share the main schema. Keep a
    # stable default while allowing a future cfg.memory.decay without changing
    # this public function.
    retention = float(getattr(getattr(cfg, "memory", object()), "decay", 0.95))
    retention = float(np.clip(retention, 0.0, 1.0))

    experience = encode_experience(state, cfg)
    state.memory[...] = retention * state.memory + (1.0 - retention) * experience
    state.memory *= ((state.health > 0.0) & (~state.obstacle)).astype(np.float32)
    np.clip(state.memory, 0.0, 1.0, out=state.memory)


def compute_identity_persistence(state: WorldState, cfg: SimulationConfig) -> np.ndarray:
    """Compute a bounded diagnostic identity-persistence field.

    Parameters
    ----------
    state:
        Runtime dense state. This function does not mutate state.
    cfg:
        Simulation coefficients. Included for API consistency and future
        weighting extensions.

    Returns
    -------
    np.ndarray
        Cell-level ``float32`` field in ``[0, 1]``. High values mean the cell has
        stable memory, boundary integrity, health, and integration.
    """
    del cfg
    shape = field_shape(state)
    for name in ("memory", "boundary", "health", "integration"):
        if getattr(state, name).shape != shape:
            raise ValueError(
                f"state.{name} must have shape {shape}, got {getattr(state, name).shape}"
            )

    alive = ((state.health > 0.0) & (~state.obstacle)).astype(np.float32)
    persistence = (
        0.35 * np.clip(state.memory, 0.0, 1.0)
        + 0.30 * np.clip(state.boundary, 0.0, 1.0)
        + 0.20 * np.clip(state.health, 0.0, 1.0)
        + 0.15 * np.clip(state.integration, 0.0, 1.0)
    )
    persistence *= alive
    return cast(np.ndarray, np.clip(persistence, 0.0, 1.0).astype(DEFAULT_FLOAT_DTYPE, copy=False))
