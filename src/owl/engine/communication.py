"""Universal observer-window communication.

Every living observer window can signal. This module implements the passive baseline
communication substrate: automatic signal intent, costly emission into fields,
channel memory, local channel trust, and signal-conflict estimates. It does not
choose cell actions; deliberate communication is introduced by later utility and
actualization passes.
"""

from __future__ import annotations

from typing import cast

import numpy as np

from owl.core.actions import SignalChannel
from owl.core.config import SimulationConfig
from owl.core.state import WorldState
from owl.engine.sensing import compute_crowding


def _channel_count(state: WorldState, cfg: SimulationConfig) -> int:
    """Validate and return configured channel count."""
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
    return cfg.communication.num_channels


def _has_channel(channels: int, channel: SignalChannel) -> bool:
    """Return whether ``channel`` exists under the configured channel count."""
    return int(channel) < channels


def _alive_mask(state: WorldState) -> np.ndarray:
    """Return living, non-obstacle cells as a float32 mask."""
    return ((state.health > 0.0) & (~state.obstacle)).astype(np.float32)


def _channel_bias(state: WorldState, channel: SignalChannel, channels: int) -> np.ndarray:
    """Return channel emission bias or zeros when the channel is unavailable."""
    if not _has_channel(channels, channel):
        return np.zeros_like(state.health, dtype=np.float32)
    return np.clip(state.channel_emission_bias[..., int(channel)], 0.0, 1.0)


def _received_channel(state: WorldState, channel: SignalChannel, channels: int) -> np.ndarray:
    """Return received signal channel or zeros when unavailable."""
    if not _has_channel(channels, channel):
        return np.zeros_like(state.health, dtype=np.float32)
    return np.clip(state.signal_reception[..., int(channel)], 0.0, 1.0)


def choose_signal_intent(state: WorldState, cfg: SimulationConfig) -> np.ndarray:
    """Choose the dominant automatic communication channel for each cell.

    Returns
    -------
    np.ndarray
        ``int16`` cell-level field with shape ``(height, width)``. Values are
        channel indices. ``-1`` means no channel has positive intent or
        communication is disabled. This function does not mutate state.
    """
    channels = _channel_count(state, cfg)
    if not cfg.communication.enabled:
        return np.full(state.health.shape, -1, dtype=np.int16)

    intents = compute_automatic_signal_intents(state, cfg)
    best = np.argmax(intents, axis=-1).astype(np.int16)
    best_value = np.max(intents, axis=-1)
    best[(best_value <= cfg.actions.epsilon) | (state.health <= 0.0) | state.obstacle] = -1
    best[best >= channels] = -1
    return cast(np.ndarray, best)


def compute_automatic_signal_intents(state: WorldState, cfg: SimulationConfig) -> np.ndarray:
    """Compute passive automatic signal intent strengths.

    Returns
    -------
    np.ndarray
        Channel-level ``float32`` tensor with shape
        ``(height, width, cfg.communication.num_channels)`` and values in
        ``[0, 1]``. The function is pure and does not mutate state.

    Channel meanings:
    - FOOD: resource opportunity, driven by food and grazing.
    - DANGER: toxin or low health.
    - THREAT: aggression under crowding.
    - COORDINATION: cooperative high-integration alignment.
    - DISTRESS: low health or low boundary.
    - REPRODUCTION: viable high-resource/high-health reproduction pressure.
    - TERRITORY: boundary/crowding claim.
    - INTEGRATION: high integration/coupling pulse.
    """
    channels = _channel_count(state, cfg)
    intents = np.zeros((*state.health.shape, channels), dtype=np.float32)

    if not cfg.communication.enabled:
        return intents

    alive = _alive_mask(state)
    emit = np.clip(state.emit_strength, 0.0, 1.0) * alive
    resource = np.clip(state.resource, 0.0, 1.0)
    health = np.clip(state.health, 0.0, 1.0)
    boundary = np.clip(state.boundary, 0.0, 1.0)
    integration = np.clip(state.integration, 0.0, 1.0)
    crowding = compute_crowding(state)

    if _has_channel(channels, SignalChannel.FOOD):
        intents[..., int(SignalChannel.FOOD)] = (
            np.clip(state.grazing, 0.0, 1.0)
            * np.clip(state.food, 0.0, 1.0)
            * emit
            * _channel_bias(state, SignalChannel.FOOD, channels)
        )

    if _has_channel(channels, SignalChannel.DANGER):
        danger_pressure = np.maximum(np.clip(state.toxin, 0.0, 1.0), 1.0 - health)
        intents[..., int(SignalChannel.DANGER)] = (
            danger_pressure * emit * _channel_bias(state, SignalChannel.DANGER, channels)
        )

    if _has_channel(channels, SignalChannel.THREAT):
        intents[..., int(SignalChannel.THREAT)] = (
            np.clip(state.aggression, 0.0, 1.0)
            * crowding
            * emit
            * _channel_bias(state, SignalChannel.THREAT, channels)
        )

    if _has_channel(channels, SignalChannel.COORDINATION):
        intents[..., int(SignalChannel.COORDINATION)] = (
            integration
            * np.clip(state.cooperation, 0.0, 1.0)
            * emit
            * _channel_bias(state, SignalChannel.COORDINATION, channels)
        )

    if _has_channel(channels, SignalChannel.DISTRESS):
        distress = np.maximum(1.0 - health, 1.0 - boundary)
        intents[..., int(SignalChannel.DISTRESS)] = (
            distress * emit * _channel_bias(state, SignalChannel.DISTRESS, channels)
        )

    if _has_channel(channels, SignalChannel.REPRODUCTION):
        reproduction_pressure = (
            np.clip(state.reproduction_rate, 0.0, 1.0) * resource * health * boundary * integration
        )
        intents[..., int(SignalChannel.REPRODUCTION)] = (
            reproduction_pressure
            * emit
            * _channel_bias(state, SignalChannel.REPRODUCTION, channels)
        )

    if _has_channel(channels, SignalChannel.TERRITORY):
        territory = boundary * crowding * (0.5 + 0.5 * np.clip(state.aggression, 0.0, 1.0))
        intents[..., int(SignalChannel.TERRITORY)] = (
            territory * emit * _channel_bias(state, SignalChannel.TERRITORY, channels)
        )

    if _has_channel(channels, SignalChannel.INTEGRATION):
        intents[..., int(SignalChannel.INTEGRATION)] = (
            integration
            * np.clip(state.coupling_strength, 0.0, 1.0)
            * emit
            * _channel_bias(state, SignalChannel.INTEGRATION, channels)
        )

    intents *= np.clip(state.signal_precision, 0.0, 1.0)[..., None]
    intents[state.obstacle, :] = 0.0
    return np.clip(intents, 0.0, 1.0).astype(np.float32, copy=False)


def _base_emit_signals(state: WorldState, cfg: SimulationConfig) -> None:
    """Convert automatic intents into costly signal emissions.

    Mutates
    -------
    state.signal_emission:
        Adds bounded per-channel emissions. Emissions enter ``state.signal`` on
        the next call to ``update_signal_fields``.
    state.resource:
        Subtracts configured communication cost and clips resource to
        ``[0, cfg.resources.max_resource]``.
    """
    channels = _channel_count(state, cfg)
    if not cfg.communication.enabled:
        state.signal_emission.fill(0.0)
        return

    intents = compute_automatic_signal_intents(state, cfg)
    integration_factor = 0.25 + 0.75 * np.clip(state.integration, 0.0, 1.0)
    resource_factor = np.clip(state.resource / cfg.resources.max_resource, 0.0, 1.0)
    efficiency = np.clip(state.emit_efficiency, 0.0, 1.0)

    emission = (
        intents * integration_factor[..., None] * resource_factor[..., None] * efficiency[..., None]
    )
    np.clip(emission, 0.0, 1.0, out=emission)

    # Resource cost is summed across all emitted channels. Higher efficiency
    # reduces cost but never makes signaling free unless base_emit_cost is zero.
    total_emission = np.sum(emission, axis=-1)
    cost = np.float32(cfg.communication.base_emit_cost) * total_emission / (0.10 + efficiency)
    state.resource -= cost.astype(state.resource.dtype, copy=False)
    np.clip(state.resource, 0.0, cfg.resources.max_resource, out=state.resource)

    state.signal_emission[..., :channels] += emission.astype(
        state.signal_emission.dtype, copy=False
    )
    np.clip(state.signal_emission, 0.0, 1.0, out=state.signal_emission)
    state.signal_emission[state.obstacle, :] = 0.0


def update_signal_memory(state: WorldState, cfg: SimulationConfig) -> None:
    """Update channel-specific signal memory traces in place.

    Mutates ``state.signal_memory`` using an exponential moving average of
    current reception. Dead and obstacle cells hold zero signal memory in this
    baseline approximation.
    """
    _channel_count(state, cfg)
    if not cfg.communication.enabled:
        state.signal_memory.fill(0.0)
        return

    retention = np.float32(0.97)
    state.signal_memory[...] = (
        retention * state.signal_memory + (1.0 - retention) * state.signal_reception
    )
    state.signal_memory *= _alive_mask(state)[..., None]
    np.clip(state.signal_memory, 0.0, 1.0, out=state.signal_memory)


def _base_update_channel_trust(
    state: WorldState,
    prev_resource: np.ndarray,
    prev_health: np.ndarray,
    prev_integration: np.ndarray,
    cfg: SimulationConfig,
) -> None:
    """Update local channel trust from post-signal outcomes.

    The baseline trust rule credits channels that were received before a favorable
    local outcome and debits channels before an unfavorable one.

    Mutates ``state.channel_trust_local`` and clips trust to ``[0, 1]``.
    """
    _channel_count(state, cfg)
    expected_shape = state.health.shape
    for name, array in (
        ("prev_resource", prev_resource),
        ("prev_health", prev_health),
        ("prev_integration", prev_integration),
    ):
        if array.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}, got {array.shape}")

    if not cfg.communication.enabled:
        return

    outcome = (
        0.4 * (state.resource - prev_resource)
        + 0.4 * (state.health - prev_health)
        + 0.2 * (state.integration - prev_integration)
    )
    outcome = np.clip(outcome, -1.0, 1.0).astype(np.float32, copy=False)

    state.channel_trust_local += (
        np.float32(cfg.communication.trust_lr)
        * outcome[..., None]
        * np.clip(state.signal_reception, 0.0, 1.0)
    )
    state.channel_trust_local *= _alive_mask(state)[..., None]
    np.clip(state.channel_trust_local, 0.0, 1.0, out=state.channel_trust_local)


def _base_compute_signal_conflict(state: WorldState, cfg: SimulationConfig) -> np.ndarray:
    """Compute bounded conflict among received communication channels.

    Returns
    -------
    np.ndarray
        Cell-level ``float32`` field with shape ``(height, width)`` and values
        in ``[0, 1]``. Conflict rises when food and danger signals coexist, when
        threat and coordination coexist, or when channel reception is highly
        dispersed across incompatible meanings.
    """
    channels = _channel_count(state, cfg)
    if not cfg.communication.enabled:
        return np.zeros_like(state.health, dtype=np.float32)

    food = _received_channel(state, SignalChannel.FOOD, channels)
    danger = _received_channel(state, SignalChannel.DANGER, channels)
    threat = _received_channel(state, SignalChannel.THREAT, channels)
    coordination = _received_channel(state, SignalChannel.COORDINATION, channels)

    incompatible = food * danger + threat * coordination
    dispersion = np.std(np.clip(state.signal_reception, 0.0, 1.0), axis=-1)
    conflict = 0.65 * incompatible + 0.35 * dispersion
    conflict = conflict.astype(np.float32, copy=False)
    conflict[state.obstacle] = 0.0
    return cast(np.ndarray, np.clip(conflict, 0.0, 1.0))


# --- Advanced build overrides ------------------------------------------------
_mvp_emit_signals = _base_emit_signals
_mvp_update_channel_trust = _base_update_channel_trust
_mvp_compute_signal_conflict = _base_compute_signal_conflict


def compute_intentional_signal_policy(state: WorldState, cfg: SimulationConfig) -> np.ndarray:
    """Return intentional communication policy for cells choosing COMMUNICATE."""
    from owl.core.advanced import ensure_advanced_fields
    from owl.kernels.numpy_kernels import normalize_last_axis

    ensure_advanced_fields(state, cfg)
    intents = compute_automatic_signal_intents(state, cfg)
    policy = np.zeros_like(intents, dtype=np.float32)
    speaking = (
        state.readout == int(__import__("owl.core.actions", fromlist=["Action"]).Action.COMMUNICATE)
    ) & _alive_mask(state).astype(bool)
    if np.any(speaking):
        policy[speaking, :] = normalize_last_axis(intents[speaking, :])
    return policy


def emit_signals(state: WorldState, cfg: SimulationConfig) -> None:
    """Emit automatic/intentional signals with optional source-aware tracking."""
    if not getattr(cfg.communication, "source_tracking_enabled", False):
        _mvp_emit_signals(state, cfg)
        return

    from owl.core.advanced import ensure_advanced_fields

    ensure_advanced_fields(state, cfg)
    if not cfg.communication.enabled:
        state.signal_emission.fill(0.0)
        return
    auto = compute_automatic_signal_intents(state, cfg)
    intentional = compute_intentional_signal_policy(state, cfg)
    mix = np.float32(cfg.communication.intentional_mix)
    emission = (1.0 - mix) * auto + mix * intentional
    emission *= np.clip(state.emit_strength, 0.0, 1.0)[..., None]
    emission *= np.clip(state.emit_efficiency, 0.0, 1.0)[..., None]
    emission *= np.clip(state.signal_precision, 0.0, 1.0)[..., None]
    emission *= (1.0 - 0.5 * np.clip(state.deception_bias, 0.0, 1.0))[..., None]
    total_emission = np.sum(emission, axis=-1)
    cost = (
        np.float32(cfg.communication.base_emit_cost)
        * total_emission
        / (0.10 + np.clip(state.emit_efficiency, 0.0, 1.0))
    )
    state.resource -= cost.astype(state.resource.dtype, copy=False)
    np.clip(state.resource, 0.0, cfg.resources.max_resource, out=state.resource)
    state.signal_emission += emission.astype(state.signal_emission.dtype, copy=False)
    np.clip(state.signal_emission, 0.0, 1.0, out=state.signal_emission)
    state.signal_emission[state.obstacle, :] = 0.0


def update_channel_trust(
    state: WorldState,
    prev_resource: np.ndarray,
    prev_health: np.ndarray,
    prev_integration: np.ndarray,
    cfg: SimulationConfig,
) -> None:
    """Update channel trust; advanced mode also updates neighbor/deception memory."""
    if not getattr(cfg.communication, "source_tracking_enabled", False):
        _mvp_update_channel_trust(state, prev_resource, prev_health, prev_integration, cfg)
        return

    from owl.core.advanced import ensure_advanced_fields, moore_directions

    ensure_advanced_fields(state, cfg)
    assert state.deception_memory is not None
    assert state.source_confidence is not None
    assert state.neighbor_trust is not None
    assert state.signal_source_id is not None
    _mvp_update_channel_trust(state, prev_resource, prev_health, prev_integration, cfg)
    outcome = (
        0.4 * (state.resource - prev_resource)
        + 0.4 * (state.health - prev_health)
        + 0.2 * (state.integration - prev_integration)
    )
    outcome = np.clip(outcome, -1.0, 1.0).astype(np.float32)
    reception = np.clip(state.signal_reception, 0.0, 1.0)
    state.deception_memory += (
        np.float32(cfg.communication.deception_penalty)
        * reception
        * np.maximum(0.0, -outcome)[..., None]
    )
    state.deception_memory -= (
        np.float32(cfg.communication.trust_lr) * reception * np.maximum(0.0, outcome)[..., None]
    )
    np.clip(state.deception_memory, 0.0, 1.0, out=state.deception_memory)
    state.source_confidence += (
        np.float32(cfg.communication.source_trust_lr) * reception * outcome[..., None]
    )
    np.clip(state.source_confidence, 0.0, 1.0, out=state.source_confidence)

    for d, (dy, dx) in enumerate(moore_directions()):
        shifted_outcome = np.roll(np.roll(outcome, -int(dy), axis=0), -int(dx), axis=1)
        delta = (
            np.float32(cfg.communication.source_trust_lr) * shifted_outcome[..., None] * reception
        )
        state.neighbor_trust[..., d, :] += delta
    state.neighbor_trust *= np.float32(1.0 - cfg.communication.neighbor_trust_decay)
    np.clip(state.neighbor_trust, 0.0, 1.0, out=state.neighbor_trust)


def compute_signal_conflict(state: WorldState, cfg: SimulationConfig) -> np.ndarray:
    """Compute signal conflict with optional deception pressure."""
    base = _mvp_compute_signal_conflict(state, cfg)
    if not getattr(cfg.communication, "source_tracking_enabled", False):
        return base
    from owl.core.advanced import ensure_advanced_fields

    ensure_advanced_fields(state, cfg)
    assert state.deception_memory is not None
    assert state.source_confidence is not None
    assert state.neighbor_trust is not None
    assert state.signal_source_id is not None
    deception = np.mean(np.clip(state.deception_memory, 0.0, 1.0), axis=-1)
    provenance = (state.signal_source_id >= 0).astype(np.float32).std(axis=-1)
    return cast(
        np.ndarray,
        np.clip(base + 0.35 * deception + 0.15 * provenance, 0.0, 1.0).astype(np.float32),
    )
