"""Observer-window integration functional.

This module implements the mathematical core of the Observer-Window Life
integration index.  It is intentionally independent of the main loop.  It
combines physical viability, memory, possibility flexibility, same-scale
coherence, cross-scale coupling, and conflict into a bounded cell-level
``state.integration`` field.
"""

from __future__ import annotations

from typing import cast

import numpy as np

from owl.core.config import SimulationConfig
from owl.core.state import WorldState, action_shape, field_shape
from owl.kernels.numpy_kernels import normalize_last_axis, sigmoid


def _cell_field(values: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    """Return ``values`` as a float32 cell field with exact target shape."""
    array = np.asarray(values, dtype=np.float32)
    try:
        broadcast = np.broadcast_to(array, shape)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be broadcastable to cell shape {shape}, got {array.shape}"
        ) from exc
    return np.asarray(broadcast, dtype=np.float32)


def entropy_normalized(possibility: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Compute normalized Shannon entropy of action possibilities.

    Parameters
    ----------
    possibility:
        Array with shape ``(..., num_actions)``. Values are repaired onto the
        probability simplex before entropy is computed so small numerical drift
        does not produce invalid logs.
    epsilon:
        Positive numerical tolerance.

    Returns
    -------
    np.ndarray
        Float32 array with shape ``possibility.shape[:-1]`` and values in
        ``[0, 1]``. A one-action distribution has entropy zero by definition.
    """
    P = np.asarray(possibility, dtype=np.float32)
    if P.ndim < 1:
        raise ValueError("possibility must have at least one axis")
    if P.shape[-1] <= 0:
        raise ValueError("possibility last axis must be nonempty")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if not np.all(np.isfinite(P)):
        raise ValueError("possibility must contain only finite values")

    k = P.shape[-1]
    if k == 1:
        return np.zeros(P.shape[:-1], dtype=np.float32)

    Pn = normalize_last_axis(P, epsilon=epsilon).astype(np.float32, copy=False)
    entropy = -np.sum(Pn * np.log(Pn + epsilon), axis=-1) / np.log(float(k))
    return cast(np.ndarray, np.clip(entropy, 0.0, 1.0).astype(np.float32))


def possibility_flexibility(
    possibility: np.ndarray,
    target: float,
    sigma: float,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Return an optimal-entropy flexibility term.

    Parameters
    ----------
    possibility:
        Action possibility distribution with shape ``(..., num_actions)``.
    target:
        Preferred normalized entropy in ``[0, 1]``. Values near this entropy are
        most flexible. This prevents either frozen certainty or pure noise from
        automatically maximizing integration.
    sigma:
        Positive Gaussian width around ``target``.
    epsilon:
        Positive numerical tolerance for entropy computation.

    Returns
    -------
    np.ndarray
        Float32 array with shape ``possibility.shape[:-1]`` and values in
        ``[0, 1]``.
    """
    if not (0.0 <= target <= 1.0):
        raise ValueError("target must be in [0, 1]")
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    entropy = entropy_normalized(possibility, epsilon=epsilon)
    flexibility = np.exp(-((entropy - target) ** 2) / (2.0 * sigma * sigma))
    return np.clip(flexibility, 0.0, 1.0).astype(np.float32)


def _readout_disagreement(state: WorldState) -> np.ndarray:
    """Return local readout disagreement in ``[0, 1]``.

    This baseline diagnostic compares each cell's discrete readout with the readouts
    of its eight toroidal neighbors. It is vectorized by summing equality masks
    over directional rolls.
    """
    readout = state.readout
    if readout.shape != field_shape(state):
        raise ValueError(
            f"state.readout must have cell shape {field_shape(state)}, got {readout.shape}"
        )

    same_count = np.zeros(readout.shape, dtype=np.float32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            shifted = np.roll(np.roll(readout, dy, axis=0), dx, axis=1)
            same_count += (shifted == readout).astype(np.float32)

    agreement = same_count / 8.0
    disagreement = 1.0 - agreement
    disagreement[state.health <= 0.0] = 0.0
    return cast(np.ndarray, np.clip(disagreement, 0.0, 1.0).astype(np.float32))


def _parent_bias_conflict(
    state: WorldState, parent_bias: np.ndarray, cfg: SimulationConfig
) -> np.ndarray:
    """Return bounded mismatch between current possibility and parent policy bias."""
    h, w, k = action_shape(state)
    bias = np.asarray(parent_bias, dtype=np.float32)
    expected = (h, w, k)
    if bias.shape != expected:
        raise ValueError(f"parent_bias must have shape {expected}, got {bias.shape}")

    # Convert arbitrary finite parent logits/biases into a nonnegative policy.
    if not np.all(np.isfinite(bias)):
        raise ValueError("parent_bias must be finite")

    strength = np.sum(np.abs(bias), axis=-1)
    if np.all(strength <= cfg.actions.epsilon):
        return np.zeros((h, w), dtype=np.float32)

    shifted = bias - np.min(bias, axis=-1, keepdims=True)
    parent_policy = normalize_last_axis(shifted, epsilon=cfg.actions.epsilon)
    child_policy = normalize_last_axis(state.possibility, epsilon=cfg.actions.epsilon)

    l1_distance = 0.5 * np.sum(np.abs(parent_policy - child_policy), axis=-1)
    strength_scale = strength / (1.0 + strength)
    conflict = l1_distance * strength_scale
    conflict[state.health <= 0.0] = 0.0
    return cast(np.ndarray, np.clip(conflict, 0.0, 1.0).astype(np.float32))


def _signal_conflict(state: WorldState, cfg: SimulationConfig) -> np.ndarray:
    """Return bounded conflict implied by incompatible received signal channels."""
    shape = field_shape(state)
    signal = state.signal_reception
    if signal.ndim != 3 or signal.shape[:2] != shape:
        raise ValueError(
            f"state.signal_reception must have shape (height, width, channels), got {signal.shape}"
        )

    n = min(cfg.communication.num_channels, signal.shape[-1])
    if n == 0:
        return np.zeros(shape, dtype=np.float32)

    food = signal[..., 0] if n > 0 else 0.0
    danger = signal[..., 1] if n > 1 else 0.0
    threat = signal[..., 2] if n > 2 else 0.0
    coord = signal[..., 3] if n > 3 else 0.0
    distress = signal[..., 4] if n > 4 else 0.0

    # Food vs danger, threat vs coordination, and distress under threat are
    # treated as operational conflicts. This bounded heuristic is not a
    # psychological claim.
    raw = food * danger + threat * coord + 0.5 * distress * threat
    return np.clip(raw, 0.0, 1.0).astype(np.float32)


def compute_conflict(
    state: WorldState, parent_bias: np.ndarray, cfg: SimulationConfig
) -> np.ndarray:
    """Compute bounded conflict/error for the integration equation.

    Parameters
    ----------
    state:
        Runtime dense state. This function does not mutate state.
    parent_bias:
        Cell-level action-bias tensor with shape
        ``(height, width, len(Action))``. Later top-down code supplies this
        after upsampling patch/global intentions.
    cfg:
        Simulation coefficients.

    Returns
    -------
    np.ndarray
        Float32 cell-level field in ``[0, 1]`` combining parent/child policy
        mismatch, local readout disagreement, incompatible communication
        signals, and physical stress.
    """
    shape = field_shape(state)

    parent = _parent_bias_conflict(state, parent_bias, cfg)
    readout = _readout_disagreement(state)
    signal = _signal_conflict(state, cfg)

    physical_stress = (
        (1.0 - np.clip(state.health, 0.0, 1.0))
        + (1.0 - np.clip(state.boundary, 0.0, 1.0))
        + np.clip(state.toxin, 0.0, 1.0)
    ) / 3.0
    physical_stress = _cell_field(physical_stress, shape, "physical_stress")

    conflict = 0.40 * parent + 0.25 * readout + 0.20 * signal + 0.15 * physical_stress
    conflict = np.clip(conflict, 0.0, 1.0).astype(np.float32)
    conflict[state.health <= 0.0] = 0.0
    return conflict


def update_integration(
    state: WorldState,
    synchrony: np.ndarray,
    coherence: np.ndarray,
    cross_scale: np.ndarray,
    conflict: np.ndarray,
    cfg: SimulationConfig,
) -> None:
    """Update ``state.integration`` in-place using the OW integration functional.

    Parameters
    ----------
    state:
        Runtime dense state. This function mutates only ``state.integration``.
    synchrony, coherence, cross_scale, conflict:
        Cell-level fields broadcastable to ``state.health.shape``. Synchrony,
        coherence, and cross-scale coupling are positive terms; conflict is a
        negative term.
    cfg:
        Simulation coefficients. ``cfg.integration`` supplies all weights.

    Notes
    -----
    Integration is an admissibility metric for the simulation, not evidence of
    consciousness. It is bounded by a sigmoid and then clipped to ``[0, 1]``.
    Dead cells receive integration zero.
    """
    shape = field_shape(state)
    icfg = cfg.integration

    sync = _cell_field(synchrony, shape, "synchrony")
    coh = _cell_field(coherence, shape, "coherence")
    cross = _cell_field(cross_scale, shape, "cross_scale")
    err = _cell_field(conflict, shape, "conflict")

    flex = possibility_flexibility(
        state.possibility,
        target=icfg.entropy_target,
        sigma=icfg.entropy_sigma,
        epsilon=cfg.actions.epsilon,
    )

    z = (
        icfg.weight_memory * np.clip(state.memory, 0.0, 1.0)
        + icfg.weight_flexibility * flex
        + icfg.weight_synchrony * np.clip(sync, 0.0, 1.0)
        + icfg.weight_coherence * np.clip(coh, 0.0, 1.0)
        + icfg.weight_cross_scale * np.clip(cross, 0.0, 1.0)
        + icfg.weight_resource * np.clip(state.resource, 0.0, 1.0)
        + icfg.weight_boundary * np.clip(state.boundary, 0.0, 1.0)
        - icfg.weight_conflict * np.clip(err, 0.0, 1.0)
        - np.clip(state.threshold, 0.0, 1.0)
    ).astype(np.float32)

    updated = sigmoid(z)
    updated = np.asarray(updated, dtype=np.float32)
    updated = np.clip(updated, 0.0, 1.0)
    updated[state.health <= 0.0] = 0.0
    state.integration[...] = updated
