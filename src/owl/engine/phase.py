"""Phase, synchrony, coherence, and cross-scale coupling.

This module implements the fractal/mosaic phase layer for Observer-Window Life.
It is deliberately array-first: all functions operate over dense cell-level
fields stored on :class:`owl.core.state.WorldState`.  The phase field supports
same-scale coherence and parent/child cross-scale coupling without importing the
main engine loop or any visualization/recording layer.
"""

from __future__ import annotations

from typing import cast

import numpy as np

from owl.core.config import SimulationConfig
from owl.core.state import WorldState, field_shape
from owl.kernels.numpy_kernels import neighbor_mean_wrap
from owl.science.counter_rng import RNGStream, normal01

_TWO_PI = np.float32(2.0 * np.pi)


def _as_cell_field(
    values: np.ndarray | float | None, shape: tuple[int, int], name: str
) -> np.ndarray:
    """Return ``values`` as a float32 cell-level array.

    Scalars and broadcast-compatible arrays are accepted. The returned array has
    exact shape ``shape``. A clear ``ValueError`` is raised for incompatible
    inputs instead of relying on accidental NumPy broadcasting.
    """
    if values is None:
        return np.zeros(shape, dtype=np.float32)
    array = np.asarray(values, dtype=np.float32)
    try:
        broadcast = np.broadcast_to(array, shape)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be broadcastable to cell shape {shape}, got {array.shape}"
        ) from exc
    return np.asarray(broadcast, dtype=np.float32)


def _alive_mask(state: WorldState) -> np.ndarray:
    """Return the cell-level alive mask used by phase diagnostics."""
    return (state.health > 0.0) & (state.boundary > 0.0)


def _neighbor_phase_mean(phase: np.ndarray) -> np.ndarray:
    """Return local toroidal circular mean phase of neighboring cells."""
    sin_mean = neighbor_mean_wrap(np.sin(phase).astype(np.float32))
    cos_mean = neighbor_mean_wrap(np.cos(phase).astype(np.float32))
    return cast(np.ndarray, np.arctan2(sin_mean, cos_mean).astype(np.float32))


def _base_update_phase(
    state: WorldState,
    parent_phase: np.ndarray | float | None,
    rng: np.random.Generator,
    cfg: SimulationConfig,
) -> None:
    """Update cell oscillator phases in-place.

    Parameters
    ----------
    state:
        Runtime dense state. This function mutates only ``state.phase``.
    parent_phase:
        Cell-level parent/patch phase field, or a scalar/array broadcastable to
        the cell shape. Later aggregation code should pass an upsampled patch
        phase field.
    rng:
        Explicit random generator used for deterministic phase noise.
    cfg:
        Simulation coefficients. ``cfg.phase`` supplies base angular velocity,
        same-scale coupling, parent coupling, and noise.

    Notes
    -----
    The update is a Kuramoto-style baseline approximation: each living cell advances
    by a base angular velocity, is pulled toward the local neighbor circular
    mean, is weakly pulled toward the parent phase, and receives bounded
    Gaussian noise. Dead cells are left unchanged. All resulting phases are
    wrapped into ``[0, 2*pi)``.
    """
    if rng is None:
        raise ValueError("rng must be an explicit np.random.Generator")

    shape = field_shape(state)
    parent = _as_cell_field(parent_phase, shape, "parent_phase")

    phase = np.asarray(state.phase, dtype=np.float32)
    if phase.shape != shape:
        raise ValueError(f"state.phase must have cell shape {shape}, got {phase.shape}")

    neighbor_phase = _neighbor_phase_mean(phase)
    flat = np.arange(phase.size, dtype=np.uint64).reshape(shape)
    occupancy = getattr(state, "occupancy", None)
    if isinstance(occupancy, np.ndarray) and occupancy.shape == shape:
        ow_ids = np.where(occupancy >= 0, occupancy, flat).astype(np.uint64)
    else:
        ow_ids = flat
    noise = (
        normal01(
            int(cfg.world.seed),
            int(state.tick),
            ow_ids,
            RNGStream.PHASE_NOISE,
            0,
            xp=np,
            dtype=np.float64,
        )
        * float(cfg.phase.phase_noise_sigma)
    ).astype(np.float32)

    same_pull = cfg.phase.same_scale_coupling * np.sin(neighbor_phase - phase)
    parent_pull = cfg.phase.parent_coupling * np.sin(parent - phase)

    delta = (cfg.phase.base_omega + same_pull + parent_pull + noise).astype(np.float32)
    alive = _alive_mask(state)

    updated = phase.copy()
    updated[alive] = np.mod(updated[alive] + delta[alive], _TWO_PI)
    state.phase[...] = updated.astype(np.float32)


def compute_local_synchrony(state: WorldState, cfg: SimulationConfig) -> np.ndarray:
    """Compute local synchrony as a bounded cell-level field.

    Parameters
    ----------
    state:
        Runtime dense state. This function does not mutate state.
    cfg:
        Simulation coefficients, accepted for API stability.

    Returns
    -------
    np.ndarray
        Float32 array with shape ``(height, width)`` and values in ``[0, 1]``.
        Values near one indicate that a cell and its local Moore-neighborhood
        have similar phase; values near zero indicate local phase dispersion.
    """
    del cfg  # Reserved for future synchrony coefficients.

    shape = field_shape(state)
    phase = np.asarray(state.phase, dtype=np.float32)
    if phase.shape != shape:
        raise ValueError(f"state.phase must have cell shape {shape}, got {phase.shape}")

    sin_phase = np.sin(phase).astype(np.float32)
    cos_phase = np.cos(phase).astype(np.float32)

    # Include the center cell plus the eight toroidal neighbors.
    sin_local = (sin_phase + 8.0 * neighbor_mean_wrap(sin_phase)) / 9.0
    cos_local = (cos_phase + 8.0 * neighbor_mean_wrap(cos_phase)) / 9.0

    synchrony = sin_local * sin_local + cos_local * cos_local
    synchrony = np.clip(synchrony, 0.0, 1.0).astype(np.float32)
    synchrony[~_alive_mask(state)] = 0.0
    return synchrony


def compute_cell_coherence(state: WorldState, cfg: SimulationConfig) -> np.ndarray:
    """Compute same-scale phase coherence with local neighbors.

    Parameters
    ----------
    state:
        Runtime dense state. This function does not mutate state.
    cfg:
        Simulation coefficients, accepted for API stability.

    Returns
    -------
    np.ndarray
        Float32 cell-level field in ``[0, 1]``. The value is one when a cell is
        aligned with its neighbor circular mean and zero when it is opposite.
    """
    del cfg

    shape = field_shape(state)
    phase = np.asarray(state.phase, dtype=np.float32)
    if phase.shape != shape:
        raise ValueError(f"state.phase must have cell shape {shape}, got {phase.shape}")

    neighbor_phase = _neighbor_phase_mean(phase)
    coherence = 0.5 + 0.5 * np.cos(neighbor_phase - phase)
    coherence = np.clip(coherence, 0.0, 1.0).astype(np.float32)
    coherence[~_alive_mask(state)] = 0.0
    return coherence


def compute_cross_scale_coupling(
    state: WorldState,
    parent_phase: np.ndarray,
    cfg: SimulationConfig,
) -> np.ndarray:
    """Compute bounded parent-child phase alignment.

    Parameters
    ----------
    state:
        Runtime dense state. This function does not mutate state.
    parent_phase:
        Parent/patch phase field broadcastable to ``state.phase.shape``.
    cfg:
        Simulation coefficients, accepted for API stability.

    Returns
    -------
    np.ndarray
        Float32 cell-level field in ``[0, 1]``. Values near one mean the cell's
        phase is aligned with its parent window phase.
    """
    del cfg

    shape = field_shape(state)
    parent = _as_cell_field(parent_phase, shape, "parent_phase")
    phase = np.asarray(state.phase, dtype=np.float32)

    coupling = 0.5 + 0.5 * np.cos(parent - phase)
    coupling = np.clip(coupling, 0.0, 1.0).astype(np.float32)
    coupling[~_alive_mask(state)] = 0.0
    return coupling


def compute_meaning_alignment(
    state: WorldState,
    coherence: np.ndarray,
    cross_scale: np.ndarray,
    cfg: SimulationConfig,
) -> np.ndarray:
    """Compute an baseline meaning/alignment diagnostic.

    Parameters
    ----------
    state:
        Runtime dense state. This function does not mutate state.
    coherence:
        Cell-level same-scale coherence field.
    cross_scale:
        Cell-level parent-child coupling field.
    cfg:
        Simulation coefficients. Used only to respect configured channel count.

    Returns
    -------
    np.ndarray
        Float32 cell-level field in ``[0, 1]``. This is not an ontological
        meaning claim; it is an operational diagnostic combining coherence,
        cross-scale alignment, memory, and relevant communication reception.
    """
    shape = field_shape(state)
    coh = _as_cell_field(coherence, shape, "coherence")
    cross = _as_cell_field(cross_scale, shape, "cross_scale")

    communication_alignment = np.zeros(shape, dtype=np.float32)
    if cfg.communication.num_channels > 3 and state.signal_reception.shape[-1] > 3:
        communication_alignment += 0.5 * state.signal_reception[..., 3]  # COORDINATION
    if cfg.communication.num_channels > 7 and state.signal_reception.shape[-1] > 7:
        communication_alignment += 0.5 * state.signal_reception[..., 7]  # INTEGRATION

    alignment = (
        0.35 * coh
        + 0.35 * cross
        + 0.15 * np.clip(state.memory, 0.0, 1.0)
        + 0.15 * np.clip(communication_alignment, 0.0, 1.0)
    )
    alignment = np.clip(alignment, 0.0, 1.0).astype(np.float32)
    alignment[~_alive_mask(state)] = 0.0
    return cast(np.ndarray, alignment)


# --- Advanced build overrides ------------------------------------------------
_mvp_update_phase = _base_update_phase


def _roll_stack_8(field: np.ndarray) -> np.ndarray:
    """Return Moore-neighborhood rolled stack in canonical order."""
    from owl.core.advanced import moore_directions

    return np.stack(
        [
            np.roll(np.roll(field, int(dy), axis=0), int(dx), axis=1)
            for dy, dx in moore_directions()
        ],
        axis=-1,
    )


def update_phase(
    state: WorldState,
    parent_phase: np.ndarray | float | None,
    rng: np.random.Generator,
    cfg: SimulationConfig,
) -> None:
    """Update phases; advanced hierarchy uses weighted Kuramoto neighbors."""
    if not getattr(cfg.hierarchy, "dynamic_patches", False):
        _mvp_update_phase(state, parent_phase, rng, cfg)
        return

    from owl.core.advanced import ensure_advanced_fields

    ensure_advanced_fields(state, cfg)
    assert state.phase_lag is not None
    assert state.same_scale_weight is not None
    assert state.parent_weight is not None
    assert state.phase_frequency is not None
    alive = _alive_mask(state)
    parent = _as_cell_field(parent_phase, field_shape(state), "parent_phase")
    neighbor_phase = _roll_stack_8(state.phase)
    delta = neighbor_phase - state.phase[..., None] - state.phase_lag[..., None]
    weighted_drive = np.sum(state.same_scale_weight * np.sin(delta), axis=-1)
    parent_drive = state.parent_weight * np.sin(parent - state.phase)
    flat = np.arange(state.phase.size, dtype=np.uint64).reshape(state.phase.shape)
    ow_ids = np.where(state.occupancy >= 0, state.occupancy, flat).astype(np.uint64)
    noise = (
        normal01(
            int(cfg.world.seed),
            int(state.tick),
            ow_ids,
            RNGStream.PHASE_NOISE,
            0,
            xp=np,
            dtype=np.float64,
        )
        * float(cfg.phase.phase_noise_sigma)
    ).astype(np.float32)
    omega = state.phase_frequency
    state.phase[...] = np.mod(
        state.phase + omega + weighted_drive + parent_drive + noise, 2.0 * np.pi
    )
    state.phase[~alive] = 0.0
