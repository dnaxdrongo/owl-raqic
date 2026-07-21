"""Environmental feeding and nutrient conversion.

This module implements the physical survival rule that an observer-window cell
can convert environmental food into internal resource when its readout is
``Action.FEED``. The functions are array-first and do not perform movement,
decision-making, rendering, or recording.
"""

from __future__ import annotations

from typing import cast

import numpy as np

from owl.core.actions import Action
from owl.core.config import SimulationConfig
from owl.core.constants import DEFAULT_FLOAT_DTYPE
from owl.core.state import WorldState, field_shape


def _alive_mask(state: WorldState) -> np.ndarray:
    """Return a boolean mask for living, non-obstacle cells."""
    return (state.health > 0.0) & (~state.obstacle)


def _base_compute_intake(state: WorldState, cfg: SimulationConfig) -> np.ndarray:
    """Compute environmental-food intake for cells currently feeding.

    Parameters
    ----------
    state:
        Runtime dense state. This function does not mutate state.
    cfg:
        Simulation coefficients.

    Returns
    -------
    np.ndarray
        Cell-level ``float32`` field with shape ``(height, width)``. Values are
        nonnegative and bounded above by the local environmental food and the
        remaining resource capacity.
    """
    shape = field_shape(state)
    if state.food.shape != shape:
        raise ValueError(f"state.food must have shape {shape}, got {state.food.shape}")
    if state.readout.shape != shape:
        raise ValueError(f"state.readout must have shape {shape}, got {state.readout.shape}")
    if state.grazing.shape != shape:
        raise ValueError(f"state.grazing must have shape {shape}, got {state.grazing.shape}")

    feeding = (state.readout == int(Action.FEED)) & _alive_mask(state)
    food = np.clip(state.food, 0.0, 1.0)
    grazing = np.clip(state.grazing, 0.0, 1.0)
    current_resource = np.clip(state.resource, 0.0, cfg.resources.max_resource)
    remaining_capacity = np.maximum(0.0, cfg.resources.max_resource - current_resource)

    raw_intake = (
        np.float32(cfg.resources.feed_efficiency) * food * grazing * feeding.astype(np.float32)
    )
    intake = np.minimum(np.minimum(raw_intake, food), remaining_capacity)
    intake[state.obstacle] = 0.0
    return cast(np.ndarray, np.clip(intake, 0.0, 1.0).astype(DEFAULT_FLOAT_DTYPE, copy=False))


def _base_apply_feeding(state: WorldState, cfg: SimulationConfig) -> None:
    """Convert local food into internal resource for feeding cells.

    Mutates
    -------
    state.resource:
        Increased by computed intake and clipped to
        ``[0, cfg.resources.max_resource]``.
    state.food:
        Decreased by the same intake and clipped to ``[0, 1]``.
    """
    intake = compute_intake(state, cfg)
    state.resource += intake.astype(state.resource.dtype, copy=False)
    state.food -= intake.astype(state.food.dtype, copy=False)

    np.clip(state.resource, 0.0, cfg.resources.max_resource, out=state.resource)
    np.clip(state.food, 0.0, 1.0, out=state.food)
    state.resource[state.obstacle] = 0.0


def deposit_resource_residue(state: WorldState, amount: np.ndarray, positions: np.ndarray) -> None:
    """Deposit resource residue into the environmental food field.

    Parameters
    ----------
    state:
        Runtime dense state. This function mutates ``state.food`` only.
    amount:
        Either a scalar-like array, a length-``N`` vector, or a full cell-level
        field. Negative values are ignored by clipping to zero.
    positions:
        Integer array with shape ``(N, 2)`` containing ``(y, x)`` positions.

    Notes
    -----
    ``np.add.at`` is used so duplicate positions accumulate correctly. This is
    useful for later sparse death/ingestion events.
    """
    pos = np.asarray(positions, dtype=np.int64)
    if pos.size == 0:
        return
    if pos.ndim != 2 or pos.shape[1] != 2:
        raise ValueError(f"positions must have shape (N, 2), got {pos.shape}")

    h, w = field_shape(state)
    y = pos[:, 0]
    x = pos[:, 1]
    if np.any((y < 0) | (y >= h) | (x < 0) | (x >= w)):
        raise ValueError("positions contain out-of-bounds coordinates")

    amt = np.asarray(amount, dtype=np.float32)
    if amt.shape == state.food.shape:
        values = amt[y, x]
    elif amt.ndim == 0:
        values = np.full(pos.shape[0], float(amt), dtype=np.float32)
    else:
        values = np.ravel(amt).astype(np.float32, copy=False)
        if values.shape[0] != pos.shape[0]:
            raise ValueError(
                "amount must be scalar, cell-shaped, or length equal to number of positions; "
                f"got amount shape {amt.shape} and {pos.shape[0]} positions"
            )

    values = np.clip(values, 0.0, 1.0)
    np.add.at(state.food, (y, x), values.astype(state.food.dtype, copy=False))
    np.clip(state.food, 0.0, 1.0, out=state.food)
    state.food[state.obstacle] = 0.0


# --- Advanced build overrides ------------------------------------------------
# Preserve the baseline feeding functions when optional dynamics are disabled.
_mvp_compute_intake = _base_compute_intake
_mvp_apply_feeding = _base_apply_feeding


def compute_intake(state: WorldState, cfg: SimulationConfig) -> np.ndarray:
    """Compute feeding intake with optional Monod saturation.

    Advanced mode uses
    ``I = FEED * feed_efficiency * grazing * F/(K+F) * capacity``.
    This is monotone in food and capacity-limited by remaining resource space.
    """
    if not getattr(cfg.ecology, "advanced_enabled", False):
        return _mvp_compute_intake(state, cfg)

    from owl.core.advanced import ensure_advanced_fields

    ensure_advanced_fields(state, cfg)
    assert state.last_intake is not None
    feeding = (state.readout == int(Action.FEED)) & _alive_mask(state)
    food = np.clip(state.food, 0.0, 1.0)
    q = np.clip(state.resource, 0.0, cfg.resources.max_resource)
    capacity = np.maximum(
        0.0, 1.0 - q / max(float(cfg.resources.max_resource), cfg.actions.epsilon)
    )
    monod = food / (float(cfg.ecology.monod_half_saturation) + food)
    intake = (
        feeding.astype(np.float32)
        * np.float32(cfg.resources.feed_efficiency)
        * np.clip(state.grazing, 0.0, 1.0)
        * monod
        * capacity
    )
    intake = np.minimum(intake, food)
    state.last_intake[...] = np.clip(intake, 0.0, 1.0)
    return state.last_intake.astype(np.float32, copy=True)


def apply_feeding(state: WorldState, cfg: SimulationConfig) -> None:
    """Apply feeding with immediate assimilation plus digestion buffering.

    Advanced mode repairs the early-starvation pathology by converting a bounded
    fraction of intake directly into usable resource in the same tick, while the
    remainder enters the digestion buffer and is assimilated over later ticks.
    """
    if not getattr(cfg.ecology, "advanced_enabled", False):
        _mvp_apply_feeding(state, cfg)
        return

    from owl.core.advanced import ensure_advanced_fields

    ensure_advanced_fields(state, cfg)
    assert state.digestion is not None
    assert state.waste is not None
    assert state.last_intake is not None
    intake = compute_intake(state, cfg).astype(np.float32, copy=False)
    immediate_fraction = np.float32(cfg.resources.feeding_immediate_fraction)
    immediate = immediate_fraction * intake
    buffered = (np.float32(1.0) - immediate_fraction) * intake

    state.food -= intake.astype(state.food.dtype, copy=False)
    state.resource += immediate.astype(state.resource.dtype, copy=False)
    state.digestion += buffered.astype(state.digestion.dtype, copy=False)
    state.last_intake[...] = np.clip(intake, 0.0, 1.0)

    # Continue digestion conversion in the same feeding pass so feeding has an
    # immediate and a delayed resource pathway.
    digested = np.float32(cfg.ecology.digestion_decay) * state.digestion
    state.resource += (np.float32(cfg.ecology.digestion_efficiency) * digested).astype(
        state.resource.dtype, copy=False
    )
    state.waste += (np.float32(1.0 - cfg.ecology.digestion_efficiency) * digested).astype(
        state.waste.dtype, copy=False
    )
    state.digestion -= digested.astype(state.digestion.dtype, copy=False)

    np.clip(state.food, 0.0, 1.0, out=state.food)
    np.clip(state.resource, 0.0, cfg.resources.max_resource, out=state.resource)
    np.clip(state.digestion, 0.0, 1.0, out=state.digestion)
    np.clip(state.waste, 0.0, 1.0, out=state.waste)
    state.food[state.obstacle] = 0.0
    state.resource[state.obstacle] = 0.0
    state.last_intake[state.obstacle] = 0.0
