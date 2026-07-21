"""Health, resource, boundary, and metabolism updates.

This module implements physical survival consequences after readouts have been
selected: metabolic resource drain, toxin/starvation damage, and constructive
repair/integration actions. It does not choose actions and does not move cells.
"""

from __future__ import annotations

import numpy as np

from owl.core.actions import Action
from owl.core.config import SimulationConfig
from owl.core.constants import BOUNDED_CELL_FIELDS
from owl.core.state import WorldState, field_shape


def _alive_mask(state: WorldState) -> np.ndarray:
    """Return living, non-obstacle cells as a boolean field."""
    return (state.health > 0.0) & (~state.obstacle)


def _base_clip_life_fields(state: WorldState, cfg: SimulationConfig) -> None:
    """Clip bounded cell-life and communication-trait fields in place.

    Mutates the bounded fields listed in ``owl.core.constants.BOUNDED_CELL_FIELDS``
    plus channel receptivity, channel emission bias, channel trust, signal memory,
    and signal reception. Resource is clipped to
    ``[0, cfg.resources.max_resource]``; other life/trait fields are clipped to
    ``[0, 1]``.
    """
    shape = field_shape(state)
    for name in BOUNDED_CELL_FIELDS:
        arr = getattr(state, name)
        if arr.shape != shape:
            raise ValueError(f"state.{name} must have shape {shape}, got {arr.shape}")
        upper = cfg.resources.max_resource if name == "resource" else 1.0
        np.clip(arr, 0.0, upper, out=arr)

    for name in (
        "signal_reception",
        "signal_emission",
        "signal_memory",
        "channel_receptivity",
        "channel_emission_bias",
        "channel_trust_local",
    ):
        arr = getattr(state, name)
        if arr.shape[:2] != shape:
            raise ValueError(f"state.{name} must begin with cell shape {shape}, got {arr.shape}")
        np.clip(arr, 0.0, 1.0, out=arr)

    state.resource[state.obstacle] = 0.0
    state.health[state.obstacle] = 0.0
    state.boundary[state.obstacle] = 0.0


def _base_apply_metabolism_damage(state: WorldState, cfg: SimulationConfig) -> None:
    """Apply metabolism, starvation damage, toxin damage, and boundary erosion.

    Mutates
    -------
    state.resource:
        Decreased by metabolic cost for living cells.
    state.health:
        Decreased by starvation and toxin damage.
    state.boundary:
        Decreased by toxin pressure and severe health loss.
    """
    shape = field_shape(state)
    for name in ("resource", "health", "boundary", "metabolism", "toxin", "toxin_resistance"):
        if getattr(state, name).shape != shape:
            raise ValueError(
                f"state.{name} must have shape {shape}, got {getattr(state, name).shape}"
            )

    alive = _alive_mask(state).astype(np.float32)
    metabolism = (
        np.float32(cfg.resources.metabolism_base) * np.clip(state.metabolism, 0.0, 1.0) * alive
    )
    state.resource -= metabolism.astype(state.resource.dtype, copy=False)

    starvation = np.maximum(0.0, 0.05 * cfg.resources.max_resource - state.resource)
    toxin_damage = np.clip(state.toxin, 0.0, 1.0) * (
        1.0 - np.clip(state.toxin_resistance, 0.0, 1.0)
    )

    state.health -= (0.02 * starvation + 0.03 * toxin_damage * alive).astype(
        state.health.dtype, copy=False
    )
    state.boundary -= (0.01 * toxin_damage * alive).astype(state.boundary.dtype, copy=False)
    state.boundary -= (0.01 * np.maximum(0.0, 0.20 - state.health) * alive).astype(
        state.boundary.dtype, copy=False
    )

    clip_life_fields(state, cfg)


def _base_apply_repair_and_integrate(state: WorldState, cfg: SimulationConfig) -> None:
    """Apply physical effects of ``REPAIR`` and ``INTEGRATE`` readouts.

    Mutates
    -------
    state.resource:
        Spent by repair and integration actions.
    state.health, state.boundary:
        Improved by repair when resources are available.
    state.memory, state.boundary:
        Mildly improved by integration as an baseline persistence effect.
    """
    shape = field_shape(state)
    if state.readout.shape != shape:
        raise ValueError(f"state.readout must have shape {shape}, got {state.readout.shape}")

    alive = _alive_mask(state)
    repair_mask = (state.readout == int(Action.REPAIR)) & alive
    integrate_mask = (state.readout == int(Action.INTEGRATE)) & alive

    resource_norm = np.clip(
        state.resource / max(float(cfg.resources.max_resource), cfg.actions.epsilon), 0.0, 1.0
    )

    repair_cost = 0.020 * repair_mask.astype(np.float32)
    repair_amount = 0.060 * repair_mask.astype(np.float32) * resource_norm
    spend_repair = np.minimum(state.resource, repair_cost)
    state.resource -= spend_repair.astype(state.resource.dtype, copy=False)
    state.health += (0.50 * repair_amount).astype(state.health.dtype, copy=False)
    state.boundary += repair_amount.astype(state.boundary.dtype, copy=False)

    integrate_cost = 0.010 * integrate_mask.astype(np.float32)
    spend_integrate = np.minimum(state.resource, integrate_cost)
    state.resource -= spend_integrate.astype(state.resource.dtype, copy=False)
    state.memory += (
        0.030 * integrate_mask.astype(np.float32) * np.clip(state.memory_capacity, 0.0, 1.0)
    ).astype(state.memory.dtype, copy=False)
    state.boundary += (0.020 * integrate_mask.astype(np.float32) * resource_norm).astype(
        state.boundary.dtype, copy=False
    )
    state.integration += (0.010 * integrate_mask.astype(np.float32)).astype(
        state.integration.dtype, copy=False
    )

    clip_life_fields(state, cfg)


# --- Advanced build overrides ------------------------------------------------
_mvp_clip_life_fields = _base_clip_life_fields
_mvp_apply_metabolism_damage = _base_apply_metabolism_damage
_mvp_apply_repair_and_integrate = _base_apply_repair_and_integrate


def clip_life_fields(state: WorldState, cfg: SimulationConfig) -> None:
    """Clip baseline and optional advanced life fields."""
    _mvp_clip_life_fields(state, cfg)
    from owl.core.advanced import ensure_advanced_fields

    ensure_advanced_fields(state, cfg)
    for name in (
        "digestion",
        "waste",
        "age_stress",
        "last_intake",
        "prediction_error",
        "starvation_debt",
        "movement_loop_score",
        "development_stage",
        "symbiosis",
    ):
        arr = getattr(state, name, None)
        if isinstance(arr, np.ndarray):
            np.clip(arr, 0.0, 1.0, out=arr)
    if isinstance(state.genome, np.ndarray):
        np.clip(state.genome, 0.0, 1.0, out=state.genome)
    if isinstance(state.action_cooldown, np.ndarray):
        np.clip(state.action_cooldown, 0.0, 1.0, out=state.action_cooldown)
    if isinstance(state.deception_memory, np.ndarray):
        np.clip(state.deception_memory, 0.0, 1.0, out=state.deception_memory)
    if isinstance(state.neighbor_trust, np.ndarray):
        np.clip(state.neighbor_trust, 0.0, 1.0, out=state.neighbor_trust)


def apply_metabolism_damage(state: WorldState, cfg: SimulationConfig) -> None:
    """Apply metabolism; advanced mode uses starvation debt and nonlinear damage.

    Resource exhaustion is no longer an immediate death condition. Low resource
    accumulates bounded starvation debt; that debt damages health over time, and
    death occurs through health/boundary failure rather than a healthy cell being
    cleared simply because resource temporarily reached zero.
    """
    if not getattr(cfg.ecology, "advanced_enabled", False):
        _mvp_apply_metabolism_damage(state, cfg)
        return

    from owl.core.advanced import ensure_advanced_fields

    ensure_advanced_fields(state, cfg)
    assert state.age_stress is not None
    assert state.starvation_debt is not None
    alive_bool = _alive_mask(state)
    alive = alive_bool.astype(np.float32)

    state.age_stress[...] = np.clip(
        state.age.astype(np.float32) / np.float32(cfg.ecology.age_stress_scale), 0.0, 1.0
    )
    maintenance = np.float32(cfg.resources.metabolism_base) * (
        0.5 + np.clip(state.metabolism, 0.0, 1.0) + 0.15 * state.age_stress
    )
    movement_actions = np.isin(state.readout, [2, 3, 4, 5, 6, 7, 8, 9]).astype(np.float32)
    action_cost = np.float32(cfg.resources.movement_cost) * movement_actions
    state.resource -= ((maintenance + action_cost) * alive).astype(state.resource.dtype, copy=False)

    resource_norm = np.clip(
        state.resource / max(float(cfg.resources.max_resource), cfg.actions.epsilon), 0.0, 1.0
    )
    starving = alive_bool & (resource_norm <= float(cfg.resources.emergency_feed_threshold))

    state.starvation_debt[starving] += np.float32(cfg.resources.starvation_debt_gain)
    recovering = alive_bool & (~starving)
    state.starvation_debt[recovering] -= (
        np.float32(cfg.resources.starvation_debt_recovery) * resource_norm[recovering]
    )
    state.starvation_debt[~alive_bool] = 0.0
    np.clip(state.starvation_debt, 0.0, 1.0, out=state.starvation_debt)

    toxin_damage = np.power(
        np.clip(state.toxin, 0.0, 1.0), np.float32(cfg.ecology.toxin_damage_exponent)
    ) * (1.0 - np.clip(state.toxin_resistance, 0.0, 1.0))
    starvation_damage = np.float32(cfg.resources.starvation_health_damage) * state.starvation_debt
    state.health -= (starvation_damage * alive + 0.035 * toxin_damage * alive).astype(
        state.health.dtype, copy=False
    )
    state.boundary -= (0.015 * toxin_damage * alive).astype(state.boundary.dtype, copy=False)
    clip_life_fields(state, cfg)


def apply_repair_and_integrate(state: WorldState, cfg: SimulationConfig) -> None:
    """Apply repair/integration; advanced repair is saturating in resource."""
    if not getattr(cfg.ecology, "advanced_enabled", False):
        _mvp_apply_repair_and_integrate(state, cfg)
        return

    from owl.core.actions import Action
    from owl.core.advanced import ensure_advanced_fields

    ensure_advanced_fields(state, cfg)
    alive = _alive_mask(state)
    repair_mask = (state.readout == int(Action.REPAIR)) & alive
    integrate_mask = (state.readout == int(Action.INTEGRATE)) & alive
    q = np.clip(state.resource, 0.0, cfg.resources.max_resource)
    repair_sat = q / (np.float32(cfg.ecology.repair_half_saturation) + q)
    state.resource -= (
        0.025 * repair_mask.astype(np.float32) + 0.010 * integrate_mask.astype(np.float32)
    ).astype(state.resource.dtype, copy=False)
    state.health += (0.030 * repair_mask.astype(np.float32) * repair_sat).astype(
        state.health.dtype, copy=False
    )
    state.boundary += (
        0.060 * repair_mask.astype(np.float32) * repair_sat
        + 0.020 * integrate_mask.astype(np.float32)
    ).astype(state.boundary.dtype, copy=False)
    state.memory += (
        0.030 * integrate_mask.astype(np.float32) * np.clip(state.memory_capacity, 0.0, 1.0)
    ).astype(state.memory.dtype, copy=False)
    state.integration += (0.015 * integrate_mask.astype(np.float32)).astype(
        state.integration.dtype, copy=False
    )
    clip_life_fields(state, cfg)
