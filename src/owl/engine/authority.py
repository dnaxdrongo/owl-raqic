"""Action authority and permission functions.

Authority is the feasibility layer: it answers which actions are physically
possible for each observer window before the possibility distribution is
actualized. Utility can desire an action; authority may still suppress it.
"""

from __future__ import annotations

import numpy as np

from owl.core.actions import MOVE_DELTAS, Action
from owl.core.config import SimulationConfig
from owl.core.constants import DEFAULT_FLOAT_DTYPE
from owl.core.state import WorldState, field_shape
from owl.kernels.numpy_kernels import neighbor_mean_wrap


def _alive_mask(state: WorldState) -> np.ndarray:
    """Return living, non-obstacle cells as a boolean field."""
    return (state.health > 0.0) & (~state.obstacle)


def _action_count() -> int:
    """Return the canonical number of cell-level actions."""
    return len(Action)


def _parse_enabled_actions(cfg: SimulationConfig) -> set[int] | None:
    """Return configured enabled action indices, or ``None`` if all are enabled."""
    names = list(cfg.actions.enabled_actions)
    if not names:
        return None

    enabled: set[int] = set()
    valid_names = {action.name.upper(): action for action in Action}
    for raw_name in names:
        key = str(raw_name).upper()
        if key not in valid_names:
            raise ValueError(
                f"unknown enabled action {raw_name!r}; valid actions are {sorted(valid_names)}"
            )
        enabled.add(int(valid_names[key]))
    return enabled


def _base_compute_authority(state: WorldState, cfg: SimulationConfig) -> np.ndarray:
    """Compute action feasibility for every cell and action.

    Parameters
    ----------
    state:
        Runtime dense state. This function does not mutate state.
    cfg:
        Simulation coefficients.

    Returns
    -------
    np.ndarray
        Float32 authority tensor with shape ``(height, width, len(Action))`` and
        values in ``[0, 1]``. Impossible actions receive zero authority. Dead or
        obstacle cells receive authority only for ``REST`` so actualization can
        produce a normalized one-hot rest distribution.

    Notes
    -----
    Authority is not preference. It is the action-permission mask that keeps
    utility from actualizing physically impossible behaviors.
    """
    h, w = field_shape(state)
    authority = np.zeros((h, w, _action_count()), dtype=DEFAULT_FLOAT_DTYPE)

    alive = _alive_mask(state)
    alive_f = alive.astype(DEFAULT_FLOAT_DTYPE)
    resource = np.clip(state.resource, 0.0, cfg.resources.max_resource)
    resource_norm = np.clip(
        resource / max(float(cfg.resources.max_resource), cfg.actions.epsilon), 0.0, 1.0
    )
    health = np.clip(state.health, 0.0, 1.0)
    boundary = np.clip(state.boundary, 0.0, 1.0)
    integration = np.clip(state.integration, 0.0, 1.0)

    # Minimal actions for living cells.
    authority[..., int(Action.REST)] = 1.0
    if bool(cfg.action_transitions.enabled):
        authority[..., int(Action.SENSE)] = (
            alive_f
            * bool(cfg.action_transitions.active_sense_enabled)
            * (resource >= float(cfg.action_transitions.active_sense_cost)).astype(
                DEFAULT_FLOAT_DTYPE
            )
        )
    else:
        authority[..., int(Action.SENSE)] = alive_f

    enough_move_resource = (resource >= cfg.resources.movement_cost).astype(DEFAULT_FLOAT_DTYPE)
    mobility = np.clip(state.mobility, 0.0, 1.0)
    for action in MOVE_DELTAS:
        authority[..., int(action)] = alive_f * mobility * enough_move_resource

    local_food = np.clip(0.5 * state.food + 0.5 * neighbor_mean_wrap(state.food), 0.0, 1.0)
    authority[..., int(Action.FEED)] = (
        alive_f
        * np.clip(state.grazing, 0.0, 1.0)
        * (local_food > 0.005).astype(DEFAULT_FLOAT_DTYPE)
    )

    if cfg.communication.enabled:
        can_afford_signal = (resource >= float(cfg.communication.base_emit_cost)).astype(
            DEFAULT_FLOAT_DTYPE
        )
        emit_capacity = np.clip(state.emit_strength, 0.0, 1.0) * np.clip(
            state.emit_efficiency, 0.0, 1.0
        )
        authority[..., int(Action.COMMUNICATE)] = alive_f * can_afford_signal * emit_capacity

    # Inhibition, integration, and repair are available when alive, with
    # strength limited by relevant physical/trait quantities.
    authority[..., int(Action.INHIBIT)] = alive_f * np.maximum(
        np.clip(state.aggression, 0.0, 1.0),
        0.5 * np.clip(state.cooperation, 0.0, 1.0),
    )
    authority[..., int(Action.INTEGRATE)] = alive_f * np.maximum(
        0.15,
        np.clip(state.coupling_strength, 0.0, 1.0),
    )
    authority[..., int(Action.REPAIR)] = alive_f * (resource > 0.03).astype(DEFAULT_FLOAT_DTYPE)

    if cfg.reproduction.enabled:
        authority[..., int(Action.REPRODUCE)] = (
            alive_f
            * (resource_norm >= cfg.reproduction.min_resource).astype(DEFAULT_FLOAT_DTYPE)
            * (health >= cfg.reproduction.min_health).astype(DEFAULT_FLOAT_DTYPE)
            * (boundary >= cfg.reproduction.min_boundary).astype(DEFAULT_FLOAT_DTYPE)
            * (integration >= cfg.reproduction.min_integration).astype(DEFAULT_FLOAT_DTYPE)
            * np.clip(state.reproduction_rate, 0.0, 1.0)
        )

    adjacent_life = neighbor_mean_wrap(alive_f)
    if cfg.predation.enabled:
        authority[..., int(Action.INGEST)] = (
            alive_f
            * (np.clip(state.predation, 0.0, 1.0) >= cfg.predation.min_predation_trait).astype(
                DEFAULT_FLOAT_DTYPE
            )
            * (resource > 0.05).astype(DEFAULT_FLOAT_DTYPE)
            * (adjacent_life > 0.0).astype(DEFAULT_FLOAT_DTYPE)
        )

    # Advanced topology actions use conservative default utility values.
    authority[..., int(Action.EXPEL)] = alive_f * 0.0
    authority[..., int(Action.SPLIT)] = alive_f * 0.0
    authority[..., int(Action.MERGE)] = alive_f * 0.0

    # FLEE/PURSUE are high-level tendencies; concrete movement still occurs
    # through directional movement actions.
    danger_pressure = np.clip(
        state.toxin + np.mean(np.clip(state.signal_reception, 0.0, 1.0), axis=-1), 0.0, 1.0
    )
    if bool(cfg.action_transitions.enabled):
        flee_compiled = getattr(state, "flee_compiled_action", None)
        pursue_compiled = getattr(state, "pursue_compiled_action", None)
        if not isinstance(flee_compiled, np.ndarray) or not isinstance(
            pursue_compiled, np.ndarray
        ):
            raise RuntimeError("v1 action target context must precede authority")
        authority[..., int(Action.FLEE)] = (
            alive_f
            * mobility
            * enough_move_resource
            * bool(cfg.action_transitions.flee_execution_enabled)
            * (flee_compiled >= 0).astype(DEFAULT_FLOAT_DTYPE)
        )
        authority[..., int(Action.PURSUE)] = (
            alive_f
            * mobility
            * enough_move_resource
            * bool(cfg.action_transitions.pursue_execution_enabled)
            * (pursue_compiled >= 0).astype(DEFAULT_FLOAT_DTYPE)
        )
    else:
        authority[..., int(Action.FLEE)] = (
            alive_f * mobility * (danger_pressure > 0.02).astype(DEFAULT_FLOAT_DTYPE)
        )
        authority[..., int(Action.PURSUE)] = (
            alive_f
            * np.clip(state.predation + state.aggression, 0.0, 1.0)
            * (adjacent_life > 0.0).astype(DEFAULT_FLOAT_DTYPE)
        )

    authority = np.clip(authority, 0.0, 1.0).astype(DEFAULT_FLOAT_DTYPE)
    authority = apply_enabled_action_mask(authority, cfg)
    authority = suppress_dead_cells(authority, state)
    return authority


def apply_enabled_action_mask(authority: np.ndarray, cfg: SimulationConfig) -> np.ndarray:
    """Apply the globally configured action set.

    Parameters
    ----------
    authority:
        Authority tensor with shape ``(height, width, len(Action))``. This
        function does not mutate the input; it returns a clipped copy.
    cfg:
        Simulation coefficients. If ``cfg.actions.enabled_actions`` is empty,
        all actions remain enabled.

    Returns
    -------
    np.ndarray
        Authority tensor with disabled actions set to zero.
    """
    auth = np.asarray(authority, dtype=np.float32)
    if auth.ndim != 3 or auth.shape[-1] != _action_count():
        raise ValueError(
            f"authority must have shape (height, width, {len(Action)}), got {auth.shape}"
        )
    if not np.all(np.isfinite(auth)):
        raise ValueError("authority must contain only finite values")

    out = np.clip(auth, 0.0, 1.0).astype(DEFAULT_FLOAT_DTYPE, copy=True)
    enabled = _parse_enabled_actions(cfg)
    if enabled is None:
        return out

    mask = np.zeros((len(Action),), dtype=DEFAULT_FLOAT_DTYPE)
    for idx in enabled:
        mask[idx] = 1.0
    # REST is always enabled so dead cells and obstacles can be represented as
    # one-hot rest distributions after actualization.
    mask[int(Action.REST)] = 1.0

    out *= mask[None, None, :]
    return out.astype(DEFAULT_FLOAT_DTYPE, copy=False)


def suppress_dead_cells(authority: np.ndarray, state: WorldState) -> np.ndarray:
    """Force dead/obstacle cells to rest-only authority.

    Parameters
    ----------
    authority:
        Authority tensor. This function does not mutate the input.
    state:
        Runtime dense state.

    Returns
    -------
    np.ndarray
        Authority tensor where cells with ``health <= 0`` or obstacles have all
        actions suppressed except ``REST``.
    """
    expected_spatial = field_shape(state)
    auth = np.asarray(authority, dtype=np.float32)
    if auth.shape != (*expected_spatial, len(Action)):
        raise ValueError(
            f"authority must have shape {(*expected_spatial, len(Action))}, got {auth.shape}"
        )

    out = np.clip(auth, 0.0, 1.0).astype(DEFAULT_FLOAT_DTYPE, copy=True)
    dead = (state.health <= 0.0) | state.obstacle
    if np.any(dead):
        out[dead, :] = 0.0
        out[dead, int(Action.REST)] = 1.0
    return out.astype(DEFAULT_FLOAT_DTYPE, copy=False)


# --- Decision-homeostasis authority override ---------------------------------
_decision_base_compute_authority = _base_compute_authority


def compute_authority(state: WorldState, cfg: SimulationConfig) -> np.ndarray:
    """Compute authority with carrying-capacity reproduction suppression."""
    authority = _decision_base_compute_authority(state, cfg)
    if getattr(cfg.cross_scale_homeostasis, "enabled", False):
        try:
            from owl.engine.utility import reproduction_viability_field

            viability = reproduction_viability_field(state, cfg)
            authority[..., int(Action.REPRODUCE)] *= np.clip(viability, 0.0, 1.0)
        except Exception:
            pass
    # Hunger/food repair: FEED authority should be available when food is local
    # or reachable, but still zero when disabled by action mask later.
    if getattr(cfg.decision_homeostasis, "enabled", False):
        alive = _alive_mask(state).astype(DEFAULT_FLOAT_DTYPE)
        local_food = np.clip(0.5 * state.food + 0.5 * neighbor_mean_wrap(state.food), 0.0, 1.0)
        authority[..., int(Action.FEED)] = np.maximum(
            authority[..., int(Action.FEED)],
            alive
            * np.clip(state.grazing, 0.0, 1.0)
            * (local_food > 0.001).astype(DEFAULT_FLOAT_DTYPE),
        )
    authority = apply_enabled_action_mask(np.clip(authority, 0.0, 1.0), cfg)
    authority = suppress_dead_cells(authority, state)
    return authority.astype(DEFAULT_FLOAT_DTYPE, copy=False)
