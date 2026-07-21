"""Possibility update and action actualization.

This module is the quantum-inspired action layer in computational form: utility
and authority are converted to logits, logits become a normalized possibility
distribution, and one readout/action is actualized for each cell.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np

from owl.core.actions import DIAGONAL_MOVES, MOVE_DELTAS, REVERSE_MOVE_ACTION, Action
from owl.core.config import SimulationConfig
from owl.core.constants import DEFAULT_FLOAT_DTYPE, DEFAULT_READOUT_DTYPE
from owl.core.state import WorldState, action_shape, field_shape
from owl.kernels.numba_kernels import sample_categorical_grid
from owl.kernels.numpy_kernels import normalize_last_axis, softmax_stable


def _validate_action_tensor(state: WorldState, values: np.ndarray, name: str) -> np.ndarray:
    """Return a float32 action tensor with exact ``state`` action shape."""
    expected = action_shape(state)
    array = np.asarray(values, dtype=np.float32)
    if array.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.astype(DEFAULT_FLOAT_DTYPE, copy=False)


def _rest_one_hot(shape: tuple[int, int], actions: int) -> np.ndarray:
    """Return a cell-action tensor with probability one on REST."""
    out = np.zeros((*shape, actions), dtype=DEFAULT_FLOAT_DTYPE)
    out[..., int(Action.REST)] = 1.0
    return out


def _enabled_actions(cfg: SimulationConfig) -> set[int] | None:
    """Return enabled action ids, or ``None`` when all actions are globally enabled."""
    names = list(cfg.actions.enabled_actions)
    if not names:
        return None
    valid = {a.name.upper(): int(a) for a in Action}
    out: set[int] = {int(Action.REST)}
    for raw in names:
        key = str(raw).upper()
        if key not in valid:
            raise ValueError(f"unknown enabled action {raw!r}; valid actions are {sorted(valid)}")
        out.add(valid[key])
    return out


def _action_is_enabled(action: Action, cfg: SimulationConfig) -> bool:
    enabled = _enabled_actions(cfg)
    return enabled is None or int(action) in enabled


def _enabled_movement_actions(cfg: SimulationConfig) -> list[Action]:
    moves: list[Action] = []
    for action in MOVE_DELTAS:
        if (not cfg.actions.diagonal_movement_enabled) and action in DIAGONAL_MOVES:
            continue
        if _action_is_enabled(action, cfg):
            moves.append(action)
    return moves


def _enabled_non_movement_actions(cfg: SimulationConfig) -> list[Action]:
    move_ids = {int(a) for a in MOVE_DELTAS}
    out = []
    for action in Action:
        if int(action) not in move_ids and _action_is_enabled(action, cfg):
            out.append(action)
    if Action.REST not in out:
        out.insert(0, Action.REST)
    return out


def _logsumexp(values: Sequence[float] | np.ndarray) -> float:
    """Stable scalar log-sum-exp."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return -1.0e9
    m = np.max(v)
    if not np.isfinite(m):
        return -1.0e9
    return float(m + np.log(np.exp(v - m).sum()))


def _softmax1d(logits: Sequence[float] | np.ndarray, temperature: float, eps: float) -> np.ndarray:
    """Stable one-dimensional softmax with fallback uniform repair."""
    z = np.asarray(logits, dtype=np.float64) / max(float(temperature), float(eps))
    if z.size == 0:
        return np.zeros((0,), dtype=np.float64)
    z = z - np.max(z)
    p = np.exp(z)
    s = p.sum()
    if not np.isfinite(s) or s <= eps:
        return np.full_like(p, 1.0 / p.size, dtype=np.float64)
    return cast(np.ndarray, p / s)


def _base_compute_action_logits(
    state: WorldState,
    utilities: np.ndarray,
    authority: np.ndarray,
    parent_bias: np.ndarray,
    cfg: SimulationConfig,
) -> np.ndarray:
    """Combine utility, authority, integration, and top-down bias into logits.

    Advanced mode additionally applies entropy-dependent temperature diagnostics,
    but it does not collapse to argmax. Later actualization samples from the
    normalized probabilities when ``cfg.actions.stochastic`` is true.
    """
    util = _validate_action_tensor(state, utilities, "utilities")
    auth = np.clip(_validate_action_tensor(state, authority, "authority"), 0.0, 1.0)
    bias = _validate_action_tensor(state, parent_bias, "parent_bias")

    logits = cfg.actions.beta * util
    positive = auth > 0.0
    logits = np.where(
        positive,
        logits + np.log(np.maximum(auth, cfg.actions.epsilon)),
        -1.0e9,
    )

    integration_bias = np.zeros((len(Action),), dtype=DEFAULT_FLOAT_DTYPE)
    integration_bias[int(Action.INTEGRATE)] = 0.50
    integration_bias[int(Action.COMMUNICATE)] = 0.20
    integration_bias[int(Action.REPAIR)] = 0.20
    integration_bias[int(Action.SENSE)] = 0.05

    logits += np.clip(state.integration, 0.0, 1.0)[..., None] * integration_bias[None, None, :]
    logits += np.clip(bias, -cfg.topdown.max_parent_control, cfg.topdown.max_parent_control)

    if getattr(cfg.possibility, "advanced_enabled", False):
        from owl.core.advanced import action_entropy, ensure_advanced_fields

        ensure_advanced_fields(state, cfg)
        entropy = action_entropy(state.possibility, cfg.actions.epsilon)
        tmin = np.float32(cfg.possibility.entropy_temperature_min)
        tmax = np.float32(cfg.possibility.entropy_temperature_max)
        temperature = tmin + (tmax - tmin) * entropy
        logits = logits / np.maximum(temperature[..., None], cfg.actions.epsilon)

    dead = (state.health <= 0.0) | state.obstacle
    if np.any(dead):
        logits[dead, :] = -1.0e9
        logits[dead, int(Action.REST)] = 0.0

    if not np.all(np.isfinite(logits)):
        raise ValueError("computed action logits contain non-finite values")

    if isinstance(state.last_logits, np.ndarray) and state.last_logits.shape == logits.shape:
        state.last_logits[...] = logits.astype(state.last_logits.dtype, copy=False)

    return logits.astype(DEFAULT_FLOAT_DTYPE, copy=False)


def movement_direction_probabilities(
    state: WorldState,
    logits: np.ndarray,
    y: int,
    x: int,
    cfg: SimulationConfig,
) -> tuple[np.ndarray, list[Action]]:
    """Return utility-weighted probabilities over enabled movement directions.

    The score combines already-computed action logits with local neighbor
    survival lookahead, inertia, and immediate-reversal suppression.
    """
    h, w = field_shape(state)
    y = int(y)
    x = int(x)
    if not (0 <= y < h and 0 <= x < w):
        raise ValueError(f"cell {(y, x)} outside field shape {(h, w)}")
    enabled_moves = _enabled_movement_actions(cfg)
    if not enabled_moves:
        return np.zeros((0,), dtype=np.float64), []

    resource_norm = np.clip(
        float(state.resource[y, x])
        / max(float(cfg.resources.max_resource), float(cfg.actions.epsilon)),
        0.0,
        1.0,
    )
    hunger = np.clip(
        (float(cfg.actions.movement_hunger_target) - resource_norm)
        / max(float(cfg.actions.movement_hunger_target), float(cfg.actions.epsilon)),
        0.0,
        1.0,
    )

    last = None
    if isinstance(state.last_movement_action, np.ndarray) and state.last_movement_action.shape == (
        h,
        w,
    ):
        last = int(state.last_movement_action[y, x])

    scores: list[float] = []
    for action in enabled_moves:
        dy, dx = MOVE_DELTAS[action]
        ny = (y + dy) % h if cfg.world.boundary_mode == "toroidal" else y + dy
        nx = (x + dx) % w if cfg.world.boundary_mode == "toroidal" else x + dx
        in_bounds = 0 <= ny < h and 0 <= nx < w
        if not in_bounds:
            blocked = 1.0
            nfood = 0.0
            ntoxin = 1.0
        else:
            blocked = float(
                state.obstacle[ny, nx] or state.occupancy[ny, nx] >= 0 or state.health[ny, nx] > 0.0
            )
            nfood = float(np.clip(state.food[ny, nx], 0.0, 1.0))
            ntoxin = float(np.clip(state.toxin[ny, nx], 0.0, 1.0))

        score = float(logits[y, x, int(action)])
        score += float(cfg.actions.movement_food_weight) * hunger * nfood
        score -= float(cfg.actions.movement_toxin_weight) * ntoxin
        score -= float(cfg.actions.movement_crowding_weight) * blocked
        if last is not None:
            if last == int(action):
                score += float(cfg.actions.movement_persistence_bonus)
            if last == int(REVERSE_MOVE_ACTION[action]):
                score -= float(cfg.actions.movement_reverse_penalty)
        scores.append(score)

    probs = _softmax1d(scores, cfg.actions.movement_temperature, cfg.actions.epsilon)
    return probs.astype(np.float64, copy=False), enabled_moves


def _legacy_sample_actions_with_movement_macro(
    state: WorldState,
    logits: np.ndarray,
    authority: np.ndarray,
    rng: np.random.Generator,
    cfg: SimulationConfig,
) -> np.ndarray:
    """Sample actions with a MOVE macro that prevents split-family suppression.

    Stage 1 samples among all non-movement actions plus one macro MOVE option
    whose logit is log-sum-exp of enabled movement logits. Stage 2 samples a
    concrete movement direction with survival/environment/inertia weighting.
    """
    if rng is None:
        raise ValueError("rng must be an explicit np.random.Generator")
    logits = _validate_action_tensor(state, logits, "logits")
    auth = np.clip(_validate_action_tensor(state, authority, "authority"), 0.0, 1.0)
    h, w, _ = action_shape(state)
    out = np.full((h, w), int(Action.REST), dtype=DEFAULT_READOUT_DTYPE)

    alive_positions = np.argwhere((state.health > 0.0) & (~state.obstacle))
    enabled_moves = _enabled_movement_actions(cfg)
    enabled_non_moves = _enabled_non_movement_actions(cfg)
    if not enabled_moves:
        probs = softmax_stable(logits, axis=-1, epsilon=cfg.actions.epsilon)
        return sample_actions(probs, rng)

    move_indices = np.array([int(a) for a in enabled_moves], dtype=np.int64)

    for y_i, x_i in alive_positions:
        y = int(y_i)
        x = int(x_i)
        allowed_non = [
            action
            for action in enabled_non_moves
            if auth[y, x, int(action)] > 0.0 and np.isfinite(logits[y, x, int(action)])
        ]
        move_allowed = auth[y, x, move_indices] > 0.0
        group_labels: list[Action | str] = list(allowed_non)
        group_logits: list[float] = [float(logits[y, x, int(action)]) for action in allowed_non]
        if np.any(move_allowed):
            group_labels.append("MOVE")
            raw_move = _logsumexp(logits[y, x, move_indices[move_allowed]])
            # Normalize part of the pure family-size boost. This preserves the
            # split-family fix while preventing 8 directions from dominating all
            # non-movement actions solely because there are many of them.
            n_allowed = max(int(np.count_nonzero(move_allowed)), 1)
            raw_move -= float(cfg.actions.movement_macro_normalization) * float(np.log(n_allowed))
            group_logits.append(raw_move)

        if not group_labels:
            out[y, x] = int(Action.REST)
            continue

        group_p = _softmax1d(group_logits, cfg.actions.action_temperature, cfg.actions.epsilon)
        chosen_idx = int(rng.choice(len(group_labels), p=group_p))
        chosen = group_labels[chosen_idx]
        if chosen == "MOVE":
            dir_p, dir_actions = movement_direction_probabilities(state, logits, y, x, cfg)
            if len(dir_actions) == 0:
                out[y, x] = int(Action.REST)
            else:
                out[y, x] = int(rng.choice([int(a) for a in dir_actions], p=dir_p))
        else:
            out[y, x] = int(chosen)

    return out


def actualize_actions(
    state: WorldState,
    utilities: np.ndarray,
    authority: np.ndarray,
    parent_bias: np.ndarray,
    rng: np.random.Generator,
    cfg: SimulationConfig,
) -> None:
    """Update ``state.possibility`` and ``state.readout`` in-place.

    In stochastic ecological mode, readouts are sampled from utility/authority
    weighted probabilities. When configured, movement is treated as a macro action
    so the total movement family can win even if no single direction is largest.
    """
    if rng is None:
        raise ValueError("rng must be an explicit np.random.Generator")

    logits = compute_action_logits(state, utilities, authority, parent_bias, cfg)
    probabilities = softmax_stable(logits, axis=-1, epsilon=cfg.actions.epsilon)

    h, w, k = action_shape(state)
    if probabilities.shape != (h, w, k):
        raise ValueError(
            f"softmax returned wrong shape {probabilities.shape}, expected {(h, w, k)}"
        )

    dead = (state.health <= 0.0) | state.obstacle
    if np.any(dead):
        probabilities[dead, :] = 0.0
        probabilities[dead, int(Action.REST)] = 1.0

    probabilities = normalize_last_axis(probabilities, epsilon=cfg.actions.epsilon)
    state.possibility[...] = probabilities.astype(state.possibility.dtype, copy=False)

    if (
        isinstance(state.last_action_probabilities, np.ndarray)
        and state.last_action_probabilities.shape == probabilities.shape
    ):
        state.last_action_probabilities[...] = probabilities.astype(
            state.last_action_probabilities.dtype, copy=False
        )

    if cfg.actions.stochastic:
        if cfg.actions.utility_weighted_sampling and cfg.actions.movement_macro_enabled:
            readout = sample_actions_with_movement_macro(state, logits, authority, rng, cfg)
        else:
            readout = sample_actions(state.possibility, rng)
    else:
        readout = deterministic_actions(state.possibility)

    readout = np.asarray(readout, dtype=DEFAULT_READOUT_DTYPE)
    if readout.shape != (h, w):
        raise ValueError(f"readout must have shape {(h, w)}, got {readout.shape}")
    readout[dead] = int(Action.REST)
    state.readout[...] = readout

    from owl.core.advanced import cooldown_decay, ensure_advanced_fields

    ensure_advanced_fields(state, cfg)
    cooldown_decay(state, state.readout)


def sample_actions(probabilities: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample one action per cell from a probability cube."""
    if rng is None:
        raise ValueError("rng must be an explicit np.random.Generator")

    probs = np.asarray(probabilities, dtype=np.float32)
    if probs.ndim != 3:
        raise ValueError(
            f"probabilities must have shape (height, width, actions), got {probs.shape}"
        )
    if probs.shape[-1] <= 0:
        raise ValueError("probabilities action axis must be nonempty")
    if not np.all(np.isfinite(probs)):
        raise ValueError("probabilities must contain only finite values")

    probs = normalize_last_axis(probs)
    random_values = rng.random(probs.shape[:2], dtype=np.float32)
    return sample_categorical_grid(probs, random_values).astype(DEFAULT_READOUT_DTYPE, copy=False)


def deterministic_actions(probabilities: np.ndarray) -> np.ndarray:
    """Select the maximum-probability action for every cell."""
    probs = np.asarray(probabilities, dtype=np.float32)
    if probs.ndim != 3:
        raise ValueError(
            f"probabilities must have shape (height, width, actions), got {probs.shape}"
        )
    if probs.shape[-1] <= 0:
        raise ValueError("probabilities action axis must be nonempty")
    if not np.all(np.isfinite(probs)):
        raise ValueError("probabilities must contain only finite values")
    return cast(np.ndarray, np.argmax(probs, axis=-1).astype(DEFAULT_READOUT_DTYPE))


# --- Decision-homeostasis actualization overrides ----------------------------
_decision_base_compute_action_logits = _base_compute_action_logits


def _apply_homeostatic_precision(
    state: WorldState, logits: np.ndarray, authority: np.ndarray, cfg: SimulationConfig
) -> np.ndarray:
    """Raise probability mass on the best feasible action under urgency."""
    if not getattr(cfg.decision_homeostasis, "enabled", False):
        return logits
    urgency = getattr(state, "last_decision_urgency", None)
    if not isinstance(urgency, np.ndarray) or urgency.shape != field_shape(state):
        return logits
    out = np.asarray(logits, dtype=np.float32).copy()
    auth = np.clip(authority, 0.0, 1.0)
    feasible = auth > float(cfg.decision_homeostasis.authority_floor)
    finite = np.isfinite(out) & (out > -1e8) & feasible
    urgent = urgency > float(cfg.decision_homeostasis.urgent_threshold)
    if not np.any(urgent):
        return out
    masked = np.where(finite, out, -1e9)
    best = np.argmax(masked, axis=-1)
    h, w = field_shape(state)
    yy, xx = np.indices((h, w))
    precision = float(cfg.decision_homeostasis.emergency_precision_min) + (
        float(cfg.decision_homeostasis.emergency_precision_max)
        - float(cfg.decision_homeostasis.emergency_precision_min)
    ) * np.clip(urgency, 0.0, 1.0)
    bonus = precision * urgent.astype(np.float32)
    out[yy, xx, best] += bonus
    # Never boost infeasible cells; REST remains the fallback.
    out[~finite.any(axis=-1), :] = -1e9
    out[~finite.any(axis=-1), int(Action.REST)] = 0.0
    return out.astype(DEFAULT_FLOAT_DTYPE, copy=False)


def compute_action_logits(
    state: WorldState,
    utilities: np.ndarray,
    authority: np.ndarray,
    parent_bias: np.ndarray,
    cfg: SimulationConfig,
) -> np.ndarray:
    """Compute logits and apply authority-coupled homeostatic precision."""
    logits = _decision_base_compute_action_logits(state, utilities, authority, parent_bias, cfg)
    adjusted = _apply_homeostatic_precision(state, logits, authority, cfg)
    if isinstance(state.last_logits, np.ndarray) and state.last_logits.shape == adjusted.shape:
        state.last_logits[...] = adjusted.astype(state.last_logits.dtype, copy=False)
    return adjusted.astype(DEFAULT_FLOAT_DTYPE, copy=False)


def sample_actions_with_movement_macro(
    state: WorldState,
    logits: np.ndarray,
    authority: np.ndarray,
    rng: np.random.Generator,
    cfg: SimulationConfig,
) -> np.ndarray:
    """Sample actions with a MOVE macro while recording macro probabilities."""
    if rng is None:
        raise ValueError("rng must be an explicit np.random.Generator")
    logits = _validate_action_tensor(state, logits, "logits")
    auth = np.clip(_validate_action_tensor(state, authority, "authority"), 0.0, 1.0)
    h, w, actions = action_shape(state)
    out = np.full((h, w), int(Action.REST), dtype=DEFAULT_READOUT_DTYPE)
    from owl.core.advanced import ensure_advanced_fields

    ensure_advanced_fields(state, cfg)
    assert state.last_macro_probabilities is not None
    assert state.last_chosen_macro is not None
    if isinstance(state.last_macro_probabilities, np.ndarray):
        state.last_macro_probabilities.fill(0.0)
    if isinstance(state.last_chosen_macro, np.ndarray):
        state.last_chosen_macro.fill(int(Action.REST))

    alive_positions = np.argwhere((state.health > 0.0) & (~state.obstacle))
    enabled_moves = _enabled_movement_actions(cfg)
    enabled_non_moves = _enabled_non_movement_actions(cfg)
    if not enabled_moves:
        probs = softmax_stable(logits, axis=-1, epsilon=cfg.actions.epsilon)
        sampled = sample_actions(probs, rng)
        state.last_chosen_macro[...] = sampled
        return sampled

    move_indices = np.array([int(a) for a in enabled_moves], dtype=np.int64)
    move_macro_index = actions

    for y_i, x_i in alive_positions:
        y = int(y_i)
        x = int(x_i)
        allowed_non = [
            action
            for action in enabled_non_moves
            if auth[y, x, int(action)] > 0.0
            and np.isfinite(logits[y, x, int(action)])
            and logits[y, x, int(action)] > -1e8
        ]
        move_allowed = auth[y, x, move_indices] > 0.0
        group_labels: list[Action | str] = list(allowed_non)
        group_logits: list[float] = [float(logits[y, x, int(action)]) for action in allowed_non]
        if np.any(move_allowed):
            group_labels.append("MOVE")
            raw_move = _logsumexp(logits[y, x, move_indices[move_allowed]])
            n_allowed = max(int(np.count_nonzero(move_allowed)), 1)
            raw_move -= float(cfg.actions.movement_macro_normalization) * float(np.log(n_allowed))
            # Urgent survival can make movement a macro winner only when movement
            # improves local viability.
            if isinstance(state.last_survival_value, np.ndarray):
                raw_move += float(
                    np.max(state.last_survival_value[y, x, move_indices[move_allowed]])
                )
            group_logits.append(raw_move)

        if not group_labels:
            out[y, x] = int(Action.REST)
            continue

        temperature = float(cfg.actions.action_temperature)
        if getattr(cfg.decision_homeostasis, "enabled", False) and isinstance(
            state.last_decision_urgency, np.ndarray
        ):
            urgency = float(np.clip(state.last_decision_urgency[y, x], 0.0, 1.0))
            precision = (
                float(cfg.decision_homeostasis.safe_precision)
                + (
                    float(cfg.decision_homeostasis.emergency_precision_max)
                    - float(cfg.decision_homeostasis.safe_precision)
                )
                * urgency
            )
            temperature = max(temperature / max(precision, 1e-6), float(cfg.actions.epsilon))
        group_p = _softmax1d(group_logits, temperature, cfg.actions.epsilon)

        # Optional optimal-action mass: mix probability toward best feasible macro
        # without becoming deterministic.
        if getattr(cfg.decision_homeostasis, "enabled", False) and isinstance(
            state.last_decision_urgency, np.ndarray
        ):
            urgency = float(np.clip(state.last_decision_urgency[y, x], 0.0, 1.0))
            if urgency > float(cfg.decision_homeostasis.urgent_threshold):
                best_idx = int(np.argmax(group_logits))
                force = float(cfg.decision_homeostasis.forced_optimal_probability) * urgency
                force = float(
                    np.clip(force, 0.0, float(cfg.decision_homeostasis.forced_optimal_probability))
                )
                group_p *= 1.0 - force
                group_p[best_idx] += force
                group_p = group_p / max(float(np.sum(group_p)), float(cfg.actions.epsilon))

        for label, prob in zip(group_labels, group_p, strict=True):
            if isinstance(label, str):
                if label == "MOVE" and isinstance(state.last_macro_probabilities, np.ndarray):
                    state.last_macro_probabilities[y, x, move_macro_index] = np.float32(prob)
            else:
                if isinstance(state.last_macro_probabilities, np.ndarray):
                    state.last_macro_probabilities[y, x, int(label)] = np.float32(prob)

        chosen_idx = int(rng.choice(len(group_labels), p=group_p))
        chosen = group_labels[chosen_idx]
        if chosen == "MOVE":
            if isinstance(state.last_chosen_macro, np.ndarray):
                state.last_chosen_macro[y, x] = move_macro_index
            dir_p, dir_actions = movement_direction_probabilities(state, logits, y, x, cfg)
            if len(dir_actions) == 0:
                out[y, x] = int(Action.REST)
            else:
                out[y, x] = int(rng.choice([int(a) for a in dir_actions], p=dir_p))
        else:
            out[y, x] = int(chosen)
            if isinstance(state.last_chosen_macro, np.ndarray):
                state.last_chosen_macro[y, x] = int(chosen)

    return out
