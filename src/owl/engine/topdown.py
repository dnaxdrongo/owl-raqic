(
    """Patch/global intention and bounded top-down modulation.

Top-down influence in Observer-Window Life is deliberately weak: parent windows
produce bounded action-logit and threshold biases, but they never overwrite child
readouts.  This module is therefore a policy-bias layer, not a controller.
"""
    ""
)

from __future__ import annotations

from typing import cast

import numpy as np

from owl.core.actions import Action, GlobalIntention, PatchIntention, SignalChannel
from owl.core.config import SimulationConfig
from owl.core.constants import DEFAULT_FLOAT_DTYPE, DEFAULT_INT_DTYPE
from owl.core.state import GlobalState, PatchState, WorldState, field_shape
from owl.engine.aggregation import upsample_patch_field


def _channel(
    field: np.ndarray, channel: SignalChannel, cfg: SimulationConfig
) -> np.ndarray | float:
    """Return a channel slice if configured, otherwise zero.

    This keeps top-down logic safe for reduced-channel test configurations while
    preserving the eight-channel communication grammar used by the baseline.
    """
    idx = int(channel)
    if field.ndim == 3:
        if idx < min(cfg.communication.num_channels, field.shape[-1]):
            return field[..., idx]
        return 0.0
    if field.ndim == 1:
        if idx < min(cfg.communication.num_channels, field.shape[0]):
            return float(field[idx])
        return 0.0
    raise ValueError(f"signal field must be 1D or 3D, got shape {field.shape}")


def _validate_patch_state(patches: PatchState, cfg: SimulationConfig) -> tuple[int, int]:
    """Validate patch state shapes needed by top-down policy functions."""
    shape = patches.integration.shape
    if len(shape) != 2:
        raise ValueError(f"patches.integration must be 2D, got shape {shape}")

    hp, wp = shape
    actions = len(Action)
    channels = cfg.communication.num_channels

    expected_action = (hp, wp, actions)
    expected_channel = (hp, wp, channels)

    if patches.possibility.shape != expected_action:
        raise ValueError(
            f"patches.possibility must have shape {expected_action}, "
            f"got {patches.possibility.shape}"
        )
    if patches.policy_bias.shape != expected_action:
        raise ValueError(
            f"patches.policy_bias must have shape {expected_action}, "
            f"got {patches.policy_bias.shape}"
        )
    if patches.signal_pressure.shape != expected_channel:
        raise ValueError(
            f"patches.signal_pressure must have shape {expected_channel}, "
            f"got {patches.signal_pressure.shape}"
        )
    if patches.intention.shape != shape:
        raise ValueError(
            f"patches.intention must have shape {shape}, got {patches.intention.shape}"
        )
    return hp, wp


def _clip_bias(bias: np.ndarray, cfg: SimulationConfig) -> np.ndarray:
    """Return float32 bias clipped to the top-down control bound."""
    limit = float(cfg.topdown.max_parent_control)
    return cast(np.ndarray, np.clip(bias, -limit, limit).astype(DEFAULT_FLOAT_DTYPE))


def _legacy_compute_patch_intention(patches: PatchState, cfg: SimulationConfig) -> None:
    """Compute patch-level intentions from signal pressure and patch state.

    Parameters
    ----------
    patches:
        Patch-level observer-window state. This function mutates only
        ``patches.intention``.
    cfg:
        Simulation coefficients.

    Notes
    -----
    Patch intentions are regional summaries, not commands. They are later
    converted into bounded action-bias tensors by :func:`patch_policy_to_bias`.
    Empty/dead patches are set to ``PatchIntention.REST``.
    """
    hp, wp = _validate_patch_state(patches, cfg)

    integration = np.clip(patches.integration, 0.0, 1.0)
    resource = np.clip(patches.resource, 0.0, 1.0)
    health = np.clip(patches.health, 0.0, 1.0)
    boundary = np.clip(patches.boundary, 0.0, 1.0)
    synchrony = np.clip(patches.synchrony, 0.0, 1.0)
    coherence = np.clip(patches.coherence, 0.0, 1.0)

    food = _channel(patches.signal_pressure, SignalChannel.FOOD, cfg)
    danger = _channel(patches.signal_pressure, SignalChannel.DANGER, cfg)
    threat = _channel(patches.signal_pressure, SignalChannel.THREAT, cfg)
    coord = _channel(patches.signal_pressure, SignalChannel.COORDINATION, cfg)
    distress = _channel(patches.signal_pressure, SignalChannel.DISTRESS, cfg)
    repro = _channel(patches.signal_pressure, SignalChannel.REPRODUCTION, cfg)
    territory = _channel(patches.signal_pressure, SignalChannel.TERRITORY, cfg)
    integration_signal = _channel(patches.signal_pressure, SignalChannel.INTEGRATION, cfg)

    # Scores are bounded heuristics for policy summaries. They use physical
    # pressures, communication pressure, and fractal/mosaic integration state.
    scores = np.zeros((hp, wp, len(PatchIntention)), dtype=DEFAULT_FLOAT_DTYPE)
    scores[..., int(PatchIntention.REST)] = 0.20 * resource + 0.20 * health + 0.10 * integration
    scores[..., int(PatchIntention.SEEK_FOOD)] = 0.80 * food + 0.35 * (1.0 - resource)
    scores[..., int(PatchIntention.AVOID_DANGER)] = 0.75 * danger + 0.50 * threat + 0.20 * distress
    scores[..., int(PatchIntention.COORDINATE)] = (
        0.55 * coord
        + 0.40 * integration_signal
        + 0.35 * (1.0 - integration)
        + 0.15 * (1.0 - synchrony)
        + 0.15 * (1.0 - coherence)
    )
    scores[..., int(PatchIntention.DEFEND)] = (
        0.55 * threat + 0.35 * territory + 0.20 * (1.0 - boundary)
    )
    scores[..., int(PatchIntention.REPRODUCE)] = (
        0.50 * repro + 0.60 * resource * health * boundary * np.maximum(integration, 0.05)
    )
    scores[..., int(PatchIntention.REPAIR)] = (
        0.65 * (1.0 - boundary) + 0.35 * (1.0 - health) + 0.25 * distress
    )
    scores[..., int(PatchIntention.EXPLORE)] = (
        0.20 * (1.0 - food) + 0.15 * resource + 0.10 * (1.0 - danger)
    )

    intention = np.argmax(scores, axis=-1).astype(DEFAULT_INT_DTYPE)
    inactive = health <= 0.0
    intention[inactive] = int(PatchIntention.REST)
    patches.intention[...] = intention


def _base_patch_policy_to_bias(patches: PatchState, cfg: SimulationConfig) -> np.ndarray:
    """Convert patch intentions into bounded action-bias vectors.

    Parameters
    ----------
    patches:
        Patch state with computed ``patches.intention``. This function mutates
        ``patches.policy_bias`` to match the returned array.
    cfg:
        Top-down coefficients.

    Returns
    -------
    np.ndarray
        Patch-level action-bias tensor with shape
        ``(patch_height, patch_width, len(Action))`` and values clipped to
        ``[-cfg.topdown.max_parent_control, cfg.topdown.max_parent_control]``.
    """
    hp, wp = _validate_patch_state(patches, cfg)
    bias = np.zeros((hp, wp, len(Action)), dtype=DEFAULT_FLOAT_DTYPE)
    intent = patches.intention

    # SEEK_FOOD: weakly favor sensing, movement, feeding, and sharing food info.
    seek = intent == int(PatchIntention.SEEK_FOOD)
    for action in (Action.SENSE, Action.MOVE_N, Action.MOVE_S, Action.MOVE_E, Action.MOVE_W):
        bias[..., int(action)] += 0.08 * seek
    bias[..., int(Action.FEED)] += 0.35 * seek
    bias[..., int(Action.COMMUNICATE)] += 0.10 * seek

    # AVOID_DANGER: favor caution and repair; future utility decides direction.
    avoid = intent == int(PatchIntention.AVOID_DANGER)
    bias[..., int(Action.FLEE)] += 0.35 * avoid
    bias[..., int(Action.REPAIR)] += 0.25 * avoid
    bias[..., int(Action.COMMUNICATE)] += 0.20 * avoid
    bias[..., int(Action.INHIBIT)] += 0.12 * avoid

    # COORDINATE: alignment, signaling, and integration are the intended modes.
    coord = intent == int(PatchIntention.COORDINATE)
    bias[..., int(Action.COMMUNICATE)] += 0.30 * coord
    bias[..., int(Action.INTEGRATE)] += 0.40 * coord
    bias[..., int(Action.SENSE)] += 0.10 * coord

    # DEFEND: bounded defensive pressure, not a command to attack.
    defend = intent == int(PatchIntention.DEFEND)
    bias[..., int(Action.INHIBIT)] += 0.30 * defend
    bias[..., int(Action.REPAIR)] += 0.20 * defend
    bias[..., int(Action.COMMUNICATE)] += 0.15 * defend
    bias[..., int(Action.INGEST)] += 0.08 * defend

    # REPRODUCE: signal readiness and make reproduction slightly more likely.
    repro = intent == int(PatchIntention.REPRODUCE)
    bias[..., int(Action.REPRODUCE)] += 0.35 * repro
    bias[..., int(Action.COMMUNICATE)] += 0.12 * repro
    bias[..., int(Action.REST)] += 0.05 * repro

    # REPAIR: stabilize boundary/health and reduce unnecessary aggression.
    repair = intent == int(PatchIntention.REPAIR)
    bias[..., int(Action.REPAIR)] += 0.40 * repair
    bias[..., int(Action.REST)] += 0.12 * repair
    bias[..., int(Action.INTEGRATE)] += 0.10 * repair
    bias[..., int(Action.INGEST)] -= 0.10 * repair

    # EXPLORE: weak movement/sensing pressure.
    explore = intent == int(PatchIntention.EXPLORE)
    for action in (
        Action.SENSE,
        Action.MOVE_N,
        Action.MOVE_S,
        Action.MOVE_E,
        Action.MOVE_W,
        Action.MOVE_NE,
        Action.MOVE_NW,
        Action.MOVE_SE,
        Action.MOVE_SW,
    ):
        bias[..., int(action)] += 0.08 * explore

    # REST: a small stabilizing pressure only when the patch is already viable.
    rest = intent == int(PatchIntention.REST)
    bias[..., int(Action.REST)] += 0.10 * rest
    bias[..., int(Action.INTEGRATE)] += 0.05 * rest

    scale = (
        cfg.topdown.lambda_action_bias
        * np.clip(patches.integration, 0.0, 1.0)
        * np.clip(patches.health, 0.0, 1.0)
    ).astype(np.float32)
    bias *= scale[..., None]
    bias = _clip_bias(bias, cfg)
    patches.policy_bias[...] = bias
    return bias


def _legacy_compute_global_intention(global_state: GlobalState, cfg: SimulationConfig) -> int:
    """Compute an apex/global intention from global summary fields.

    Parameters
    ----------
    global_state:
        Apex observer-window summary. This function does not mutate it.
    cfg:
        Simulation coefficients.

    Returns
    -------
    int
        A member value of :class:`GlobalIntention`.
    """
    signal = np.asarray(global_state.signal_pressure, dtype=np.float32)
    if signal.ndim != 1:
        raise ValueError(f"global_state.signal_pressure must be 1D, got shape {signal.shape}")

    food = _channel(signal, SignalChannel.FOOD, cfg)
    danger = _channel(signal, SignalChannel.DANGER, cfg)
    threat = _channel(signal, SignalChannel.THREAT, cfg)
    coord = _channel(signal, SignalChannel.COORDINATION, cfg)
    distress = _channel(signal, SignalChannel.DISTRESS, cfg)
    repro = _channel(signal, SignalChannel.REPRODUCTION, cfg)
    integration_signal = _channel(signal, SignalChannel.INTEGRATION, cfg)

    integration = float(np.clip(global_state.integration, 0.0, 1.0))
    fragmentation = float(np.clip(global_state.fragmentation, 0.0, 1.0))
    diversity = float(np.clip(global_state.diversity, 0.0, 1.0))
    complexity = float(np.clip(global_state.complexity, 0.0, 1.0))

    scores = np.zeros((len(GlobalIntention),), dtype=DEFAULT_FLOAT_DTYPE)
    scores[int(GlobalIntention.REST)] = 0.20 * integration + 0.20 * (1.0 - fragmentation)
    scores[int(GlobalIntention.EXPAND)] = 0.30 * diversity + 0.25 * food + 0.15 * complexity
    scores[int(GlobalIntention.CONSERVE)] = 0.30 * (1.0 - integration) + 0.25 * distress
    scores[int(GlobalIntention.SEEK_FOOD)] = 0.70 * food + 0.20 * (1.0 - integration)
    scores[int(GlobalIntention.AVOID_THREAT)] = 0.75 * danger + 0.50 * threat
    scores[int(GlobalIntention.COORDINATE)] = (
        0.55 * coord + 0.45 * integration_signal + 0.40 * fragmentation
    )
    scores[int(GlobalIntention.REPRODUCE)] = 0.45 * repro + 0.30 * integration + 0.20 * diversity
    scores[int(GlobalIntention.DEFEND)] = 0.55 * threat + 0.30 * danger
    scores[int(GlobalIntention.EXPLORE)] = 0.25 * diversity + 0.20 * (1.0 - food)
    scores[int(GlobalIntention.REPAIR)] = (
        0.45 * distress + 0.25 * fragmentation + 0.20 * (1.0 - integration)
    )

    return int(np.argmax(scores))


def _legacy_global_policy_to_bias(global_state: GlobalState, cfg: SimulationConfig) -> np.ndarray:
    """Convert global intention into a weak action-policy bias.

    Parameters
    ----------
    global_state:
        Apex summary. ``global_state.policy_bias`` is updated to the returned
        vector.
    cfg:
        Top-down coefficients.

    Returns
    -------
    np.ndarray
        Float32 vector with shape ``(len(Action),)``. This vector is deliberately
        weaker than patch-level policy because the apex should bias rather than
        micromanage lower windows.
    """
    bias = np.zeros((len(Action),), dtype=DEFAULT_FLOAT_DTYPE)
    try:
        intention = GlobalIntention(int(global_state.intention))
    except ValueError:
        intention = GlobalIntention(compute_global_intention(global_state, cfg))

    if intention == GlobalIntention.SEEK_FOOD:
        bias[int(Action.SENSE)] += 0.08
        bias[int(Action.FEED)] += 0.12
        bias[int(Action.COMMUNICATE)] += 0.05
    elif intention == GlobalIntention.AVOID_THREAT:
        bias[int(Action.FLEE)] += 0.14
        bias[int(Action.REPAIR)] += 0.10
        bias[int(Action.COMMUNICATE)] += 0.06
    elif intention == GlobalIntention.COORDINATE:
        bias[int(Action.INTEGRATE)] += 0.16
        bias[int(Action.COMMUNICATE)] += 0.12
        bias[int(Action.SENSE)] += 0.05
    elif intention == GlobalIntention.REPRODUCE:
        bias[int(Action.REPRODUCE)] += 0.12
        bias[int(Action.COMMUNICATE)] += 0.05
    elif intention == GlobalIntention.DEFEND:
        bias[int(Action.INHIBIT)] += 0.12
        bias[int(Action.REPAIR)] += 0.07
        bias[int(Action.INGEST)] += 0.04
    elif intention == GlobalIntention.EXPLORE:
        for action in (Action.SENSE, Action.MOVE_N, Action.MOVE_S, Action.MOVE_E, Action.MOVE_W):
            bias[int(action)] += 0.06
    elif intention == GlobalIntention.REPAIR:
        bias[int(Action.REPAIR)] += 0.14
        bias[int(Action.REST)] += 0.04
    elif intention == GlobalIntention.CONSERVE:
        bias[int(Action.REST)] += 0.08
        bias[int(Action.REPAIR)] += 0.06
        bias[int(Action.INGEST)] -= 0.04
    elif intention == GlobalIntention.EXPAND:
        for action in (Action.SENSE, Action.MOVE_N, Action.MOVE_S, Action.MOVE_E, Action.MOVE_W):
            bias[int(action)] += 0.04
        bias[int(Action.REPRODUCE)] += 0.04
    else:  # REST
        bias[int(Action.REST)] += 0.05
        bias[int(Action.INTEGRATE)] += 0.03

    scale = (
        0.50 * cfg.topdown.lambda_action_bias * float(np.clip(global_state.integration, 0.0, 1.0))
    )
    bias = _clip_bias(bias * scale, cfg)
    global_state.policy_bias[...] = bias
    return bias


def apply_threshold_modulation(
    state: WorldState, patches: PatchState, cfg: SimulationConfig
) -> None:
    """Apply bounded top-down threshold modulation to child cells.

    Parameters
    ----------
    state:
        Runtime dense state. This function mutates only ``state.threshold``.
    patches:
        Patch-level parent windows with computed intentions.
    cfg:
        Top-down coefficients.

    Notes
    -----
    Threshold modulation is an asymptotic nudge toward intention-dependent target
    thresholds. It does not overwrite readouts and it does not accumulate without
    bound: repeated calls move thresholds toward bounded targets in ``[0, 1]``.
    """
    _validate_patch_state(patches, cfg)

    h, w = field_shape(state)
    patch = int(cfg.world.patch_size)
    expected_patch_shape = (h // patch, w // patch)
    if patches.integration.shape != expected_patch_shape:
        raise ValueError(
            f"patches shape {patches.integration.shape} does not match cell shape {(h, w)} "
            f"and patch_size={patch}"
        )
    if state.threshold.shape != (h, w):
        raise ValueError(f"state.threshold must have shape {(h, w)}, got {state.threshold.shape}")

    intent_cells = upsample_patch_field(patches.intention, patch)
    integration_cells = upsample_patch_field(np.clip(patches.integration, 0.0, 1.0), patch)
    health_cells = upsample_patch_field(np.clip(patches.health, 0.0, 1.0), patch)

    target = np.full((h, w), 0.50, dtype=DEFAULT_FLOAT_DTYPE)
    urgent = (
        (intent_cells == int(PatchIntention.AVOID_DANGER))
        | (intent_cells == int(PatchIntention.DEFEND))
        | (intent_cells == int(PatchIntention.REPAIR))
    )
    active = intent_cells != int(PatchIntention.REST)
    target[active] = 0.40
    target[urgent] = 0.30

    alpha = cfg.topdown.lambda_threshold * integration_cells * health_cells
    alpha = np.clip(alpha, 0.0, cfg.topdown.max_parent_control).astype(DEFAULT_FLOAT_DTYPE)

    alive = state.health > 0.0
    updated = state.threshold.astype(np.float32, copy=True)
    updated[alive] = updated[alive] + alpha[alive] * (target[alive] - updated[alive])
    state.threshold[...] = np.clip(updated, 0.0, 1.0).astype(DEFAULT_FLOAT_DTYPE)


# --- Advanced build overrides ------------------------------------------------
_mvp_patch_policy_to_bias = _base_patch_policy_to_bias
_mvp_apply_threshold_modulation = apply_threshold_modulation


def _advanced_patch_policy_to_bias(patches: PatchState, cfg: SimulationConfig) -> np.ndarray:
    """Convert policy to bias; advanced mode suppresses risky high-error patches."""
    bias = _mvp_patch_policy_to_bias(patches, cfg)
    if getattr(cfg.hierarchy, "predictive_topdown", False) and isinstance(
        patches.prediction_error, np.ndarray
    ):
        error = np.clip(patches.prediction_error, 0.0, 1.0)
        # High prediction error lowers aggressive/nonlocal control and boosts SENSE/REPAIR.
        bias[..., int(Action.INGEST)] -= cfg.hierarchy.prediction_error_weight * error
        bias[..., int(Action.INHIBIT)] -= 0.5 * cfg.hierarchy.prediction_error_weight * error
        bias[..., int(Action.SENSE)] += cfg.hierarchy.prediction_error_weight * error
        bias[..., int(Action.REPAIR)] += 0.5 * cfg.hierarchy.prediction_error_weight * error
        np.clip(bias, -cfg.topdown.max_parent_control, cfg.topdown.max_parent_control, out=bias)
        patches.policy_bias[...] = bias
    return bias


# --- Decision-homeostasis top-down overrides ---------------------------------
def _homeostasis_compute_patch_intention_impl(patches: PatchState, cfg: SimulationConfig) -> None:
    """Compute patch intentions from actual lower-level viability pressures."""
    hp, wp = _validate_patch_state(patches, cfg)
    integration = np.clip(patches.integration, 0.0, 1.0)
    resource = np.clip(patches.resource, 0.0, 1.0)
    health = np.clip(patches.health, 0.0, 1.0)
    boundary = np.clip(patches.boundary, 0.0, 1.0)
    crisis = np.clip(
        getattr(patches, "patch_crisis", np.zeros((hp, wp), dtype=np.float32)), 0.0, 1.0
    )
    carrying = np.clip(
        getattr(patches, "patch_carrying_pressure", np.zeros((hp, wp), dtype=np.float32)), 0.0, 1.0
    )
    food_mean = np.clip(
        getattr(patches, "food_mean", np.ones((hp, wp), dtype=np.float32)), 0.0, 1.0
    )
    starvation = np.clip(
        getattr(patches, "starvation_debt_mean", np.zeros((hp, wp), dtype=np.float32)), 0.0, 1.0
    )
    density = np.clip(
        getattr(patches, "alive_density", np.zeros((hp, wp), dtype=np.float32)), 0.0, 1.0
    )

    food_signal = _channel(patches.signal_pressure, SignalChannel.FOOD, cfg)
    danger = _channel(patches.signal_pressure, SignalChannel.DANGER, cfg)
    threat = _channel(patches.signal_pressure, SignalChannel.THREAT, cfg)
    coord = _channel(patches.signal_pressure, SignalChannel.COORDINATION, cfg)
    distress = _channel(patches.signal_pressure, SignalChannel.DISTRESS, cfg)
    repro = _channel(patches.signal_pressure, SignalChannel.REPRODUCTION, cfg)
    integration_signal = _channel(patches.signal_pressure, SignalChannel.INTEGRATION, cfg)

    scores = np.zeros((hp, wp, len(PatchIntention)), dtype=DEFAULT_FLOAT_DTYPE)
    food_deficit = 1.0 - food_mean
    scores[..., int(PatchIntention.REST)] = 0.15 * resource + 0.10 * health + 0.10 * (1.0 - crisis)
    scores[..., int(PatchIntention.SEEK_FOOD)] = (
        0.75 * food_deficit + 0.70 * starvation + 0.30 * food_signal + 0.10 * crisis
    )
    scores[..., int(PatchIntention.AVOID_DANGER)] = 0.75 * danger + 0.50 * threat + 0.25 * distress
    scores[..., int(PatchIntention.COORDINATE)] = (
        0.55 * coord + 0.45 * integration_signal + 0.25 * crisis + 0.15 * (1.0 - integration)
    )
    scores[..., int(PatchIntention.DEFEND)] = (
        0.55 * threat + 0.25 * danger + 0.25 * (1.0 - boundary)
    )
    repro_safe = np.clip(
        resource * health * boundary * integration * (1.0 - carrying) * (1.0 - crisis), 0.0, 1.0
    )
    scores[..., int(PatchIntention.REPRODUCE)] = (
        0.45 * repro + 0.75 * repro_safe - 0.80 * crisis - 0.50 * carrying
    )
    scores[..., int(PatchIntention.REPAIR)] = (
        0.55 * (1.0 - boundary) + 0.35 * (1.0 - health) + 0.35 * distress + 0.35 * crisis
    )
    scores[..., int(PatchIntention.EXPLORE)] = (
        0.25 * food_deficit * (1.0 - starvation) + 0.15 * density + 0.10 * (1.0 - crisis)
    )

    crisis_mask = crisis >= float(
        getattr(cfg.cross_scale_homeostasis, "patch_crisis_threshold", 1.1)
    )
    if np.any(crisis_mask):
        scores[..., int(PatchIntention.REPRODUCE)] = np.where(
            crisis_mask, -1.0, scores[..., int(PatchIntention.REPRODUCE)]
        )
    intention = np.argmax(scores, axis=-1).astype(DEFAULT_INT_DTYPE)
    inactive = health <= 0.0
    intention[inactive] = int(PatchIntention.REST)
    patches.intention[...] = intention


def patch_policy_to_bias(patches: PatchState, cfg: SimulationConfig) -> np.ndarray:
    """Convert patch intentions to bounded bias with crisis-aware survival pressure."""
    bias = _mvp_patch_policy_to_bias(patches, cfg)
    hp, wp = patches.integration.shape
    crisis = np.clip(
        getattr(patches, "patch_crisis", np.zeros((hp, wp), dtype=np.float32)), 0.0, 1.0
    )
    carrying = np.clip(
        getattr(patches, "patch_carrying_pressure", np.zeros((hp, wp), dtype=np.float32)), 0.0, 1.0
    )
    starvation = np.clip(
        getattr(patches, "starvation_debt_mean", np.zeros((hp, wp), dtype=np.float32)), 0.0, 1.0
    )
    food_mean = np.clip(
        getattr(patches, "food_mean", np.ones((hp, wp), dtype=np.float32)), 0.0, 1.0
    )
    urgent = np.clip(0.50 * crisis + 0.35 * starvation + 0.15 * (1.0 - food_mean), 0.0, 1.0)

    bias[..., int(Action.FEED)] += 0.35 * urgent
    bias[..., int(Action.SENSE)] += 0.16 * urgent
    bias[..., int(Action.REPAIR)] += 0.22 * crisis
    for action in (
        Action.MOVE_N,
        Action.MOVE_S,
        Action.MOVE_E,
        Action.MOVE_W,
        Action.MOVE_NE,
        Action.MOVE_NW,
        Action.MOVE_SE,
        Action.MOVE_SW,
    ):
        bias[..., int(action)] += 0.08 * urgent
    suppression = float(getattr(cfg.cross_scale_homeostasis, "max_reproduction_suppression", 0.95))
    bias[..., int(Action.REPRODUCE)] -= suppression * np.clip(
        0.70 * crisis + 0.50 * carrying, 0.0, 1.0
    )
    if getattr(cfg.hierarchy, "predictive_topdown", False) and isinstance(
        patches.prediction_error, np.ndarray
    ):
        error = np.clip(patches.prediction_error, 0.0, 1.0)
        bias[..., int(Action.SENSE)] += cfg.hierarchy.prediction_error_weight * error
        bias[..., int(Action.REPAIR)] += 0.5 * cfg.hierarchy.prediction_error_weight * error
        bias[..., int(Action.INHIBIT)] -= 0.5 * cfg.hierarchy.prediction_error_weight * error
        bias[..., int(Action.INGEST)] -= cfg.hierarchy.prediction_error_weight * error
    bias = _clip_bias(bias, cfg)
    patches.policy_bias[...] = bias
    return bias


def compute_global_intention(global_state: GlobalState, cfg: SimulationConfig) -> int:
    """Compute apex intention with crisis able to override reproduction."""
    signal = np.asarray(global_state.signal_pressure, dtype=np.float32)
    food = _channel(signal, SignalChannel.FOOD, cfg)
    danger = _channel(signal, SignalChannel.DANGER, cfg)
    threat = _channel(signal, SignalChannel.THREAT, cfg)
    coord = _channel(signal, SignalChannel.COORDINATION, cfg)
    distress = _channel(signal, SignalChannel.DISTRESS, cfg)
    repro = _channel(signal, SignalChannel.REPRODUCTION, cfg)
    integration_signal = _channel(signal, SignalChannel.INTEGRATION, cfg)
    integration = float(np.clip(global_state.integration, 0.0, 1.0))
    fragmentation = float(np.clip(global_state.fragmentation, 0.0, 1.0))
    diversity = float(np.clip(global_state.diversity, 0.0, 1.0))
    crisis = float(np.clip(getattr(global_state, "crisis", 0.0), 0.0, 1.0))
    carrying = float(np.clip(getattr(global_state, "carrying_pressure", 0.0), 0.0, 1.0))
    starv = float(np.clip(getattr(global_state, "starvation_pressure", 0.0), 0.0, 1.0))
    food_deficit = float(np.clip(getattr(global_state, "food_deficit", 0.0), 0.0, 1.0))

    scores = np.zeros((len(GlobalIntention),), dtype=DEFAULT_FLOAT_DTYPE)
    scores[int(GlobalIntention.REST)] = 0.20 * integration + 0.20 * (1.0 - fragmentation)
    scores[int(GlobalIntention.EXPAND)] = 0.20 * diversity + 0.20 * food + 0.10 * (1.0 - crisis)
    scores[int(GlobalIntention.CONSERVE)] = 0.80 * crisis + 0.40 * starv + 0.20 * fragmentation
    scores[int(GlobalIntention.SEEK_FOOD)] = 0.70 * food_deficit + 0.60 * starv + 0.20 * food
    scores[int(GlobalIntention.AVOID_THREAT)] = 0.75 * danger + 0.50 * threat
    scores[int(GlobalIntention.COORDINATE)] = (
        0.45 * coord + 0.35 * integration_signal + 0.30 * fragmentation + 0.20 * crisis
    )
    scores[int(GlobalIntention.REPRODUCE)] = (
        0.60 * repro + 0.35 * integration + 0.25 * diversity
    ) * (
        1.0
        - float(getattr(cfg.cross_scale_homeostasis, "max_reproduction_suppression", 0.95))
        * max(crisis, carrying)
    )
    scores[int(GlobalIntention.DEFEND)] = 0.55 * threat + 0.30 * danger
    scores[int(GlobalIntention.EXPLORE)] = 0.25 * diversity + 0.20 * (1.0 - crisis)
    scores[int(GlobalIntention.REPAIR)] = (
        0.45 * fragmentation + 0.40 * distress + 0.30 * (1.0 - integration) + 0.30 * crisis
    )

    old = getattr(global_state, "intention_scores", None)
    if (
        isinstance(old, np.ndarray)
        and old.shape == scores.shape
        and crisis < float(getattr(cfg.cross_scale_homeostasis, "crisis_threshold", 1.1))
    ):
        alpha = float(getattr(cfg.cross_scale_homeostasis, "apex_smoothing", 0.0))
        scores = (alpha * old + (1.0 - alpha) * scores).astype(DEFAULT_FLOAT_DTYPE)
    global_state.intention_scores = scores.astype(DEFAULT_FLOAT_DTYPE, copy=False)
    return int(np.argmax(scores))


def global_policy_to_bias(global_state: GlobalState, cfg: SimulationConfig) -> np.ndarray:
    """Convert apex intention to weak broad bias; suppress reproduction under crisis."""
    bias = np.zeros((len(Action),), dtype=DEFAULT_FLOAT_DTYPE)
    try:
        intention = GlobalIntention(int(global_state.intention))
    except ValueError:
        intention = GlobalIntention(compute_global_intention(global_state, cfg))
    crisis = float(np.clip(getattr(global_state, "crisis", 0.0), 0.0, 1.0))
    carrying = float(np.clip(getattr(global_state, "carrying_pressure", 0.0), 0.0, 1.0))
    starv = float(np.clip(getattr(global_state, "starvation_pressure", 0.0), 0.0, 1.0))

    if intention == GlobalIntention.SEEK_FOOD:
        bias[int(Action.SENSE)] += 0.08
        bias[int(Action.FEED)] += 0.18
        bias[int(Action.COMMUNICATE)] += 0.05
    elif intention == GlobalIntention.CONSERVE:
        bias[int(Action.REST)] += 0.10
        bias[int(Action.REPAIR)] += 0.10
        bias[int(Action.FEED)] += 0.08 * starv
        bias[int(Action.REPRODUCE)] -= 0.18
        bias[int(Action.INGEST)] -= 0.04
    elif intention == GlobalIntention.REPAIR:
        bias[int(Action.REPAIR)] += 0.16
        bias[int(Action.REST)] += 0.06
        bias[int(Action.SENSE)] += 0.04
    elif intention == GlobalIntention.COORDINATE:
        bias[int(Action.INTEGRATE)] += 0.14 * (1.0 - crisis)
        bias[int(Action.COMMUNICATE)] += 0.12
        bias[int(Action.SENSE)] += 0.05
    elif intention == GlobalIntention.REPRODUCE:
        bias[int(Action.REPRODUCE)] += 0.12 * (1.0 - max(crisis, carrying))
        bias[int(Action.COMMUNICATE)] += 0.05
    elif intention == GlobalIntention.AVOID_THREAT:
        bias[int(Action.FLEE)] += 0.14
        bias[int(Action.REPAIR)] += 0.10
        bias[int(Action.COMMUNICATE)] += 0.06
    elif intention == GlobalIntention.EXPLORE:
        for action in (
            Action.SENSE,
            Action.MOVE_N,
            Action.MOVE_S,
            Action.MOVE_E,
            Action.MOVE_W,
            Action.MOVE_NE,
            Action.MOVE_NW,
            Action.MOVE_SE,
            Action.MOVE_SW,
        ):
            bias[int(action)] += 0.05 * (1.0 - crisis)
    else:
        bias[int(Action.REST)] += 0.05
        bias[int(Action.INTEGRATE)] += 0.03 * (1.0 - crisis)

    # Always feed/repair/sense more under apex crisis and make reproduction less likely.
    bias[int(Action.FEED)] += 0.12 * starv + 0.08 * crisis
    bias[int(Action.REPAIR)] += 0.08 * crisis
    bias[int(Action.SENSE)] += 0.04 * crisis
    bias[int(Action.REPRODUCE)] -= (
        float(getattr(cfg.cross_scale_homeostasis, "max_reproduction_suppression", 0.95))
        * max(crisis, carrying)
        * 0.18
    )
    scale = (
        0.50
        * cfg.topdown.lambda_action_bias
        * max(float(np.clip(global_state.integration, 0.0, 1.0)), 0.1)
    )
    bias = _clip_bias(bias * scale, cfg)
    global_state.policy_bias[...] = bias
    return bias


# Use baseline patch intentions unless homeostasis is enabled.
_homeostasis_compute_patch_intention = _homeostasis_compute_patch_intention_impl


def compute_patch_intention(patches: PatchState, cfg: SimulationConfig) -> None:
    """Compute patch intentions and use baseline scoring when homeostasis is disabled."""
    if getattr(cfg.cross_scale_homeostasis, "enabled", False):
        _homeostasis_compute_patch_intention(patches, cfg)
        return
    hp, wp = _validate_patch_state(patches, cfg)
    integration = np.clip(patches.integration, 0.0, 1.0)
    resource = np.clip(patches.resource, 0.0, 1.0)
    health = np.clip(patches.health, 0.0, 1.0)
    boundary = np.clip(patches.boundary, 0.0, 1.0)
    synchrony = np.clip(patches.synchrony, 0.0, 1.0)
    coherence = np.clip(patches.coherence, 0.0, 1.0)
    food = _channel(patches.signal_pressure, SignalChannel.FOOD, cfg)
    danger = _channel(patches.signal_pressure, SignalChannel.DANGER, cfg)
    threat = _channel(patches.signal_pressure, SignalChannel.THREAT, cfg)
    coord = _channel(patches.signal_pressure, SignalChannel.COORDINATION, cfg)
    distress = _channel(patches.signal_pressure, SignalChannel.DISTRESS, cfg)
    repro = _channel(patches.signal_pressure, SignalChannel.REPRODUCTION, cfg)
    territory = _channel(patches.signal_pressure, SignalChannel.TERRITORY, cfg)
    integration_signal = _channel(patches.signal_pressure, SignalChannel.INTEGRATION, cfg)
    scores = np.zeros((hp, wp, len(PatchIntention)), dtype=DEFAULT_FLOAT_DTYPE)
    scores[..., int(PatchIntention.REST)] = 0.20 * resource + 0.20 * health + 0.10 * integration
    scores[..., int(PatchIntention.SEEK_FOOD)] = 0.80 * food + 0.35 * (1.0 - resource)
    scores[..., int(PatchIntention.AVOID_DANGER)] = 0.75 * danger + 0.50 * threat + 0.20 * distress
    scores[..., int(PatchIntention.COORDINATE)] = (
        0.55 * coord
        + 0.40 * integration_signal
        + 0.35 * (1.0 - integration)
        + 0.15 * (1.0 - synchrony)
        + 0.15 * (1.0 - coherence)
    )
    scores[..., int(PatchIntention.DEFEND)] = (
        0.55 * threat + 0.35 * territory + 0.20 * (1.0 - boundary)
    )
    scores[..., int(PatchIntention.REPRODUCE)] = (
        0.50 * repro + 0.60 * resource * health * boundary * np.maximum(integration, 0.05)
    )
    scores[..., int(PatchIntention.REPAIR)] = (
        0.65 * (1.0 - boundary) + 0.35 * (1.0 - health) + 0.25 * distress
    )
    scores[..., int(PatchIntention.EXPLORE)] = (
        0.20 * (1.0 - food) + 0.15 * resource + 0.10 * (1.0 - danger)
    )
    intention = np.argmax(scores, axis=-1).astype(DEFAULT_INT_DTYPE)
    intention[health <= 0.0] = int(PatchIntention.REST)
    patches.intention[...] = intention
