"""Top-level simulation step and run loop.

This module is the only engine layer that composes all lower-level subsystems.
Lower-level engine modules must not import this module. The loop keeps the
three model layers explicit:

* physical/classical: environment, resources, movement, collision, health,
  death, reproduction, and finite ticks;
* quantum-inspired possibility: utility, authority, softmax/argmax
  actualization, and normalized possibility vectors;
* fractal/mosaic: phase, synchrony, coherence, cross-scale coupling,
  patch/global aggregation, and weak top-down bias.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from owl.core.actions import Action
from owl.core.advanced import ensure_action_transition_fields, ensure_advanced_fields
from owl.core.config import SimulationConfig
from owl.core.init import initialize_world
from owl.core.state import WorldState, action_shape, channel_shape, field_shape
from owl.engine.action_transitions import (
    apply_active_sense_transition,
    compile_selected_action_transition,
    prepare_action_transition_context,
)
from owl.engine.aggregation import (
    aggregate_global,
    aggregate_patches,
    upsample_patch_bias,
    upsample_patch_field,
)
from owl.engine.authority import compute_authority
from owl.engine.collision import apply_inhibition, resolve_collisions
from owl.engine.communication import emit_signals, update_channel_trust, update_signal_memory
from owl.engine.death import apply_death
from owl.engine.decision_policy import apply_decision_policy
from owl.engine.environment import update_environment
from owl.engine.feeding import apply_feeding
from owl.engine.health import apply_metabolism_damage, apply_repair_and_integrate, clip_life_fields
from owl.engine.integration import compute_conflict, update_integration
from owl.engine.memory import update_memory
from owl.engine.movement import apply_movement
from owl.engine.phase import (
    compute_cell_coherence as compute_cell_coherence,
)
from owl.engine.phase import (
    compute_cross_scale_coupling as compute_cross_scale_coupling,
)
from owl.engine.phase import (
    compute_local_synchrony as compute_local_synchrony,
)
from owl.engine.phase import (
    update_phase,
)
from owl.engine.reproduction import apply_reproduction
from owl.engine.scheduler import should_record, should_update_global, should_update_patches
from owl.engine.sensing import compute_signal_reception
from owl.engine.topdown import (
    apply_threshold_modulation,
    compute_global_intention,
    compute_patch_intention,
    global_policy_to_bias,
    patch_policy_to_bias,
)
from owl.engine.topology import apply_topology_events, detect_topology_events
from owl.engine.utility import compute_utilities


def _validate_steps(max_steps: int | None, cfg: SimulationConfig) -> int:
    """Return a positive run length from an optional override."""
    steps = int(cfg.world.max_steps if max_steps is None else max_steps)
    if steps < 0:
        raise ValueError(f"max_steps must be nonnegative, got {max_steps!r}")
    return steps


def _ensure_parent_context(
    state: WorldState, cfg: SimulationConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate current patch/global state and return cell-level bias/phase.

    Mutates ``state.patches`` and possibly ``state.global_state``. Top-down
    influence remains a bounded logit/threshold bias; no child readouts are
    overwritten.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(parent_bias, parent_phase)`` with shapes
        ``(height, width, len(Action))`` and ``(height, width)``.
    """
    h, w = field_shape(state)

    if should_update_patches(state.tick, cfg):
        state.patches = aggregate_patches(state, cfg)
        compute_patch_intention(state.patches, cfg)
        patch_bias = patch_policy_to_bias(state.patches, cfg)
    else:
        patch_bias = state.patches.policy_bias

    if should_update_global(state.tick, cfg):
        state.global_state = aggregate_global(state.patches, cfg)
        state.global_state.intention = compute_global_intention(state.global_state, cfg)
        global_policy_to_bias(state.global_state, cfg)

    parent_bias = upsample_patch_bias(patch_bias, cfg.world.patch_size)
    expected_bias = (h, w, len(Action))
    if parent_bias.shape != expected_bias:
        raise ValueError(f"parent_bias must have shape {expected_bias}, got {parent_bias.shape}")

    # Apex/global policy is a weak, broad action-bias vector broadcast to all
    # cells. This is a bias only, never a direct readout overwrite.
    global_bias = np.asarray(state.global_state.policy_bias, dtype=np.float32)
    if global_bias.shape != (len(Action),):
        raise ValueError(
            f"global policy_bias must have shape {(len(Action),)}, got {global_bias.shape}"
        )
    parent_bias = parent_bias + global_bias[None, None, :]

    limit = float(cfg.topdown.max_parent_control)
    parent_bias = np.clip(parent_bias, -limit, limit).astype(np.float32, copy=False)

    parent_phase = upsample_patch_field(state.patches.phase, cfg.world.patch_size)
    if parent_phase.shape != (h, w):
        raise ValueError(f"parent_phase must have shape {(h, w)}, got {parent_phase.shape}")
    parent_phase = parent_phase.astype(np.float32, copy=False)

    apply_threshold_modulation(state, state.patches, cfg)
    return parent_bias, parent_phase


def _post_state_refresh(state: WorldState, cfg: SimulationConfig) -> None:
    """Refresh patch/global summaries after dense cell mutations."""
    state.patches = aggregate_patches(state, cfg)
    compute_patch_intention(state.patches, cfg)
    patch_policy_to_bias(state.patches, cfg)
    state.global_state = aggregate_global(state.patches, cfg)
    state.global_state.intention = compute_global_intention(state.global_state, cfg)
    global_policy_to_bias(state.global_state, cfg)


def _collect_loop_metrics(state: WorldState, cfg: SimulationConfig) -> dict[str, Any]:
    """Collect lightweight scalar metrics without depending on later record pass."""
    alive = (state.health > 0.0) & (~state.obstacle)
    alive_count = int(np.count_nonzero(alive))
    patch_integration = np.asarray(state.patches.integration, dtype=np.float32)
    metrics = {
        "tick": int(state.tick),
        "record": bool(should_record(state.tick, cfg)),
        "alive_count": alive_count,
        "food_total": float(np.sum(state.food, dtype=np.float64)),
        "signal_total": float(np.sum(state.signal, dtype=np.float64)),
        "event_count": int(len(state.event_queue)),
        "global_integration": float(state.global_state.integration),
        "fragmentation": float(state.global_state.fragmentation),
        "diversity": float(state.global_state.diversity),
        "complexity": float(state.global_state.complexity),
        "mean_patch_integration": float(np.mean(patch_integration))
        if patch_integration.size
        else 0.0,
    }
    if alive_count:
        metrics.update(
            {
                "mean_integration": float(np.mean(state.integration[alive], dtype=np.float64)),
                "mean_resource": float(np.mean(state.resource[alive], dtype=np.float64)),
                "mean_health": float(np.mean(state.health[alive], dtype=np.float64)),
                "mean_boundary": float(np.mean(state.boundary[alive], dtype=np.float64)),
            }
        )
    else:
        metrics.update(
            {
                "mean_integration": 0.0,
                "mean_resource": 0.0,
                "mean_health": 0.0,
                "mean_boundary": 0.0,
            }
        )
    return metrics


def capture_pre_decision_state(
    state: WorldState,
    cfg: SimulationConfig,
    authority: np.ndarray,
    utilities: np.ndarray,
    parent_bias: np.ndarray,
) -> None:
    """Snapshot pre-action fields for causal decision audit recording."""
    ensure_advanced_fields(state, cfg)
    ensure_action_transition_fields(state, cfg)
    assert state.pre_resource is not None
    assert state.pre_health is not None
    assert state.pre_food is not None
    assert state.pre_starvation_debt is not None
    assert state.pre_authority is not None
    assert state.pre_utilities is not None
    assert state.pre_parent_bias is not None
    alive = (state.health > 0.0) & (~state.obstacle)
    state.pre_resource[...] = np.clip(state.resource, 0.0, cfg.resources.max_resource)
    state.pre_health[...] = np.clip(state.health, 0.0, 1.0)
    state.pre_food[...] = np.clip(state.food, 0.0, 1.0)
    if isinstance(state.starvation_debt, np.ndarray):
        state.pre_starvation_debt[...] = np.clip(state.starvation_debt, 0.0, 1.0)
    state.pre_authority[...] = authority.astype(np.float32, copy=False)
    state.pre_utilities[...] = utilities.astype(np.float32, copy=False)
    state.pre_parent_bias[...] = parent_bias.astype(np.float32, copy=False)
    for name in ("pre_resource", "pre_health", "pre_food", "pre_starvation_debt"):
        arr = getattr(state, name, None)
        if isinstance(arr, np.ndarray):
            arr[~alive] = 0.0


def assert_invariants(state: WorldState, cfg: SimulationConfig) -> None:
    """Assert baseline loop invariants for debug runs.

    Raises
    ------
    AssertionError
        If a bounded field leaves range, a probability vector drifts off the
        simplex for living cells, or a core shape is inconsistent.
    """
    ensure_advanced_fields(state, cfg)
    h, w = field_shape(state)
    ah, aw, actions = action_shape(state)
    ch, cw, channels = channel_shape(state)
    assert (ah, aw) == (h, w), "possibility spatial shape must match cell shape"
    assert (ch, cw) == (h, w), "signal spatial shape must match cell shape"
    assert actions == len(Action), "possibility action axis must equal len(Action)"
    assert channels == cfg.communication.num_channels, "signal channel axis must match config"

    bounded = (
        state.activation,
        state.memory,
        state.integration,
        state.health,
        state.boundary,
        state.emit_strength,
        state.emit_efficiency,
        state.receive_sensitivity,
        state.signal_precision,
        state.honesty_bias,
        state.deception_bias,
        state.channel_receptivity,
        state.channel_emission_bias,
        state.channel_trust_local,
        state.signal_reception,
        state.signal_memory,
        state.food,
        state.toxin,
        state.signal,
    )
    for array in bounded:
        assert np.all(np.isfinite(array)), "bounded arrays must be finite"
        assert np.nanmin(array) >= -1e-6, "bounded arrays must not be negative"
        assert np.nanmax(array) <= 1.0 + 1e-6, "bounded arrays must not exceed one"

    for name in (
        "digestion",
        "waste",
        "age_stress",
        "last_intake",
        "prediction_error",
        "development_stage",
        "symbiosis",
        "deception_memory",
        "source_confidence",
        "neighbor_trust",
        "genome",
        "action_cooldown",
        "pre_resource",
        "pre_health",
        "pre_food",
        "pre_starvation_debt",
        "last_decision_urgency",
        "last_homeostatic_error",
        "last_survival_value",
        "last_macro_probabilities",
        "noetic_B",
        "noetic_M",
        "noetic_P",
        "noetic_C",
        "noetic_K",
        "noetic_Theta",
        "noetic_N",
    ):
        optional_array = getattr(state, name, None)
        if isinstance(optional_array, np.ndarray):
            assert np.all(np.isfinite(optional_array)), f"{name} must be finite"
            assert np.nanmin(optional_array) >= -1e-6, f"{name} must not be negative"
            assert np.nanmax(optional_array) <= 1.0 + 1e-6, f"{name} must not exceed one"

    assert np.all(np.isfinite(state.phase)), "phase must be finite"
    assert np.all((state.resource >= 0.0) & (state.resource <= cfg.resources.max_resource + 1e-6))
    assert np.all((state.readout >= 0) & (state.readout < len(Action)))

    alive = (state.health > 0.0) & (~state.obstacle)
    if np.any(alive):
        sums = state.possibility.sum(axis=-1)
        assert np.allclose(sums[alive], 1.0, atol=1e-4), (
            "living possibility vectors must sum to one"
        )
        assert np.all(state.possibility[alive] >= -1e-7), "living possibilities must be nonnegative"

    if (
        getattr(cfg.identity, "enabled", False)
        or getattr(cfg.decision_homeostasis, "enabled", False)
        or getattr(cfg.cross_scale_homeostasis, "enabled", False)
    ) and np.any(alive):
        ids = state.occupancy[alive]
        ids = ids[ids >= 0]
        if ids.size:
            _, counts = np.unique(ids, return_counts=True)
            assert int(np.max(counts)) == 1, "living OW occupancy ids must be unique"

    dead = ~alive
    if np.any(dead):
        # Dead cells are expected to be quiescent after death clearing. During a
        # tick, reproduction/movement may briefly alter fields, but invariant
        # checks run only at completed tick boundaries.
        assert np.all(state.readout[dead] == int(Action.REST)), (
            "dead cells must REST at tick boundary"
        )


# Private alias retained for callers that use the alternate name.
_assert_core_invariants = assert_invariants


def step(state: WorldState, cfg: SimulationConfig, rng: np.random.Generator) -> None:
    """Advance the simulation by one tick.

    Parameters
    ----------
    state:
        Runtime dense state. This function mutates most dense state fields:
        environment, signals, phase, possibility/readout, resources, health,
        boundary, memory, integration, topology events, and patch/global
        summaries.
    cfg:
        Validated simulation configuration.
    rng:
        Explicit random generator used for deterministic replay.
    """
    if rng is None:
        raise ValueError("rng must be an explicit np.random.Generator")
    if getattr(getattr(cfg, "raqic", None), "mode", "") in ("gpu_full", "gpu_full_hybrid_audit"):
        from owl.gpu.full_loop import step_gpu_full

        step_gpu_full(state, cfg, rng)
        return
    ensure_advanced_fields(state, cfg)
    if getattr(cfg.raqic, "enabled", False):
        from owl.raqic.state import ensure_raqic_fields

        ensure_raqic_fields(state, cfg)

    # Tick increments before scheduling, so tick=1 is the first completed step.
    state.tick += 1

    prev_resource = state.resource.copy()
    prev_health = state.health.copy()
    prev_integration = state.integration.copy()

    # Physical communication substrate and sensing.
    update_environment(state, cfg)
    compute_signal_reception(state, cfg)
    prepare_action_transition_context(state, cfg)

    # Fractal/mosaic parent context for this tick.
    parent_bias, parent_phase = _ensure_parent_context(state, cfg)

    # Phase/coupling layer.
    update_phase(state, parent_phase, rng, cfg)
    synchrony = compute_local_synchrony(state, cfg)
    coherence = compute_cell_coherence(state, cfg)
    cross_scale = compute_cross_scale_coupling(state, parent_phase, cfg)

    # Possibility/actualization layer. RAQIC mode replaces probability/readout
    # production while preserving OWL's physical consequence modules below.
    utilities = compute_utilities(state, parent_bias, cfg)
    authority = compute_authority(state, cfg)
    capture_pre_decision_state(state, cfg, authority, utilities, parent_bias)
    apply_decision_policy(
        state=state,
        cfg=cfg,
        rng=rng,
        utilities=utilities,
        authority=authority,
        parent_bias=parent_bias,
        parent_phase=parent_phase,
        synchrony=synchrony,
        coherence=coherence,
        cross_scale=cross_scale,
    )
    compile_selected_action_transition(state, cfg)

    # Physical action consequences.
    apply_movement(state, cfg, rng)
    resolve_collisions(state, cfg, rng)
    apply_inhibition(state, cfg)
    apply_feeding(state, cfg)
    apply_repair_and_integrate(state, cfg)
    emit_signals(state, cfg)
    apply_reproduction(state, cfg, rng)
    detect_topology_events(state, cfg)
    apply_topology_events(state, cfg)
    apply_active_sense_transition(state, cfg)

    # Survival, memory, and integration after action consequences.
    apply_metabolism_damage(state, cfg)
    update_memory(state, cfg)
    update_signal_memory(state, cfg)

    conflict = compute_conflict(state, parent_bias, cfg)
    update_integration(state, synchrony, coherence, cross_scale, conflict, cfg)
    update_channel_trust(state, prev_resource, prev_health, prev_integration, cfg)

    # Death/release is last among dense mutation stages, then refresh summaries.
    apply_death(state, cfg)
    clip_life_fields(state, cfg)
    if getattr(cfg.raqic, "enabled", False):
        from owl.raqic.state import quiesce_dead_raqic_fields

        quiesce_dead_raqic_fields(state)
    _post_state_refresh(state, cfg)

    if cfg.debug.assert_invariants:
        assert_invariants(state, cfg)
        if getattr(cfg.raqic, "enabled", False):
            from owl.raqic.invariants import assert_raqic_invariants

            assert_raqic_invariants(state, cfg)


def run(
    cfg: SimulationConfig, max_steps: int | None = None
) -> tuple[WorldState, list[dict[str, Any]]]:
    """Run a complete simulation and return final state plus scalar metrics.

    Parameters
    ----------
    cfg:
        Validated simulation configuration.
    max_steps:
        Optional nonnegative step override. ``None`` uses ``cfg.world.max_steps``.

    Returns
    -------
    tuple[WorldState, list[dict]]
        Final runtime state and one lightweight metrics dictionary per completed
        tick. No visualization or recording is performed in .
    """
    rng = np.random.default_rng(cfg.world.seed)
    state = initialize_world(cfg, rng)
    ensure_advanced_fields(state, cfg)
    if getattr(cfg.raqic, "enabled", False):
        from owl.raqic.state import ensure_raqic_fields

        ensure_raqic_fields(state, cfg)
    steps = _validate_steps(max_steps, cfg)
    metrics: list[dict[str, Any]] = []

    # Ensure parent/global summaries are initialized consistently even for
    # zero-step runs.
    _post_state_refresh(state, cfg)

    for _ in range(steps):
        step(state, cfg, rng)
        metrics.append(_collect_loop_metrics(state, cfg))

    return state, metrics


def run_headless(
    cfg: SimulationConfig, max_steps: int | None = None
) -> tuple[WorldState, list[dict[str, Any]]]:
    """Run without real-time visualization.

    This is currently an alias for :func:`run`. Later visualization/recording
    passes may add a graphical ``run`` path while keeping this function stable
    for automated experiments and tests.
    """
    return run(cfg, max_steps=max_steps)
