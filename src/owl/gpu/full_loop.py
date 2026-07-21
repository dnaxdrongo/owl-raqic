from __future__ import annotations

from typing import Any

import numpy as np

from owl.core.advanced import ensure_action_transition_fields, ensure_advanced_fields
from owl.core.config import SimulationConfig
from owl.core.state import WorldState
from owl.gpu.device_state import OWLDeviceState
from owl.gpu.invariants import assert_gpu_full_invariants, invariant_summary
from owl.gpu.profiler import GPUFullProfiler
from owl.gpu.stages.action_transitions_gpu import (
    apply_active_sense_transition_gpu,
    compile_selected_action_transition_gpu,
    prepare_action_transition_context_gpu,
)
from owl.gpu.stages.aggregation_gpu import aggregate_global_gpu, aggregate_patches_gpu
from owl.gpu.stages.authority_gpu import compute_authority_gpu
from owl.gpu.stages.collision_gpu import apply_inhibition_gpu, resolve_collisions_gpu
from owl.gpu.stages.communication_gpu import (
    emit_signals_gpu,
    update_channel_trust_gpu,
    update_signal_memory_gpu,
)
from owl.gpu.stages.death_gpu import apply_death_gpu
from owl.gpu.stages.environment_gpu import update_environment_gpu
from owl.gpu.stages.feeding_gpu import apply_feeding_gpu
from owl.gpu.stages.health_gpu import (
    apply_metabolism_damage_gpu,
    apply_repair_and_integrate_gpu,
    clip_life_fields_gpu,
)
from owl.gpu.stages.integration_gpu import update_integration_gpu
from owl.gpu.stages.memory_gpu import update_memory_gpu
from owl.gpu.stages.movement_gpu import apply_movement_gpu
from owl.gpu.stages.phase_gpu import (
    compute_cell_coherence_gpu,
    compute_cross_scale_coupling_gpu,
    compute_local_synchrony_gpu,
    update_phase_gpu,
)
from owl.gpu.stages.raqic_gpu_stage import (
    quiesce_dead_raqic_fields_gpu,
    run_raqic_gpu_stage,
)
from owl.gpu.stages.reproduction_gpu import apply_reproduction_gpu
from owl.gpu.stages.sensing_gpu import compute_sensing_bundle_gpu
from owl.gpu.stages.topdown_gpu import apply_threshold_modulation_gpu, dispatch_parent_context_gpu
from owl.gpu.stages.topology_gpu import apply_topology_events_gpu, detect_topology_events_gpu
from owl.gpu.stages.utility_gpu import compute_utilities_gpu
from owl.raqic.state import ensure_raqic_fields, quiesce_dead_raqic_fields


def _strict_and_fallback(cfg: SimulationConfig) -> tuple[bool, bool]:
    strict = bool(getattr(cfg.raqic, "full_gpu_strict", getattr(cfg.raqic, "strict_gpu", True)))
    allow_fallback = bool(getattr(cfg.raqic, "fallback_on_backend_error", False))
    if getattr(cfg.raqic, "mode", "") == "gpu_full_hybrid_audit":
        allow_fallback = True
    return strict, allow_fallback


def step_gpu_full(
    state: WorldState, cfg: SimulationConfig, rng: np.random.Generator | None = None
) -> dict[str, Any]:
    """Advance one full-stack GPU tick.

    ``stage_once`` preserves the  compatibility behavior. The optimized
     persistent engine is used by ``run_gpu_full`` and by scripts that call
    :mod:`owl.gpu.run_context`; single-step CPU callers still receive writeback
    after the step so ordinary metrics code stays valid.
    """
    execution_tier = str(getattr(cfg.raqic, "full_gpu_execution_tier", "reference"))
    transfer_policy = str(getattr(cfg.raqic, "full_gpu_transfer_policy", "stage_once"))
    if (
        execution_tier in {"persistent", "graph", "distributed"}
        or transfer_policy == "persistent_mirror"
    ):
        raise RuntimeError(
            "step_gpu_full is a stage-once compatibility API and cannot be used "
            f"for production execution tier {execution_tier!r}"
        )

    ensure_advanced_fields(state, cfg)
    ensure_action_transition_fields(state, cfg)
    if getattr(cfg.raqic, "enabled", False):
        ensure_raqic_fields(state, cfg)

    state.tick += 1
    strict, allow_fallback = _strict_and_fallback(cfg)
    ds = OWLDeviceState.from_world_state(state, cfg, strict=strict, allow_fallback=allow_fallback)
    ds.tick = state.tick
    profiler = GPUFullProfiler()
    diagnostics: dict[str, object] = {
        "mode": cfg.raqic.mode,
        "backend": ds.backend.name,
        "fallback": not ds.is_gpu,
    }

    with profiler.stage("environment"):
        update_environment_gpu(ds, cfg)
    with profiler.stage("sensing"):
        compute_sensing_bundle_gpu(ds, cfg)
        prepare_action_transition_context_gpu(ds, cfg)
    with profiler.stage("aggregation_pre"):
        aggregate_patches_gpu(ds, cfg)
        aggregate_global_gpu(ds, cfg)
        dispatch_parent_context_gpu(ds, cfg)
        apply_threshold_modulation_gpu(ds, cfg)
    with profiler.stage("phase"):
        update_phase_gpu(ds, cfg)
        compute_local_synchrony_gpu(ds, cfg)
        compute_cell_coherence_gpu(ds, cfg)
        compute_cross_scale_coupling_gpu(ds, cfg)
    with profiler.stage("utility_authority"):
        compute_utilities_gpu(ds, cfg)
        compute_authority_gpu(ds, cfg)
    with profiler.stage("raqic_decision"):
        diagnostics["raqic"] = run_raqic_gpu_stage(ds, cfg)
        compile_selected_action_transition_gpu(ds, cfg)
    with profiler.stage("movement_collision"):
        diagnostics["movement"] = apply_movement_gpu(ds, cfg)
        diagnostics["collision"] = resolve_collisions_gpu(ds, cfg)
        diagnostics["inhibition"] = apply_inhibition_gpu(ds, cfg)
    with profiler.stage("feeding_repair_comm_repro_topology"):
        diagnostics["feeding"] = apply_feeding_gpu(ds, cfg)
        apply_repair_and_integrate_gpu(ds, cfg)
        diagnostics["communication"] = emit_signals_gpu(ds, cfg)
        diagnostics["reproduction"] = apply_reproduction_gpu(ds, cfg)
        events = detect_topology_events_gpu(ds, cfg)
        diagnostics["topology"] = apply_topology_events_gpu(ds, cfg, events)
        diagnostics["active_sense"] = apply_active_sense_transition_gpu(ds, cfg)
    with profiler.stage("health_memory_integration"):
        diagnostics["metabolism"] = apply_metabolism_damage_gpu(ds, cfg)
        update_memory_gpu(ds, cfg)
        update_signal_memory_gpu(ds, cfg)
        update_integration_gpu(ds, cfg)
        update_channel_trust_gpu(ds, cfg)
    with profiler.stage("death_clip_aggregation_post"):
        diagnostics["death"] = apply_death_gpu(ds, cfg)
        clip_life_fields_gpu(ds, cfg)
        if getattr(cfg.raqic, "enabled", False):
            quiesce_dead_raqic_fields_gpu(ds)
        aggregate_patches_gpu(ds, cfg)
        aggregate_global_gpu(ds, cfg)
        dispatch_parent_context_gpu(ds, cfg)
    if cfg.debug.assert_invariants:
        with profiler.stage("invariants"):
            assert_gpu_full_invariants(ds, cfg)

    ds.metadata["compatibility_full_writeback_count"] = (
        int(ds.metadata.get("compatibility_full_writeback_count", 0)) + 1
    )
    ds.write_back_to_cpu(state)
    if getattr(cfg.raqic, "enabled", False):
        quiesce_dead_raqic_fields(state)
    diagnostics["invariants"] = invariant_summary(ds, cfg)
    diagnostics["profile"] = profiler.to_dict()
    from owl.core.state import EventRecord

    state.event_queue.append(EventRecord("gpu_full_diagnostics", state.tick, payload=diagnostics))
    return diagnostics


def run_gpu_full(cfg: SimulationConfig, max_steps: int | None = None) -> Any:
    if getattr(
        cfg.raqic, "full_gpu_transfer_policy", "stage_once"
    ) == "persistent_mirror" or getattr(cfg.raqic, "full_gpu_execution_tier", "reference") in (
        "persistent",
        "graph",
    ):
        from owl.gpu.run_context import run_gpu_full_persistent

        return run_gpu_full_persistent(cfg, max_steps=max_steps)

    from owl.core.init import initialize_world
    from owl.engine.loop import _collect_loop_metrics

    rng = np.random.default_rng(cfg.world.seed)
    state = initialize_world(cfg, rng)
    steps = cfg.world.max_steps if max_steps is None else int(max_steps)
    metrics = []
    for _ in range(steps):
        step_gpu_full(state, cfg, rng)
        metrics.append(_collect_loop_metrics(state, cfg))
    return state, metrics


def run_headless_gpu_full(cfg: SimulationConfig, max_steps: int | None = None) -> Any:
    return run_gpu_full(cfg, max_steps=max_steps)
