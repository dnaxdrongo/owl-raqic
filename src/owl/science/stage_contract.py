from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StageContract:
    name: str
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    neighborhood_radius: int
    stochastic_slots: tuple[str, ...]
    cpu_callable: str
    array_callable: str
    pre_epoch: str
    post_epoch: str
    event_types: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _s(
    name: str,
    reads: Any,
    writes: Any,
    radius: Any,
    cpu: Any,
    array: Any,
    pre: Any,
    post: Any,
    rng: Any = (),
    events: Any = (),
) -> Any:
    return StageContract(
        name, tuple(reads), tuple(writes), radius, tuple(rng), cpu, array, pre, post, tuple(events)
    )


STAGE_CONTRACTS: tuple[StageContract, ...] = (
    _s(
        "environment",
        ("food", "toxin", "signal", "signal_emission", "waste", "obstacle"),
        ("food", "toxin", "signal", "signal_emission", "waste"),
        1,
        "owl.engine.environment:update_environment",
        "owl.gpu.stages.environment_gpu:update_environment_gpu",
        "tick_start",
        "environment_updated",
    ),
    _s(
        "sensing",
        ("signal", "signal_memory", "channel_receptivity", "receive_sensitivity", "neighbor_trust"),
        ("signal_reception", "food_mean", "toxin_mean", "living_density"),
        1,
        "owl.engine.sensing:compute_signal_reception",
        "owl.gpu.stages.sensing_gpu:compute_sensing_bundle_gpu",
        "environment_updated",
        "sensed",
    ),
    _s(
        "parent_context",
        ("patches", "global_state", "threshold"),
        ("patches", "global_state", "threshold"),
        0,
        "owl.engine.loop:_ensure_parent_context",
        "owl.gpu.stages.topdown_gpu:dispatch_parent_context_gpu",
        "sensed",
        "parent_context",
    ),
    _s(
        "phase",
        ("phase", "noise", "coupling_strength", "parent_phase"),
        ("phase", "noetic_S", "noetic_C", "noetic_K"),
        1,
        "owl.engine.phase:update_phase",
        "owl.gpu.stages.phase_gpu:update_phase_gpu",
        "parent_context",
        "phase_updated",
        rng=("PHASE_NOISE",),
    ),
    _s(
        "utility",
        (
            "health",
            "resource",
            "food",
            "toxin",
            "memory",
            "integration",
            "signal_reception",
            "parent_bias",
        ),
        ("pre_utilities",),
        1,
        "owl.engine.utility:compute_utilities",
        "owl.gpu.stages.utility_gpu:compute_utilities_gpu",
        "phase_updated",
        "utility_ready",
    ),
    _s(
        "authority",
        ("health", "resource", "boundary", "integration", "traits", "food", "signal_reception"),
        ("pre_authority", "_authority_bool"),
        1,
        "owl.engine.authority:compute_authority",
        "owl.gpu.stages.authority_gpu:compute_authority_gpu",
        "utility_ready",
        "authority_ready",
    ),
    _s(
        "decision",
        (
            "pre_utilities",
            "pre_authority",
            "raqic_parent_intention",
            "raqic_parent_action_phase",
            "raqic_parent_action_coherence",
            "phase",
            "noetic_S",
            "noetic_C",
            "noetic_K",
        ),
        (
            "possibility",
            "readout",
            "raqic_probabilities",
            "raqic_phase",
            "raqic_readout",
            "raqic_record_action",
            "raqic_record_readout",
            "raqic_record_confidence",
            "raqic_score",
            "raqic_trace_error",
            "raqic_min_eigenvalue",
            "raqic_audit_flags",
            "raqic_backend_code",
            "raqic_pre_mixer_probabilities",
            "raqic_utility_innovation",
            "raqic_phase_alignment",
            "raqic_resonant_parent_intention",
            "raqic_interference_delta_l1",
            "raqic_policy_kl",
            "raqic_utility_projection_fraction",
            "raqic_utility_score_cosine",
            "raqic_utility_orthogonality_residual",
            "raqic_utility_innovation_norm",
            "raqic_interference_norm_error",
            "raqic_interference_illegal_mass",
            "raqic_shadow_probabilities",
            "raqic_shadow_readout",
        ),
        0,
        "owl.engine.decision_policy:apply_decision_policy",
        "owl.gpu.stages.raqic_gpu_stage:run_raqic_gpu_stage",
        "authority_ready",
        "decision_ready",
        rng=("RAQIC_READOUT",),
    ),
    _s(
        "movement",
        ("readout", "occupancy", "cell_fields"),
        ("occupancy", "cell_fields"),
        1,
        "owl.engine.movement:apply_movement",
        "owl.gpu.stages.movement_gpu:apply_movement_gpu",
        "decision_ready",
        "movement_done",
        rng=("MOVEMENT_TIE",),
        events=("movement",),
    ),
    _s(
        "collision",
        ("occupancy", "readout", "health", "resource"),
        ("health", "resource", "readout"),
        1,
        "owl.engine.collision:resolve_collisions",
        "owl.gpu.stages.collision_gpu:resolve_collisions_gpu",
        "movement_done",
        "collision_done",
        rng=("MOVEMENT_TIE",),
        events=("collision",),
    ),
    _s(
        "inhibition",
        ("readout", "health", "resource", "boundary"),
        ("health", "resource", "boundary"),
        1,
        "owl.engine.collision:apply_inhibition",
        "owl.gpu.stages.collision_gpu:apply_inhibition_gpu",
        "collision_done",
        "inhibition_done",
        events=("inhibition",),
    ),
    _s(
        "feeding",
        ("readout", "food", "resource", "digestion"),
        ("food", "resource", "digestion", "waste"),
        0,
        "owl.engine.feeding:apply_feeding",
        "owl.gpu.stages.feeding_gpu:apply_feeding_gpu",
        "inhibition_done",
        "feeding_done",
    ),
    _s(
        "health_actions",
        ("readout", "health", "resource", "integration"),
        ("health", "resource", "integration"),
        0,
        "owl.engine.health:apply_repair_and_integrate",
        "owl.gpu.stages.health_gpu:apply_repair_and_integrate_gpu",
        "feeding_done",
        "health_actions_done",
    ),
    _s(
        "communication_emit",
        ("readout", "signal_emission", "resource", "traits"),
        ("signal_emission", "resource"),
        0,
        "owl.engine.communication:emit_signals",
        "owl.gpu.stages.communication_gpu:emit_signals_gpu",
        "health_actions_done",
        "communication_emitted",
        events=("communication",),
    ),
    _s(
        "reproduction",
        ("readout", "occupancy", "health", "resource", "boundary", "integration", "genome"),
        ("occupancy", "cell_fields", "next_ow_id"),
        1,
        "owl.engine.reproduction:apply_reproduction",
        "owl.gpu.stages.reproduction_gpu:apply_reproduction_gpu",
        "communication_emitted",
        "reproduction_done",
        rng=("REPRODUCTION_TIE",),
        events=("reproduction",),
    ),
    _s(
        "topology",
        ("readout", "occupancy", "cell_fields"),
        ("occupancy", "cell_fields"),
        1,
        "owl.engine.topology:apply_topology_events",
        "owl.gpu.stages.topology_gpu:apply_topology_events_gpu",
        "reproduction_done",
        "topology_done",
        rng=("TOPOLOGY_TIE",),
        events=("merge", "split", "expel"),
    ),
    _s(
        "metabolism",
        ("health", "resource", "toxin", "readout", "traits"),
        ("health", "resource", "age", "starvation_debt"),
        0,
        "owl.engine.health:apply_metabolism_damage",
        "owl.gpu.stages.health_gpu:apply_metabolism_damage_gpu",
        "topology_done",
        "metabolism_done",
    ),
    _s(
        "memory",
        ("memory", "resource", "health", "integration", "food"),
        ("memory",),
        0,
        "owl.engine.memory:update_memory",
        "owl.gpu.stages.memory_gpu:update_memory_gpu",
        "metabolism_done",
        "memory_done",
    ),
    _s(
        "signal_memory",
        ("signal_reception", "signal_memory"),
        ("signal_memory",),
        0,
        "owl.engine.communication:update_signal_memory",
        "owl.gpu.stages.communication_gpu:update_signal_memory_gpu",
        "memory_done",
        "signal_memory_done",
    ),
    _s(
        "integration",
        (
            "memory",
            "resource",
            "health",
            "boundary",
            "possibility",
            "noetic_S",
            "noetic_C",
            "noetic_K",
            "parent_bias",
        ),
        ("integration",),
        0,
        "owl.engine.integration:update_integration",
        "owl.gpu.stages.integration_gpu:update_integration_gpu",
        "signal_memory_done",
        "integration_done",
    ),
    _s(
        "trust",
        ("signal_reception", "resource", "health", "integration"),
        ("channel_trust_local", "neighbor_trust"),
        1,
        "owl.engine.communication:update_channel_trust",
        "owl.gpu.stages.communication_gpu:update_channel_trust_gpu",
        "integration_done",
        "trust_done",
    ),
    _s(
        "death",
        ("health", "occupancy", "cell_fields"),
        ("occupancy", "cell_fields"),
        0,
        "owl.engine.death:apply_death",
        "owl.gpu.stages.death_gpu:apply_death_gpu",
        "trust_done",
        "death_done",
        events=("death",),
    ),
    _s(
        "clip",
        ("bounded_fields",),
        ("bounded_fields",),
        0,
        "owl.engine.health:clip_life_fields",
        "owl.gpu.stages.health_gpu:clip_life_fields_gpu",
        "death_done",
        "clipped",
    ),
    _s(
        "aggregation",
        ("cell_fields",),
        ("patches", "global_state"),
        0,
        "owl.engine.aggregation:aggregate_patches",
        "owl.gpu.stages.aggregation_gpu:aggregate_patches_gpu",
        "clipped",
        "aggregated",
    ),
    _s(
        "topdown_dispatch",
        ("patches", "global_state"),
        ("raqic_parent_intention",),
        0,
        "owl.engine.topdown:patch_policy_to_bias",
        "owl.gpu.stages.topdown_gpu:dispatch_parent_context_gpu",
        "aggregated",
        "tick_complete",
    ),
)


def scientific_stage_order() -> tuple[str, ...]:
    return tuple(stage.name for stage in STAGE_CONTRACTS)


def write_stage_contract(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"schema_version": "1", "stages": [s.to_dict() for s in STAGE_CONTRACTS]},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path
