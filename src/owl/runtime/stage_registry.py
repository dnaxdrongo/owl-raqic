from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TickStageSpec:
    """Authoritative semantic stage descriptor shared by runtime planners."""

    name: str
    phase: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    graph_segment: str
    distributed_policy: str
    visual_event_types: tuple[str, ...] = ()


TICK_STAGES: tuple[TickStageSpec, ...] = (
    TickStageSpec(
        "environment",
        "predecision",
        ("food", "toxin", "signal"),
        ("food", "toxin", "signal"),
        "predecision",
        "halo",
    ),
    TickStageSpec(
        "sensing",
        "predecision",
        ("food", "toxin", "signal", "occupancy"),
        ("sensing",),
        "predecision",
        "halo",
    ),
    TickStageSpec(
        "parent_context",
        "predecision",
        ("patch_state", "global_state"),
        ("parent_intention",),
        "predecision",
        "reduce",
    ),
    TickStageSpec(
        "phase_coherence",
        "predecision",
        ("phase", "parent_intention"),
        ("phase", "coherence", "synchrony"),
        "predecision",
        "halo",
    ),
    TickStageSpec(
        "utility_authority",
        "predecision",
        ("sensing", "health", "resource"),
        ("utility", "authority"),
        "predecision",
        "local",
    ),
    TickStageSpec(
        "raqic",
        "decision",
        ("features", "authority", "parent_intention"),
        ("probabilities", "readout"),
        "decision",
        "local",
    ),
    TickStageSpec(
        "movement",
        "actions",
        ("readout", "occupancy"),
        ("occupancy", "movement_events"),
        "actions",
        "boundary_events",
        ("move",),
    ),
    TickStageSpec(
        "collision_inhibition",
        "actions",
        ("movement_events", "occupancy"),
        ("health", "resource"),
        "actions",
        "boundary_events",
        ("ingest", "inhibit"),
    ),
    TickStageSpec(
        "feeding_repair",
        "actions",
        ("readout", "food", "health", "resource"),
        ("food", "health", "resource"),
        "actions",
        "local",
        ("feed", "repair"),
    ),
    TickStageSpec(
        "communication",
        "actions",
        ("readout", "signal"),
        ("signal_emission",),
        "actions",
        "halo",
        ("communicate",),
    ),
    TickStageSpec(
        "reproduction",
        "actions",
        ("readout", "occupancy", "genome"),
        ("occupancy", "lineage"),
        "actions",
        "boundary_events",
        ("reproduce",),
    ),
    TickStageSpec(
        "topology",
        "actions",
        ("readout", "occupancy"),
        ("topology_events", "occupancy"),
        "actions",
        "boundary_events",
        ("merge", "split", "expel"),
    ),
    TickStageSpec(
        "biology_memory",
        "postdecision",
        ("health", "resource", "toxin", "readout"),
        ("health", "resource", "memory"),
        "postdecision",
        "local",
    ),
    TickStageSpec(
        "integration_trust",
        "postdecision",
        ("memory", "coherence", "signals"),
        ("integration", "trust"),
        "postdecision",
        "halo",
    ),
    TickStageSpec(
        "death_cleanup",
        "postdecision",
        ("health", "occupancy"),
        ("death_mask", "occupancy"),
        "postdecision",
        "boundary_events",
        ("death",),
    ),
    TickStageSpec(
        "post_aggregation",
        "postdecision",
        ("health", "resource", "readout"),
        ("patch_state", "global_state"),
        "postdecision",
        "reduce",
    ),
    TickStageSpec(
        "metrics_visual",
        "postdecision",
        ("world_state",),
        ("metric_slab", "visual_events"),
        "postdecision",
        "gather",
    ),
)


GRAPH_SEGMENTS: tuple[str, ...] = ("predecision", "decision", "actions", "postdecision")


def stage_names() -> tuple[str, ...]:
    return tuple(stage.name for stage in TICK_STAGES)


def stages_for_segment(name: str) -> tuple[TickStageSpec, ...]:
    if name not in GRAPH_SEGMENTS:
        raise KeyError(name)
    return tuple(stage for stage in TICK_STAGES if stage.graph_segment == name)
