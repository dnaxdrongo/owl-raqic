from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ShapeKind = Literal["cell", "action", "channel", "patch", "global", "event", "scalar"]
OwnerKind = Literal["cell_resident", "field_resident", "patch_resident", "global", "event"]


@dataclass(frozen=True)
class FieldSpec:
    name: str
    shape_kind: ShapeKind
    owner: OwnerKind
    dtype: str = "float32"
    moves_with_cell: bool = False
    clears_on_death: bool = False
    record_default: bool = False
    description: str = ""
    layout_group: str = ""
    visual_role: str = ""
    audit_role: str = ""
    record_role: str = ""
    copy_on_reproduction: bool = False


CELL_RESIDENT_FIELDS: tuple[str, ...] = (
    "activation",
    "memory",
    "phase",
    "threshold",
    "readout",
    "integration",
    "resource",
    "health",
    "boundary",
    "age",
    "ow_type",
    "lineage_id",
    "parent_id",
    "mobility",
    "metabolism",
    "predation",
    "grazing",
    "cooperation",
    "aggression",
    "curiosity",
    "reproduction_rate",
    "toxin_resistance",
    "memory_capacity",
    "coupling_strength",
    "emit_strength",
    "emit_efficiency",
    "receive_sensitivity",
    "signal_precision",
    "honesty_bias",
    "deception_bias",
    "digestion",
    "waste",
    "age_stress",
    "last_intake",
    "starvation_debt",
    "last_movement_action",
    "movement_loop_score",
    "development_stage",
    "symbiosis",
    "pre_resource",
    "pre_health",
    "pre_starvation_debt",
    "last_decision_urgency",
    "last_survival_value",
    "last_homeostatic_error",
    "prediction_error",
    "alive_density",
    "food_mean",
    "death_pressure",
    "noetic_B",
    "noetic_M",
    "noetic_P",
    "noetic_C",
    "noetic_K",
    "noetic_Theta",
    "noetic_N",
)

FIELD_RESIDENT_FIELDS: tuple[str, ...] = (
    "food",
    "toxin",
    "obstacle",
    "occupancy",
    "noise",
)

CHANNEL_RESIDENT_FIELDS: tuple[str, ...] = (
    "signal",
    "signal_emission",
    "signal_reception",
    "signal_memory",
    "channel_receptivity",
    "channel_emission_bias",
    "channel_trust_local",
    "neighbor_trust",
    "deception_memory",
    "source_confidence",
)

ACTION_RESIDENT_FIELDS: tuple[str, ...] = (
    "possibility",
    "last_utilities",
    "last_logits",
    "last_action_probabilities",
    "action_cooldown",
    "pre_authority",
    "pre_utilities",
    "pre_parent_bias",
    "last_macro_probabilities",
    "raqic_probabilities",
    "raqic_score",
    "raqic_phase",
    "raqic_parent_intention",
    "raqic_legacy_shadow_possibility",
    "raqic_parent_action_phase",
    "raqic_parent_action_coherence",
    "raqic_pre_mixer_probabilities",
    "raqic_utility_innovation",
    "raqic_phase_alignment",
    "raqic_resonant_parent_intention",
    "raqic_shadow_probabilities",
)

RAQIC_CELL_FIELDS: tuple[str, ...] = (
    "raqic_readout",
    "raqic_record_action",
    "raqic_record_readout",
    "raqic_record_confidence",
    "raqic_audit_flags",
    "raqic_trace_error",
    "raqic_min_eigenvalue",
    "raqic_backend_code",
    "raqic_legacy_shadow_readout",
    "raqic_compare_l1",
    "raqic_compare_kl",
    "raqic_debug_density_diag",
    "raqic_interference_delta_l1",
    "raqic_policy_kl",
    "raqic_utility_projection_fraction",
    "raqic_utility_score_cosine",
    "raqic_utility_orthogonality_residual",
    "raqic_utility_innovation_norm",
    "raqic_interference_norm_error",
    "raqic_interference_illegal_mass",
    "raqic_shadow_readout",
)

ACTION_TRANSITION_CELL_FIELDS: tuple[str, ...] = (
    "active_sense_food_memory",
    "active_sense_toxin_memory",
    "active_sense_alive_memory",
    "active_sense_ttl",
    "active_sense_new_cell_count",
    "active_sense_new_target_count",
    "flee_compiled_action",
    "pursue_compiled_action",
    "compiled_execution_action",
    "action_target_y",
    "action_target_x",
    "action_target_ow_id",
    "action_target_kind",
    "action_target_source",
    "action_target_distance",
    "action_target_confidence",
    "action_direction_y",
    "action_direction_x",
    "action_direction_executable",
    "action_direction_score",
    "action_direction_distance_delta",
    "action_direction_hazard",
    "action_direction_opportunity",
)

PATCH_FIELDS: tuple[str, ...] = (
    "activation",
    "memory",
    "phase",
    "possibility",
    "integration",
    "resource",
    "health",
    "boundary",
    "signal_pressure",
    "synchrony",
    "coherence",
    "cross_scale",
    "intention",
    "policy_bias",
    "intention_scores",
    "centroid_y",
    "centroid_x",
    "velocity_y",
    "velocity_x",
    "prediction_error",
    "alive_density",
    "food_mean",
    "starvation_debt_mean",
    "reproduction_fraction",
    "movement_fraction",
    "feed_fraction",
    "death_pressure",
    "patch_crisis",
    "patch_carrying_pressure",
    "noetic_B",
    "noetic_M",
    "noetic_P",
    "noetic_C",
    "noetic_K",
    "noetic_Theta",
    "noetic_N",
)


def build_field_registry() -> dict[str, FieldSpec]:
    specs: dict[str, FieldSpec] = {}
    for name in CELL_RESIDENT_FIELDS:
        specs[name] = FieldSpec(
            name,
            "cell",
            "cell_resident",
            moves_with_cell=True,
            clears_on_death=True,
            layout_group="cell_movable",
            audit_role="state",
            record_role="cell",
            copy_on_reproduction=True,
        )
    for name in FIELD_RESIDENT_FIELDS:
        specs[name] = FieldSpec(
            name,
            "cell",
            "field_resident",
            moves_with_cell=False,
            clears_on_death=False,
            layout_group="environment",
            audit_role="field",
            record_role="environment",
        )
    for name in CHANNEL_RESIDENT_FIELDS:
        specs[name] = FieldSpec(
            name,
            "channel",
            "cell_resident",
            moves_with_cell=True,
            clears_on_death=True,
            layout_group="channel_movable",
            audit_role="communication",
            record_role="channel",
            copy_on_reproduction=True,
        )
    for name in ACTION_RESIDENT_FIELDS:
        specs[name] = FieldSpec(
            name,
            "action",
            "cell_resident",
            moves_with_cell=False,
            clears_on_death=True,
            layout_group="action_ephemeral",
            visual_role="action",
            audit_role="decision",
            record_role="action",
        )
    for name in RAQIC_CELL_FIELDS:
        specs[name] = FieldSpec(
            name,
            "cell",
            "cell_resident",
            moves_with_cell=False,
            clears_on_death=True,
            layout_group="raqic_ephemeral",
            visual_role="raqic",
            audit_role="quantum",
            record_role="raqic",
        )
    for name in ACTION_TRANSITION_CELL_FIELDS:
        specs[name] = FieldSpec(
            name,
            "cell",
            "cell_resident",
            moves_with_cell=True,
            clears_on_death=True,
            layout_group="action_transition_movable",
            audit_role="agent_visible_action_context",
            record_role="action_transition",
            # A newborn did not acquire its parent's active observation or
            # pre-choice target context.
            copy_on_reproduction=False,
        )
    for name in ("raqic_patch_action_phase", "raqic_patch_action_coherence"):
        specs[name] = FieldSpec(
            name,
            "patch",
            "patch_resident",
            dtype="float64",
            moves_with_cell=False,
            clears_on_death=False,
            layout_group="raqic_phase_context",
            audit_role="decision_context",
            record_role="patch_optional",
        )
    for name in ("raqic_global_action_phase", "raqic_global_action_coherence"):
        specs[name] = FieldSpec(
            name,
            "global",
            "global",
            dtype="float64",
            moves_with_cell=False,
            clears_on_death=False,
            layout_group="raqic_phase_context",
            audit_role="decision_context",
            record_role="global_optional",
        )
    return specs


FIELD_REGISTRY = build_field_registry()


def fields_that_move_with_cell() -> tuple[str, ...]:
    return tuple(name for name, spec in FIELD_REGISTRY.items() if spec.moves_with_cell)


def fields_cleared_on_death() -> tuple[str, ...]:
    return tuple(name for name, spec in FIELD_REGISTRY.items() if spec.clears_on_death)
