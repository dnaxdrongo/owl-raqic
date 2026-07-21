"""Pydantic configuration schema for all Observer-Window Life coefficients.

Configuration is the only place where global simulation coefficients should live.
Engine code receives a validated :class:`SimulationConfig` and should not rely on
module-level magic numbers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class OWLBaseModel(BaseModel):
    """Base class for strict project configuration models."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class WorldConfig(OWLBaseModel):
    """Spatial and scheduling parameters for the simulated world."""

    height: int = Field(default=100, ge=10)
    width: int = Field(default=100, ge=10)
    seed: int = 42
    boundary_mode: Literal["toroidal", "reflective", "absorbing", "obstacle"] = "toroidal"
    patch_size: int = Field(default=5, ge=1)
    max_steps: int = Field(default=1000, ge=1)

    @model_validator(mode="after")
    def validate_patch_tiling(self) -> WorldConfig:
        """Require exact patch tiling so patch aggregation never silently crops."""
        if self.height % self.patch_size != 0:
            raise ValueError("world.height must be divisible by world.patch_size")
        if self.width % self.patch_size != 0:
            raise ValueError("world.width must be divisible by world.patch_size")
        return self


class ActionConfig(OWLBaseModel):
    """Actualization and action-set parameters.

    Advanced ecological runs default to utility-weighted stochastic actualization.
    Deterministic argmax remains available by setting ``stochastic=False`` for
    controlled tests, but the ordinary baseline/advanced configuration now samples
    from normalized survival-weighted action probabilities.
    """

    beta: float = Field(default=2.0, gt=0)
    stochastic: bool = True
    epsilon: float = Field(default=1e-8, gt=0)
    enabled_actions: list[str] = Field(default_factory=list)

    # Select movement as a macro-action before compiling a direction from
    # survival, environmental, and inertia weights.
    movement_macro_enabled: bool = True
    diagonal_movement_enabled: bool = True
    utility_weighted_sampling: bool = True
    action_temperature: float = Field(default=0.50, gt=0)
    movement_temperature: float = Field(default=0.65, gt=0)
    movement_persistence_bonus: float = Field(default=0.18, ge=0)
    movement_reverse_penalty: float = Field(default=0.35, ge=0)
    movement_food_weight: float = Field(default=0.65, ge=0)
    movement_toxin_weight: float = Field(default=0.80, ge=0)
    movement_crowding_weight: float = Field(default=0.45, ge=0)
    movement_hunger_target: float = Field(default=0.35, gt=0, le=1)
    movement_macro_normalization: float = Field(default=0.75, ge=0, le=1)


class DirectionScoreWeights(OWLBaseModel):
    """Explicit coefficients for a high-level movement compiler."""

    distance: float = Field(default=1.0, ge=0.0)
    hazard: float = Field(default=0.75, ge=0.0)
    cost: float = Field(default=0.25, ge=0.0)
    opportunity: float = Field(default=0.20, ge=0.0)


class ActionTransitionConfig(OWLBaseModel):
    """Configure the completed SENSE, FLEE, and PURSUE action transitions.

    The default is the certified  baseline behavior.  No compatibility run gains
    new observations, arrays, costs, masks, or transitions unless ``enabled``
    and the v1 contract are selected explicitly.
    """

    enabled: bool = False
    action_contract_version: Literal["v099_legacy_unsupported", "owl.action-transitions.v1"] = (
        "v099_legacy_unsupported"
    )
    legacy_unsupported_action_recovery: bool = True
    active_sense_enabled: bool = False
    active_sense_ordinary_radius: int = Field(default=1, ge=1, le=4)
    active_sense_radius_bonus: int = Field(default=1, ge=1, le=4)
    active_sense_noise_multiplier: float = Field(default=1.0, ge=0.0, le=1.0)
    active_sense_memory_persistence: int = Field(default=1, ge=1, le=16)
    active_sense_cost: float = Field(default=0.005, ge=0.0)
    flee_execution_enabled: bool = False
    pursue_execution_enabled: bool = False
    target_sense_radius: int = Field(default=2, ge=1, le=4)
    perceived_threat_threshold: float = Field(default=0.10, ge=0.0, le=1.0)
    pursuit_trait_threshold: float = Field(default=0.10, ge=0.0, le=1.0)
    flee_target_policy: Literal["highest_perceived_threat_then_nearest"] = (
        "highest_perceived_threat_then_nearest"
    )
    pursue_target_policy: Literal["nearest_visible_living_target"] = "nearest_visible_living_target"
    flee_score_weights: DirectionScoreWeights = Field(
        default_factory=lambda: DirectionScoreWeights(
            distance=1.0, hazard=0.85, cost=0.25, opportunity=0.20
        )
    )
    pursue_score_weights: DirectionScoreWeights = Field(
        default_factory=lambda: DirectionScoreWeights(
            distance=1.0, hazard=0.70, cost=0.25, opportunity=0.60
        )
    )
    high_level_movement_tie_break: Literal["immutable_direction_order"] = (
        "immutable_direction_order"
    )

    @model_validator(mode="after")
    def validate_transition_mode(self) -> ActionTransitionConfig:
        v1 = self.action_contract_version == "owl.action-transitions.v1"
        any_transition = (
            self.active_sense_enabled
            or self.flee_execution_enabled
            or self.pursue_execution_enabled
        )
        if self.enabled != v1:
            raise ValueError(
                "action_transitions.enabled must exactly match the v1 contract selection"
            )
        if self.enabled and self.legacy_unsupported_action_recovery:
            raise ValueError("v1 action transitions cannot enable legacy unsupported recovery")
        if not self.enabled and any_transition:
            raise ValueError("legacy recovery cannot enable completed action transitions")
        if self.active_sense_enabled and self.active_sense_cost <= 0.0:
            raise ValueError("active SENSE requires a positive authoritative cost")
        return self


class CounterfactualConfig(OWLBaseModel):
    """Configure bounded counterfactual rollouts.

    The default is inert so loading an existing configuration cannot create
    a source observer, allocate branch state, or change factual execution.
    Active execution is intentionally limited to the certified segmented,
    single-device action-transition contract.
    """

    enabled: bool = False
    source_boundary: Literal["post_selection_pre_actions"] = "post_selection_pre_actions"
    source_selection_mode: Literal["explicit", "deterministic_hash", "action_family_stratified"] = (
        "deterministic_hash"
    )
    explicit_source_decision_ids: tuple[str, ...] = ()
    max_source_ticks: int = Field(default=1, ge=1)
    max_source_decisions: int = Field(default=32, ge=1)
    repeats: int = Field(default=1, ge=1)
    horizons: tuple[int, ...] = (1, 3, 5, 10)
    family_horizons: dict[str, tuple[int, ...]] = Field(
        default_factory=lambda: {
            "SENSE": (25,),
            "COMMUNICATE": (25,),
            "FLEE": (25,),
            "PURSUE": (25,),
            "REPRODUCE": (25,),
            "SPLIT": (25,),
            "MERGE": (25,),
        }
    )
    include_selected_anchor: bool = True
    executable_only: bool = True
    emit_nonexecutable_candidates: bool = True
    backend: Literal["auto", "numpy", "cupy"] = "auto"
    branch_execution_mode: Literal["segmented", "target_gpu_required"] = "segmented"
    max_active_branches: int = Field(default=8, ge=1)
    stream_lanes: int = Field(default=2, ge=1)
    source_pool_capacity: int = Field(default=2, ge=1)
    max_device_bytes: int = Field(default=8 * 1024**3, ge=1)
    memory_safety_fraction: float = Field(default=0.70, gt=0.0, le=0.95)
    event_mode: Literal["exact", "summary"] = "exact"
    event_capacity_per_branch_tick: int = Field(default=4096, ge=1)
    strict_overflow: bool = True
    state_hash_algorithm: Literal["owl.device-state-merkle-sha256.v1"] = (
        "owl.device-state-merkle-sha256.v1"
    )
    rng_registry_version: Literal["owl.counterfactual-rng-registry.v1"] = (
        "owl.counterfactual-rng-registry.v1"
    )
    branch_seed_derivation_version: Literal["owl.branch-seed-sha256.v1"] = (
        "owl.branch-seed-sha256.v1"
    )
    fail_on_alias: bool = True
    fail_on_factual_mutation: bool = True
    require_anchor_equivalence: bool = True
    writer_queue_depth: int = Field(default=2, ge=1)
    max_packet_bytes: int = Field(default=64 * 1024**2, ge=1)
    max_pending_bytes: int = Field(default=128 * 1024**2, ge=1)
    parquet_row_group_rows: int = Field(default=65_536, ge=1)
    allow_per_ow_qiskit: bool = False

    @field_validator("horizons")
    @classmethod
    def validate_horizons(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value or any(item <= 0 for item in value):
            raise ValueError("counterfactual horizons must be non-empty and positive")
        if tuple(sorted(set(value))) != value:
            raise ValueError("counterfactual horizons must be unique and sorted")
        return value

    @field_validator("family_horizons")
    @classmethod
    def validate_family_horizons(
        cls, value: dict[str, tuple[int, ...]]
    ) -> dict[str, tuple[int, ...]]:
        normalized: dict[str, tuple[int, ...]] = {}
        for family, horizons in value.items():
            key = str(family).upper()
            if not horizons or any(item <= 0 for item in horizons):
                raise ValueError(f"family horizons for {key} must be positive")
            if tuple(sorted(set(horizons))) != tuple(horizons):
                raise ValueError(f"family horizons for {key} must be unique and sorted")
            normalized[key] = tuple(horizons)
        return normalized

    @model_validator(mode="after")
    def validate_phase3_resources(self) -> CounterfactualConfig:
        if self.source_selection_mode == "explicit" and not self.explicit_source_decision_ids:
            raise ValueError("explicit source selection requires decision IDs")
        if self.event_mode == "exact" and not self.strict_overflow:
            raise ValueError("exact counterfactual events require strict overflow")
        if self.stream_lanes > self.max_active_branches:
            raise ValueError("stream_lanes cannot exceed max_active_branches")
        if self.max_pending_bytes < self.max_packet_bytes:
            raise ValueError("max_pending_bytes must hold at least one packet")
        return self


class ResourceConfig(OWLBaseModel):
    """Physical resource, food, toxin, and metabolic parameters."""

    food_growth: float = Field(default=0.01, ge=0)
    food_decay: float = Field(default=0.002, ge=0)
    food_diffusion: float = Field(default=0.05, ge=0)
    toxin_diffusion: float = Field(default=0.02, ge=0)
    toxin_decay: float = Field(default=0.01, ge=0)
    metabolism_base: float = Field(default=0.003, ge=0)
    movement_cost: float = Field(default=0.01, ge=0)
    feed_efficiency: float = Field(default=0.40, ge=0)
    max_resource: float = Field(default=1.0, gt=0)

    # Resource exhaustion accumulates starvation debt instead of causing an
    # instantaneous death. Feeding assimilates part of the intake immediately
    # and routes the remainder through digestion.
    starvation_grace_ticks: float = Field(default=18.0, gt=0)
    starvation_debt_gain: float = Field(default=0.060, ge=0)
    starvation_debt_recovery: float = Field(default=0.090, ge=0)
    starvation_health_damage: float = Field(default=0.035, ge=0)
    feeding_immediate_fraction: float = Field(default=0.30, ge=0, le=1)
    emergency_feed_threshold: float = Field(default=0.18, ge=0, le=1)
    emergency_feed_boost: float = Field(default=0.80, ge=0)


class EcologyConfig(OWLBaseModel):
    """Advanced ecological dynamics coefficients.

    Disabled by default to preserve the  baseline behavior. When enabled,
    feeding uses Monod saturation, digested mass enters a resource buffer, waste
    can recycle into food, and toxin/starvation damage are nonlinear.
    """

    advanced_enabled: bool = False
    monod_half_saturation: float = Field(default=0.20, gt=0)
    digestion_decay: float = Field(default=0.35, ge=0, le=1)
    digestion_efficiency: float = Field(default=0.75, ge=0, le=1)
    waste_decay: float = Field(default=0.05, ge=0, le=1)
    waste_recycle_rate: float = Field(default=0.15, ge=0)
    food_regrowth_rate: float = Field(default=0.012, ge=0)
    food_carrying_capacity: float = Field(default=1.0, gt=0)
    starvation_threshold: float = Field(default=0.05, ge=0)
    toxin_damage_exponent: float = Field(default=1.35, ge=0.25)
    age_stress_scale: float = Field(default=1000.0, gt=0)
    repair_half_saturation: float = Field(default=0.20, gt=0)
    ingestion_handling_time: float = Field(default=0.75, ge=0)


class PossibilityConfig(OWLBaseModel):
    """Advanced possibility/actualization parameters."""

    advanced_enabled: bool = False
    epistemic_weight: float = Field(default=0.20, ge=0)
    risk_weight: float = Field(default=0.50, ge=0)
    effort_weight: float = Field(default=0.30, ge=0)
    cooldown_weight: float = Field(default=0.40, ge=0)
    entropy_temperature_min: float = Field(default=0.25, gt=0)
    entropy_temperature_max: float = Field(default=5.0, gt=0)
    amplitude_diagnostics: bool = False


class DecisionHomeostasisConfig(OWLBaseModel):
    """Survival-weighted stochastic decision policy coefficients.

    This layer keeps action selection stochastic, but raises precision and
    optimal-action mass under homeostatic urgency while respecting authority.
    """

    enabled: bool = False
    survival_weight: float = Field(default=2.0, ge=0.0)
    viability_weight: float = Field(default=1.25, ge=0.0)
    epistemic_weight_safe: float = Field(default=0.25, ge=0.0)
    epistemic_weight_emergency: float = Field(default=0.03, ge=0.0)
    authority_floor: float = Field(default=1e-6, gt=0.0)
    emergency_precision_min: float = Field(default=2.0, gt=0.0)
    emergency_precision_max: float = Field(default=9.0, gt=0.0)
    safe_precision: float = Field(default=1.5, gt=0.0)
    optimality_margin: float = Field(default=0.20, ge=0.0)
    forced_optimal_probability: float = Field(default=0.80, ge=0.0, le=1.0)
    noetic_bias_scale: float = Field(default=0.30, ge=0.0)
    urgent_threshold: float = Field(default=0.35, ge=0.0, le=1.0)


class CrossScaleHomeostasisConfig(OWLBaseModel):
    """Patch/apex responsiveness to lower-level viability and carrying pressure."""

    enabled: bool = False
    carrying_capacity_enabled: bool = True
    reproduction_pressure_weight: float = Field(default=1.25, ge=0.0)
    starvation_pressure_weight: float = Field(default=1.75, ge=0.0)
    crowding_pressure_weight: float = Field(default=1.25, ge=0.0)
    food_deficit_weight: float = Field(default=1.50, ge=0.0)
    apex_smoothing: float = Field(default=0.80, ge=0.0, le=0.999)
    crisis_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    patch_crisis_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    max_reproduction_suppression: float = Field(default=0.95, ge=0.0, le=1.0)


class IdentityConfig(OWLBaseModel):
    """Globally unique OW identity allocation."""

    enabled: bool = False
    start_id: int = Field(default=1, ge=1)


class HierarchyConfig(OWLBaseModel):
    """Advanced dynamic patch/fractal hierarchy parameters."""

    dynamic_patches: bool = False
    predictive_topdown: bool = False
    centroid_smoothing: float = Field(default=0.35, ge=0, le=1)
    prediction_error_weight: float = Field(default=0.20, ge=0)
    phase_lag_default: float = 0.0


class IntegrationConfig(OWLBaseModel):
    """Coefficients for the bounded observer-window integration functional."""

    weight_memory: float = 1.0
    weight_flexibility: float = 0.8
    weight_synchrony: float = 1.2
    weight_coherence: float = 1.0
    weight_cross_scale: float = 1.0
    weight_resource: float = 0.6
    weight_boundary: float = 0.7
    weight_conflict: float = 1.5
    entropy_target: float = Field(default=0.55, ge=0, le=1)
    entropy_sigma: float = Field(default=0.22, gt=0)


class PhaseConfig(OWLBaseModel):
    """Phase and cross-scale oscillator parameters."""

    base_omega: float = 0.08
    same_scale_coupling: float = 0.05
    parent_coupling: float = 0.03
    phase_noise_sigma: float = Field(default=0.005, ge=0)
    patch_resultant_support_epsilon: float = Field(default=1e-7, ge=0.0, le=1.0)


class CommunicationConfig(OWLBaseModel):
    """Universal communication-channel parameters.

    Every observer window can signal. These coefficients define the physical
    channel substrate, not a dedicated signaler species.
    """

    enabled: bool = True
    num_channels: int = Field(default=8, ge=1)
    diffusion: list[float] = Field(
        default_factory=lambda: [0.08, 0.12, 0.03, 0.05, 0.10, 0.04, 0.02, 0.03]
    )
    decay: list[float] = Field(
        default_factory=lambda: [0.02, 0.06, 0.08, 0.04, 0.05, 0.03, 0.01, 0.04]
    )
    base_emit_cost: float = Field(default=0.005, ge=0)
    trust_lr: float = Field(default=0.03, ge=0)
    receptivity_lr: float = Field(default=0.01, ge=0)
    relation_lr: float = Field(default=0.02, ge=0)
    deception_penalty: float = Field(default=0.10, ge=0)
    source_tracking_enabled: bool = False
    intentional_mix: float = Field(default=0.35, ge=0, le=1)
    source_trust_lr: float = Field(default=0.03, ge=0)
    neighbor_trust_decay: float = Field(default=0.005, ge=0, le=1)

    @field_validator("diffusion", "decay")
    @classmethod
    def nonempty_channel_coefficients(cls, value: list[float]) -> list[float]:
        """Validate channel coefficient lists before model-wide length checks."""
        if not value:
            raise ValueError("channel coefficient lists cannot be empty")
        if any(v < 0 for v in value):
            raise ValueError("channel coefficients must be nonnegative")
        return value


class TopDownConfig(OWLBaseModel):
    """Bounded parent/apex modulation parameters."""

    lambda_threshold: float = Field(default=0.10, ge=0)
    lambda_action_bias: float = Field(default=0.20, ge=0)
    max_parent_control: float = Field(default=0.25, ge=0, le=1)
    apex_update_every: int = Field(default=10, ge=1)


class PredationConfig(OWLBaseModel):
    """Predation and ingestion coefficients."""

    enabled: bool = True
    min_predation_trait: float = Field(default=0.6, ge=0, le=1)
    resource_transfer: float = Field(default=0.7, ge=0, le=1)
    resistance_weight: float = Field(default=1.0, ge=0)
    memory_transfer: float = Field(default=0.05, ge=0, le=1)


class ReproductionConfig(OWLBaseModel):
    """Reproduction, inheritance, and mutation coefficients."""

    enabled: bool = True
    min_resource: float = Field(default=0.75, ge=0, le=1)
    min_health: float = Field(default=0.70, ge=0, le=1)
    min_boundary: float = Field(default=0.60, ge=0, le=1)
    min_integration: float = Field(default=0.50, ge=0, le=1)
    mutation_sigma: float = Field(default=0.03, ge=0)
    channel_mutation_sigma: float = Field(default=0.04, ge=0)
    child_resource_fraction: float = Field(default=0.35, ge=0, le=1)
    initial_child_health: float = Field(default=0.80, ge=0, le=1)
    initial_child_boundary: float = Field(default=0.60, ge=0, le=1)
    memory_inheritance: float = Field(default=0.25, ge=0, le=1)
    advanced_enabled: bool = False
    genome_length: int = Field(default=8, ge=1, le=64)
    recombination_enabled: bool = True
    genotype_mutation_sigma: float = Field(default=0.04, ge=0)
    mate_radius: int = Field(default=1, ge=1, le=5)
    symbiosis_enabled: bool = False


class InitializationConfig(OWLBaseModel):
    """Initial-condition parameters for deterministic world construction.

    These values are deliberately separated from runtime engine coefficients.
    Later experiments can alter starting conditions without changing the update
    equations.
    """

    population_density: float = Field(default=0.35, ge=0.0, le=1.0)
    background_food: float = Field(default=0.02, ge=0.0, le=1.0)
    food_patch_count: int = Field(default=8, ge=0)
    food_patch_radius: int = Field(default=5, ge=1)
    food_patch_intensity: float = Field(default=0.75, ge=0.0, le=1.0)
    obstacle_density: float = Field(default=0.0, ge=0.0, le=1.0)
    toxin_patch_count: int = Field(default=0, ge=0)
    toxin_patch_radius: int = Field(default=4, ge=1)
    toxin_patch_intensity: float = Field(default=0.25, ge=0.0, le=1.0)
    initial_activation_mean: float = Field(default=0.10, ge=0.0, le=1.0)
    initial_activation_sigma: float = Field(default=0.03, ge=0.0)
    initial_memory_mean: float = Field(default=0.05, ge=0.0, le=1.0)
    initial_memory_sigma: float = Field(default=0.02, ge=0.0)
    initial_integration_mean: float = Field(default=0.10, ge=0.0, le=1.0)
    initial_integration_sigma: float = Field(default=0.03, ge=0.0)
    initial_resource_mean: float = Field(default=0.55, ge=0.0, le=1.0)
    initial_resource_sigma: float = Field(default=0.10, ge=0.0)
    initial_health_mean: float = Field(default=0.90, ge=0.0, le=1.0)
    initial_health_sigma: float = Field(default=0.05, ge=0.0)
    initial_boundary_mean: float = Field(default=0.80, ge=0.0, le=1.0)
    initial_boundary_sigma: float = Field(default=0.05, ge=0.0)
    initial_threshold_mean: float = Field(default=0.50, ge=0.0, le=1.0)
    initial_threshold_sigma: float = Field(default=0.05, ge=0.0)
    initial_trust: float = Field(default=1.0, ge=0.0, le=1.0)
    trait_noise_sigma: float = Field(default=0.025, ge=0.0)
    type_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "grazer": 0.72,
            "cooperator": 0.16,
            "proto_carnivore": 0.08,
            "scavenger": 0.04,
        }
    )

    @field_validator("type_weights")
    @classmethod
    def validate_type_weights(cls, value: dict[str, float]) -> dict[str, float]:
        """Require nonnegative type weights with at least one positive entry."""
        if not value:
            raise ValueError("initialization.type_weights cannot be empty")
        if any(weight < 0 for weight in value.values()):
            raise ValueError("initialization.type_weights must be nonnegative")
        if sum(value.values()) <= 0:
            raise ValueError(
                "initialization.type_weights must contain at least one positive weight"
            )
        return value


class CADCRecordingConfig(OWLBaseModel):
    """Additive factual evidence recorder configuration.

    The CADC recorder is observational and disabled by default.  Its limits
    are validated before any device or host buffer is allocated so an exact
    profile cannot silently degrade to sampled evidence.
    """

    enabled: bool = False
    profile: Literal["compact", "exact"] = "compact"
    capture_agent_context: bool = True
    capture_oracle_context: bool = True
    capture_candidates: bool = True
    capture_execution: bool = True
    capture_events: bool = True
    capture_contributions: bool = True
    capture_information: bool = True
    include_dense_context: bool = False
    exact_local_radius: int = Field(default=1, ge=0, le=8)
    device_buffer_slots: int = Field(default=1, ge=1, le=1)
    host_queue_depth: int = Field(default=1, ge=1, le=1)
    max_device_buffer_bytes: int = Field(default=512 * 1024 * 1024, ge=1024 * 1024)
    max_pending_bytes: int = Field(default=512 * 1024 * 1024, ge=1024 * 1024)
    event_capacity_per_tick: int = Field(default=262_144, ge=1)
    max_batch_rows: int = Field(default=131_072, ge=22)
    max_batch_bytes: int = Field(default=128 * 1024 * 1024, ge=1024 * 1024)
    parquet_row_group_rows: int = Field(default=131_072, ge=1)
    table_flush_ticks: int = Field(default=1, ge=1)
    compression: Literal["zstd", "snappy", "gzip", "none"] = "zstd"
    validation: Literal["minimal", "full"] = "full"
    strict_overflow: bool = True

    @model_validator(mode="after")
    def validate_evidence_dependencies(self) -> CADCRecordingConfig:
        """Reject profiles that could make a stronger claim than their inputs."""
        if self.capture_information and not self.capture_agent_context:
            raise ValueError("CADC information capture requires agent-context capture")
        if self.include_dense_context and not self.capture_oracle_context:
            raise ValueError("CADC dense context requires oracle-context capture")
        if self.profile == "exact" and not self.include_dense_context:
            raise ValueError("CADC exact profile requires exact-local dense context")
        if self.profile == "exact" and not self.strict_overflow:
            raise ValueError("CADC exact profile requires strict overflow handling")
        if self.max_batch_bytes > self.max_pending_bytes:
            raise ValueError("CADC max_batch_bytes must not exceed max_pending_bytes")
        return self


class RecordingConfig(OWLBaseModel):
    """Output and recording parameters."""

    enabled: bool = True
    zarr_path: str = "runs/run_001.zarr"
    metrics_path: str = "results/metrics.parquet"
    record_every: int = Field(default=5, ge=1)
    save_fields: list[str] = Field(default_factory=list)
    cadc: CADCRecordingConfig = Field(default_factory=CADCRecordingConfig)


class VisualizationConfig(OWLBaseModel):
    output_resolution: tuple[int, int] | None = None
    """Live visualization parameters.

    The v0.9.6 scientific overlay retains the visual-overhaul schema so this
    file can safely be extracted after the visual-only overlay without erasing
    its configuration contract.
    """

    enabled: bool = True
    backend: Literal["pygame", "none"] = "pygame"
    fps: int = Field(default=30, ge=1)
    scale: int = Field(default=6, ge=1)
    color_by: str = "integration"
    snapshot_backed_replay: bool = False
    max_history_frames: int = Field(default=2000, ge=1)
    show_advanced_overlays: bool = True
    render_every: int = Field(default=1, ge=1)
    event_every: int = Field(default=1, ge=1)
    frame_transfer_policy: Literal["synchronous", "double_buffered", "triple_buffered"] = (
        "double_buffered"
    )
    max_events: int = Field(default=4096, ge=0)
    lod_mode: Literal["automatic", "heatmap", "dots", "glyphs", "full"] = "automatic"
    overlay: str = "life"
    record_frames: bool = False
    adaptive_lod: bool = True
    max_slowdown_fraction: float = Field(default=0.15, ge=0.0, le=1.0)

    # Visualization controls are separate from the decision mathematics.
    # They affect rendering and interpretability only.
    renderer_mode: Literal["legacy", "interpretability"] = "interpretability"
    schedule_mode: Literal["live_adaptive", "record_fixed", "record_keyframes"] = "live_adaptive"
    frames_per_tick: int = Field(default=8, ge=1, le=60)
    fixed_tick_stride: int = Field(default=1, ge=1)
    fail_on_dropped_recording_frame: bool = True
    window_width: int = Field(default=1920, ge=640)
    window_height: int = Field(default=1080, ge=480)
    resizable: bool = True
    viewport_sidebar_width: int = Field(default=320, ge=0)
    camera_mode: Literal["fit", "free", "follow", "cinematic"] = "fit"
    min_zoom: float = Field(default=0.5, gt=0.0)
    max_zoom: float = Field(default=64.0, gt=0.0)
    trait_color_mode: Literal["raw_hex", "perceptual"] = "raw_hex"
    accessibility_mode: Literal[
        "standard", "high_contrast", "deuteranopia", "protanopia", "tritanopia", "monochrome"
    ] = "standard"
    show_environment_sprites: bool = True
    show_action_effects: bool = True
    show_patch_overlay: bool = True
    max_high_detail_effects: int = Field(default=4096, ge=0)
    atlas_max_entries: int = Field(default=8192, ge=128)
    visual_theme_seed: int = 0

    @model_validator(mode="after")
    def validate_interpretability_view(self) -> VisualizationConfig:
        if self.window_width <= self.viewport_sidebar_width:
            raise ValueError("visualization.window_width must exceed viewport_sidebar_width")
        if self.max_zoom < self.min_zoom:
            raise ValueError("visualization.max_zoom must be greater than or equal to min_zoom")
        if self.schedule_mode == "record_fixed" and self.adaptive_lod:
            # Fixed cadence and adaptive scene detail can coexist; only cadence
            # adaptation is forbidden. Existing adaptive_lod controls detail.
            pass
        return self


ActualizationVariant = Literal[
    "stable_baseline",
    "utility_innovation",
    "fractal_resonance",
    "phase_interference",
]


class RAQICConfig(OWLBaseModel):
    """Configuration for the RAQIC decision substrate.

    Disabled by default so compatibility OWL remains the baseline path. When enabled
    with ``decision_policy='raqic'``, RAQIC replaces the probability/readout
    decision substrate while OWL still applies physical consequences.
    """

    enabled: bool = False
    # Optional actualization extensions use zero-valued defaults.
    # Zero values preserve the baseline decision behavior.
    actualization_variant: ActualizationVariant = "stable_baseline"
    utility_coupling: float = Field(default=0.0, ge=0.0)
    utility_projection_epsilon: float = Field(default=1e-8, gt=0.0)
    utility_bound_floor: float = Field(default=1.0, gt=0.0)
    phase_resonance_coupling: float = Field(default=0.0, ge=0.0)
    phase_resonance_patch_weight: float = Field(default=0.75, ge=0.0)
    phase_resonance_global_weight: float = Field(default=0.25, ge=0.0)
    phase_resonance_support_epsilon: float = Field(default=1e-10, gt=0.0)
    interference_mixer_strength: float = Field(default=0.0, ge=0.0)
    interference_trotter_steps: int = Field(default=1, ge=1)
    interference_action_graph: Literal["semantic_families_v1"] = "semantic_families_v1"
    experimental_shadow_only: bool = False
    record_actualization_diagnostics: bool = False

    mode: Literal[
        "cpu_audit",
        "cpu_qiskit",
        "dynamic",
        "deferred",
        "hybrid",
        "walk",
        "gpu_batch",
        "gpu_hybrid_audit",
        "gpu_full",
        "gpu_full_hybrid_audit",
    ] = "cpu_audit"
    decision_policy: Literal["legacy", "raqic", "hybrid_compare"] = "legacy"
    epsilon_raqic: float = Field(default=1.0, ge=0.0)
    epsilon_adelic: float = Field(default=1.0, ge=0.0)
    beta_intention: float = Field(default=1.0, ge=0.0)
    action_temperature: float = Field(default=1.0, gt=0.0)
    rounds_per_tick: int = Field(default=1, ge=0)
    shots: int = Field(default=1024, ge=1)
    active_primes: tuple[int, ...] = (2, 3, 5)
    prime_weights: dict[int, float] = Field(default_factory=lambda: {2: 0.25, 3: 0.15, 5: 0.10})
    use_qiskit_for_all: bool = False
    qiskit_subset_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    qiskit_debug_ow_limit: int = Field(default=0, ge=0)
    batch_by_feature_signature: bool = True
    cache_templates: bool = True
    persist_quantum_state: bool = True
    store_density_diagnostics: bool = False
    record_measurement_records: bool = True
    record_action_probabilities: bool = True
    fallback_on_backend_error: bool = True
    assert_recovery_gates: bool = True
    parent_intention_eta: float = Field(default=0.25, ge=0.0, le=1.0)
    max_cells_per_tick: int | None = Field(default=None, ge=1)
    debug_store_full_records: bool = False
    # GPU controls are explicit so CPU reference behavior remains unchanged
    # unless a GPU execution mode is selected.
    gpu_backend: Literal["cupy"] = "cupy"
    gpu_precision: Literal["audit64", "mixed", "balanced32", "fast32"] = "audit64"
    strict_gpu: bool = True
    gpu_all_cells_required: bool = True
    gpu_transfer_policy: Literal["stage_once", "persistent_mirror"] = "stage_once"
    gpu_profile: bool = True
    gpu_audit_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    gpu_audit_limit: int = Field(default=32, ge=0)
    gpu_validate_qiskit: bool = False
    gpu_validate_cpu: bool = False
    gpu_memory_limit_mb: float | None = Field(default=None, ge=1)
    qiskit_gpu_method: Literal["statevector", "density_matrix", "tensor_network"] = "statevector"
    qiskit_gpu_device: Literal["GPU", "CPU"] = "GPU"
    qiskit_batched_shots_gpu: bool = False
    qiskit_enable_cuStateVec: bool = False
    dense_signature_grouping: bool = False
    gpu_chunk_size: int | None = Field(default=None, ge=1)
    gpu_probability_tolerance: float = Field(default=1e-8, gt=0.0)
    gpu_kl_tolerance: float = Field(default=1e-7, gt=0.0)

    # Full-stack GPU controls are active only for the matching execution mode.
    # gpu_full or gpu_full_hybrid_audit.
    full_gpu_enabled: bool = False
    full_gpu_strict: bool = True
    full_gpu_backend: Literal["cupy"] = "cupy"
    full_gpu_transfer_policy: Literal["stage_once", "persistent_mirror", "hybrid_shadow"] = (
        "stage_once"
    )
    full_gpu_precision: Literal["audit64", "mixed", "balanced32", "fast32"] = "audit64"
    full_gpu_physical_modules: tuple[str, ...] = (
        "environment",
        "sensing",
        "utility",
        "authority",
        "movement",
        "collision",
        "feeding",
        "health",
        "communication",
        "memory",
        "phase",
        "integration",
        "aggregation",
        "topdown",
        "reproduction",
        "death",
        "topology",
        "recording",
        "visualization",
    )
    full_gpu_sparse_event_capacity: int = Field(default=4096, ge=1)
    full_gpu_movement_conflict_policy: Literal["sort_priority"] = "sort_priority"
    full_gpu_reproduction_conflict_policy: Literal["sort_priority"] = "sort_priority"
    full_gpu_visual_backend: Literal["none", "pygame_copy", "vispy_gpu", "headless_export"] = "none"
    full_gpu_recording_level: Literal["summary_gpu", "sampled_gpu", "full_gpu_snapshot"] = (
        "summary_gpu"
    )
    full_gpu_profile: bool = True
    full_gpu_audit_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    full_gpu_cpu_shadow_ticks: int = Field(default=0, ge=0)
    full_gpu_no_silent_fallback: bool = True

    # Persistent and graph controls are active only for matching GPU modes.
    # explicitly selected by gpu_full configs/scripts.
    full_gpu_execution_tier: Literal["reference", "persistent", "graph"] = "reference"
    full_gpu_memory_policy: Literal["preallocate", "elastic", "conservative"] = "elastic"
    full_gpu_memory_safety_fraction: float = Field(default=0.85, gt=0.0, le=1.0)
    full_gpu_graph_mode: Literal["off", "segments", "full_tick"] = "off"
    full_gpu_graph_warmup_ticks: int = Field(default=2, ge=0)
    full_gpu_stencil_backend: Literal["vectorized", "raw_toroidal", "auto"] = "vectorized"
    full_gpu_fuse_biology: bool = False
    full_gpu_fuse_scatter: bool = False
    full_gpu_phase_mode: Literal["scalar_reference", "canonical_device"] = "scalar_reference"
    full_gpu_phase_policy: Literal["always", "audit_or_visual", "skip"] = "audit_or_visual"
    full_gpu_policy_backend: Literal["stable", "legacy"] = "stable"
    full_gpu_recording_level_v07: Literal[
        "metrics_only",
        "metrics_plus_events",
        "sampled_cells",
        "patch_summaries",
        "full_snapshot_decimated",
        "debug_full_every_tick",
    ] = "metrics_plus_events"
    full_gpu_render_every: int = Field(default=10, ge=1)
    full_gpu_record_every: int = Field(default=1, ge=1)
    full_gpu_writer_queue_capacity: int = Field(default=1024, ge=1)
    full_gpu_writer_overflow_policy: Literal["block", "raise"] = "block"
    full_gpu_visual_event_capacity: int = Field(default=16384, ge=1)
    full_gpu_sprite_theme: str = "owl_dark_neon"
    full_gpu_visual_clutter_budget: int = Field(default=2048, ge=1)
    full_gpu_benchmark_label: str = "v0.8"

    # Execution, validation, provenance, and visualization controls.
    full_gpu_metric_every: int = Field(default=1, ge=1)
    full_gpu_checkpoint_every: int = Field(default=0, ge=0)
    full_gpu_validation_every: int = Field(default=0, ge=0)
    full_gpu_graph_allow_fallback: bool = False
    full_gpu_qiskit_strict: bool = True
    full_gpu_qiskit_allow_cpu_fallback: bool = False
    full_gpu_run_class: Literal["validation", "production", "exploratory", "stress"] = "validation"
    full_gpu_enable_numerical_ledger: bool = True
    full_gpu_command_capacity: int = Field(default=1024, ge=1)
    full_gpu_certification_required: bool = False
    full_gpu_memory_preflight: bool = True
    full_gpu_visual_adaptive_lod: bool = True
    full_gpu_visual_max_slowdown_fraction: float = Field(default=0.15, ge=0.0, le=1.0)
    qiskit_shot_branching_enable: bool = False
    qiskit_runtime_parameter_bind_enable: bool = False
    qiskit_runtime_binding_policy: Literal["required_native", "concrete_reference", "disabled"] = (
        "required_native"
    )
    qiskit_state_preparation_strategy: Literal["exact_native_rotation_tree"] = (
        "exact_native_rotation_tree"
    )
    qiskit_allow_automatic_execution_fallback: bool = False
    qiskit_require_transpiled_native_instructions: bool = True
    qiskit_preflight_required: bool = True
    qiskit_preflight_batch_size: int = Field(default=8, ge=1, le=64)
    qiskit_preflight_tolerance: float = Field(default=1e-10, gt=0.0)
    qiskit_preflight_cache: bool = True
    qiskit_validation_max_qubits: int = Field(default=28, ge=1)
    qiskit_validation_shots: int = Field(default=4096, ge=1)

    # Production orchestration and Qiskit execution controls.
    qiskit_decision_mode: Literal[
        "off",
        "validation_sample",
        "every_ow_static_exact",
        "every_ow_dynamic_shots",
        "every_ow_circuit_family",
    ] = "off"
    qiskit_circuit_families: tuple[
        Literal[
            "static",
            "deferred",
            "dynamic_recursive",
            "walk",
            "density_noise",
            "interference",
        ],
        ...,
    ] = ("static",)
    qiskit_authoritative_family: Literal[
        "static", "deferred", "dynamic_recursive", "walk", "density_noise", "interference"
    ] = "static"
    qiskit_readout_policy: Literal["deterministic_sample", "argmax", "first_shot"] = (
        "deterministic_sample"
    )
    qiskit_target_gpus: tuple[int, ...] = ()
    qiskit_chunk_size: int = Field(default=64, ge=1)
    qiskit_job_queue_depth: int = Field(default=2, ge=1)
    qiskit_confirm_expensive: bool = False
    full_gpu_graph_requirement: Literal["allow_partial", "full_tick"] = "allow_partial"
    full_gpu_devices: tuple[int, ...] = ()
    full_gpu_multi_gpu: bool = False
    full_gpu_distributed_timeout_seconds: float = Field(default=120.0, gt=0.0)
    full_gpu_shadow_strict: bool = True
    full_gpu_shadow_tolerance: float = Field(default=1e-8, gt=0.0)
    full_gpu_shadow_reference: Literal[
        "implementation_numpy", "scientific_cpu", "dense_numpy_exact", "legacy_cpu_semantic"
    ] = "scientific_cpu"
    full_gpu_implementation_shadow_required: bool = False
    full_gpu_certification_dir: str = "certifications"
    full_gpu_production_marker: str = "READY_FOR_PRODUCTION"

    @field_validator("full_gpu_shadow_reference", mode="before")
    @classmethod
    def normalize_shadow_reference(cls, value: str) -> str:
        aliases = {
            "dense_numpy_exact": "implementation_numpy",
            "legacy_cpu_semantic": "scientific_cpu",
        }
        return aliases.get(str(value), str(value))

    @field_validator("active_primes")
    @classmethod
    def active_primes_are_prime(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if len(set(value)) != len(value):
            raise ValueError("raqic.active_primes must be unique")
        for p in value:
            if p < 2:
                raise ValueError("active primes must be >= 2")
            for q in range(2, int(p**0.5) + 1):
                if p % q == 0:
                    raise ValueError(f"{p} is not prime")
        return value

    @model_validator(mode="after")
    def validate_raqic(self) -> RAQICConfig:
        for p in self.active_primes:
            self.prime_weights.setdefault(p, 1.0 / max(len(self.active_primes), 1))
        if any(not (float("-inf") < float(w) < float("inf")) for w in self.prime_weights.values()):
            raise ValueError("raqic.prime_weights must be finite")
        if self.use_qiskit_for_all and self.mode == "cpu_audit":
            raise ValueError("use_qiskit_for_all requires a qiskit-capable mode")
        if (
            self.mode in ("gpu_batch", "gpu_hybrid_audit", "gpu_full", "gpu_full_hybrid_audit")
            and self.max_cells_per_tick is not None
            and self.gpu_all_cells_required
        ):
            # A configured cap is valid only when it is not pretending to be an all-cell run.
            # The runtime will raise if eligible cells exceed this cap.
            pass
        if self.mode in ("gpu_full", "gpu_full_hybrid_audit") and (
            self.full_gpu_no_silent_fallback
            and self.fallback_on_backend_error
            and self.mode == "gpu_full"
        ):
            # Non-fatal at validation time: scripts may deliberately set fallback
            # for CPU smoke tests. Runtime strict mode records fallback status.
            pass
        if self.actualization_variant == "stable_baseline" and any(
            value != 0.0
            for value in (
                self.utility_coupling,
                self.phase_resonance_coupling,
                self.interference_mixer_strength,
            )
        ):
            raise ValueError("stable_baseline requires all extension couplings to be zero")
        weight_sum = self.phase_resonance_patch_weight + self.phase_resonance_global_weight
        if abs(weight_sum - 1.0) > 1e-12:
            raise ValueError("phase resonance patch/global weights must sum to one")
        needs_phase = self.phase_resonance_coupling > 0.0 or self.interference_mixer_strength > 0.0
        if needs_phase and self.full_gpu_phase_policy == "skip":
            raise ValueError("phase resonance/interference requires RAQIC phase computation")
        if self.actualization_variant == "utility_innovation":
            if self.utility_coupling <= 0.0:
                raise ValueError("utility_innovation requires utility_coupling > 0")
            if self.phase_resonance_coupling != 0.0 or self.interference_mixer_strength != 0.0:
                raise ValueError("utility_innovation does not enable resonance or interference")
        if self.actualization_variant == "fractal_resonance":
            if self.utility_coupling <= 0.0 or self.phase_resonance_coupling <= 0.0:
                raise ValueError(
                    "fractal_resonance requires positive utility and resonance couplings"
                )
            if self.interference_mixer_strength != 0.0:
                raise ValueError("fractal_resonance does not enable interference")
        if self.actualization_variant == "phase_interference":
            if self.utility_coupling <= 0.0:
                raise ValueError("phase_interference requires utility_coupling > 0")
            if self.interference_mixer_strength <= 0.0:
                raise ValueError("phase_interference requires interference_mixer_strength > 0")
        if self.experimental_shadow_only and self.actualization_variant == "stable_baseline":
            raise ValueError("experimental_shadow_only requires a nonbaseline variant")
        if self.full_gpu_run_class in {"validation", "production"}:
            safety_limits = {
                "utility_coupling": (self.utility_coupling, 0.25),
                "phase_resonance_coupling": (self.phase_resonance_coupling, 0.25),
                "interference_mixer_strength": (self.interference_mixer_strength, 0.15),
            }
            for name, (value, ceiling) in safety_limits.items():
                if float(value) > ceiling:
                    raise ValueError(
                        f"{name} exceeds the registered {self.full_gpu_run_class} "
                        f"ceiling of {ceiling}; use exploratory run class explicitly"
                    )
        if self.qiskit_authoritative_family not in self.qiskit_circuit_families:
            raise ValueError(
                "raqic.qiskit_authoritative_family must be listed in qiskit_circuit_families"
            )
        if self.qiskit_runtime_parameter_bind_enable:
            if self.qiskit_runtime_binding_policy == "disabled":
                raise ValueError(
                    "qiskit_runtime_parameter_bind_enable requires a non-disabled policy"
                )
            if (
                self.qiskit_runtime_binding_policy == "required_native"
                and self.qiskit_allow_automatic_execution_fallback
            ):
                raise ValueError(
                    "required_native runtime binding forbids automatic execution fallback"
                )
            if (
                self.qiskit_runtime_binding_policy == "required_native"
                and self.full_gpu_qiskit_allow_cpu_fallback
            ):
                raise ValueError("required_native runtime binding forbids Qiskit CPU fallback")
        if len(set(self.qiskit_target_gpus)) != len(self.qiskit_target_gpus):
            raise ValueError("raqic.qiskit_target_gpus must be unique")
        if len(set(self.full_gpu_devices)) != len(self.full_gpu_devices):
            raise ValueError("raqic.full_gpu_devices must be unique")
        if self.full_gpu_multi_gpu and len(self.full_gpu_devices) < 2:
            raise ValueError("full_gpu_multi_gpu requires at least two full_gpu_devices")
        if (
            self.full_gpu_graph_requirement == "full_tick"
            and self.full_gpu_execution_tier != "graph"
        ):
            raise ValueError(
                "full_gpu_graph_requirement='full_tick' requires execution tier 'graph'"
            )
        return self


class DebugConfig(OWLBaseModel):
    """Configure diagnostic and debugging switches."""

    assert_invariants: bool = False
    profile: bool = False


class SimulationConfig(OWLBaseModel):
    """Top-level validated configuration for a simulation run."""

    world: WorldConfig = Field(default_factory=WorldConfig)
    actions: ActionConfig = Field(default_factory=ActionConfig)
    action_transitions: ActionTransitionConfig = Field(default_factory=ActionTransitionConfig)
    counterfactual: CounterfactualConfig = Field(default_factory=CounterfactualConfig)
    resources: ResourceConfig = Field(default_factory=ResourceConfig)
    ecology: EcologyConfig = Field(default_factory=EcologyConfig)
    possibility: PossibilityConfig = Field(default_factory=PossibilityConfig)
    hierarchy: HierarchyConfig = Field(default_factory=HierarchyConfig)
    integration: IntegrationConfig = Field(default_factory=IntegrationConfig)
    phase: PhaseConfig = Field(default_factory=PhaseConfig)
    communication: CommunicationConfig = Field(default_factory=CommunicationConfig)
    topdown: TopDownConfig = Field(default_factory=TopDownConfig)
    predation: PredationConfig = Field(default_factory=PredationConfig)
    reproduction: ReproductionConfig = Field(default_factory=ReproductionConfig)
    decision_homeostasis: DecisionHomeostasisConfig = Field(
        default_factory=DecisionHomeostasisConfig
    )
    cross_scale_homeostasis: CrossScaleHomeostasisConfig = Field(
        default_factory=CrossScaleHomeostasisConfig
    )
    identity: IdentityConfig = Field(default_factory=IdentityConfig)
    initialization: InitializationConfig = Field(default_factory=InitializationConfig)
    recording: RecordingConfig = Field(default_factory=RecordingConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    raqic: RAQICConfig = Field(default_factory=RAQICConfig)
    debug: DebugConfig = Field(default_factory=DebugConfig)

    @model_validator(mode="after")
    def validate_channel_vector_lengths(self) -> SimulationConfig:
        """Require diffusion/decay vectors to match the configured channel count."""
        n = self.communication.num_channels
        if len(self.communication.diffusion) != n:
            raise ValueError("communication.diffusion length must equal communication.num_channels")
        if len(self.communication.decay) != n:
            raise ValueError("communication.decay length must equal communication.num_channels")
        counterfactual = self.counterfactual
        if counterfactual.enabled:
            if not self.action_transitions.enabled:
                raise ValueError("counterfactual execution requires owl.action-transitions.v1")
            if self.action_transitions.action_contract_version != "owl.action-transitions.v1":
                raise ValueError("counterfactual execution requires the v1 action contract")
            if self.action_transitions.legacy_unsupported_action_recovery:
                raise ValueError("counterfactual execution forbids legacy unsupported recovery")
            if (
                self.raqic.full_gpu_graph_requirement == "full_tick"
                or self.raqic.full_gpu_multi_gpu
            ):
                raise ValueError("Phase 3 v1 requires segmented single-device execution")
            if (
                counterfactual.branch_execution_mode == "target_gpu_required"
                and counterfactual.backend == "numpy"
            ):
                raise ValueError("target-GPU-required counterfactuals cannot use NumPy")
        return self


_CONFIG_MIGRATIONS: dict[str, tuple[str, str]] = {
    "full_gpu_validation_limit": ("gpu_audit_limit", "0.9"),
    "full_gpu_recording_level_v07": ("full_gpu_recording_level_v07", "0.9"),
}


def _normalize_config_data(data: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Apply deterministic, recorded configuration migrations.

    Unknown fields remain Pydantic errors. Migrations are deliberately narrow so
    supported configuration files can be loaded without silently ignoring typos.
    """
    import copy

    normalized = copy.deepcopy(data)
    migrations: list[dict[str, str]] = []
    raqic = normalized.get("raqic")
    if isinstance(raqic, dict) and "full_gpu_validation_limit" in raqic:
        if "gpu_audit_limit" in raqic:
            raise ValueError(
                "configuration defines both deprecated full_gpu_validation_limit "
                "and replacement gpu_audit_limit"
            )
        raqic["gpu_audit_limit"] = raqic.pop("full_gpu_validation_limit")
        migrations.append(
            {
                "from": "raqic.full_gpu_validation_limit",
                "to": "raqic.gpu_audit_limit",
                "since": "0.9",
            }
        )
    return normalized, migrations


def normalize_config(path: str | Path) -> tuple[SimulationConfig, list[dict[str, str]]]:
    """Load, migrate, and validate a configuration with a migration ledger."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"configuration file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if raw is None:
        raise ValueError(f"configuration file is empty: {cfg_path}")
    if not isinstance(raw, dict):
        raise TypeError(f"configuration root must be a mapping, got {type(raw).__name__}")
    normalized, migrations = _normalize_config_data(raw)
    return SimulationConfig.model_validate(normalized), migrations


def load_config(path: str | Path) -> SimulationConfig:
    """Load YAML/JSON configuration into a validated :class:`SimulationConfig`.

    Parameters
    ----------
    path:
        Path to a YAML or JSON configuration file.

    Returns
    -------
    SimulationConfig
        Fully validated configuration object.
    """
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"configuration file not found: {cfg_path}")

    cfg, _migrations = normalize_config(cfg_path)
    return cfg


def save_config_schema(path: str | Path) -> None:
    """Export the :class:`SimulationConfig` JSON schema.

    The schema is useful for editor validation, CI validation, and future config
    tooling. Parent directories are created automatically.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    schema = SimulationConfig.model_json_schema()
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(schema, handle, indent=2, sort_keys=True)
        handle.write("\n")
