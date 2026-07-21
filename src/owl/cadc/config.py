"""Strict analysis-only configuration for CADC-MORE 2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, model_validator

from owl.cadc.schema import (
    EXPECTED_COUNTERFACTUAL_DIGEST,
    EXPECTED_FACTUAL_V2_DIGEST,
    EXPECTED_PHASE3_CLASSIFICATION,
    EXPECTED_PHASE3_SOURCE_SHA256,
    EXPECTED_RNG_REGISTRY_DIGEST,
    HardwareTarget,
)
from owl.core.config import OWLBaseModel


class Phase3InputConfig(OWLBaseModel):
    """Store the exact counterfactual engine, certificate, and schema identities."""

    immutable_engine_root: str = ""
    roots: tuple[str, ...] = ()
    factual_roots: tuple[str, ...] = ()
    required_source_sha256: str = EXPECTED_PHASE3_SOURCE_SHA256
    required_certificate_classification: str = EXPECTED_PHASE3_CLASSIFICATION
    factual_v2_digest: str = EXPECTED_FACTUAL_V2_DIGEST
    counterfactual_schema_digest: str = EXPECTED_COUNTERFACTUAL_DIGEST
    rng_registry_digest: str = EXPECTED_RNG_REGISTRY_DIGEST
    phase3_certificate: str = ""
    phase25_certificate: str = ""
    phase25_hardening_receipt: str = ""
    base_phase3_config: str = "configs/cadc_phase3_phase25_h100_acceptance.yaml"


class CorpusConfig(OWLBaseModel):
    """Define the stratified modeling corpus and sealed confirmatory seeds."""

    development_seeds: tuple[int, ...] = ()
    validation_seeds: tuple[int, ...] = ()
    calibration_seeds: tuple[int, ...] = ()
    reserved_phase5_seeds: tuple[int, ...] = ()
    reserved_phase6_seeds: tuple[int, ...] = ()
    source_sampling_profile: Literal["stratified_v1", "deterministic_hash"] = "stratified_v1"
    max_source_ticks_per_seed: int = Field(default=4, ge=1)
    max_source_decisions_per_seed: int = Field(default=32, ge=1)
    source_ticks: tuple[int, ...] = (2, 5, 10, 15)
    context_families: tuple[str, ...] = (
        "balanced",
        "resource_scarce",
        "toxin_stress",
        "social_dense",
        "boundary_stress",
    )
    repeat_pilot: tuple[int, ...] = (2, 4, 8, 16)
    repeat_policy: int | Literal["pilot_derived"] = "pilot_derived"
    horizons: tuple[int, ...] = (1, 3, 5, 10)
    family_horizons: dict[str, tuple[int, ...]] = Field(default_factory=dict)
    minimum_independent_seeds: int = Field(default=3, ge=2)
    minimum_source_decisions: int = Field(default=100, ge=2)
    world_height: int | None = Field(default=None, ge=16)
    world_width: int | None = Field(default=None, ge=16)
    patch_size: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_seed_seal(self) -> CorpusConfig:
        """Reject overlapping seeds and invalid source/horizon registries."""
        roles = {
            "development": self.development_seeds,
            "validation": self.validation_seeds,
            "calibration": self.calibration_seeds,
            "phase5": self.reserved_phase5_seeds,
            "phase6": self.reserved_phase6_seeds,
        }
        seen: dict[int, str] = {}
        for role, values in roles.items():
            if len(set(values)) != len(values):
                raise ValueError(f"duplicate seed inside {role}")
            for seed in values:
                if seed in seen:
                    raise ValueError(f"seed {seed} overlaps {seen[seed]} and {role}")
                seen[seed] = role
        if tuple(sorted(set(self.repeat_pilot))) != self.repeat_pilot:
            raise ValueError("repeat_pilot must be strictly increasing and unique")
        if any(value < 1 for value in self.horizons):
            raise ValueError("horizons must be positive")
        if tuple(sorted(set(self.source_ticks))) != self.source_ticks:
            raise ValueError("source_ticks must be strictly increasing and unique")
        if any(value < 1 for value in self.source_ticks):
            raise ValueError("source_ticks must be positive")
        if len(set(self.context_families)) != len(self.context_families):
            raise ValueError("context_families must be unique")
        if len(self.source_ticks) > self.max_source_ticks_per_seed:
            raise ValueError("source_ticks exceeds max_source_ticks_per_seed")
        dimensions = (self.world_height, self.world_width, self.patch_size)
        if any(value is not None for value in dimensions) and not all(
            value is not None for value in dimensions
        ):
            raise ValueError(
                "world_height, world_width, and patch_size must be set together"
            )
        if (
            self.world_height is not None
            and self.world_width is not None
            and self.patch_size is not None
            and (
                self.world_height % self.patch_size
                or self.world_width % self.patch_size
            )
        ):
            raise ValueError("corpus world dimensions must be divisible by patch_size")
        return self


class TrainingConfig(OWLBaseModel):
    """Finite optimization, checkpoint, and ensemble-member policy."""

    epochs: int = Field(default=100, ge=1)
    learning_rate: float = Field(default=3e-4, gt=0.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    gradient_clip: float = Field(default=5.0, gt=0.0)
    early_stopping_patience: int = Field(default=12, ge=1)
    member_seeds: tuple[int, ...] = (4101, 4102, 4103, 4104, 4105)

    @model_validator(mode="after")
    def validate_member_seeds(self) -> TrainingConfig:
        """Require a nonempty unique ensemble seed registry."""
        if not self.member_seeds or len(set(self.member_seeds)) != len(self.member_seeds):
            raise ValueError("member seeds must be nonempty and unique")
        return self


class EvaluationConfig(OWLBaseModel):
    """Define evaluation metrics and subgroup export policy."""

    tie_tolerance: float = Field(default=1e-6, ge=0.0)
    subgroup_minimum: int = Field(default=50, ge=1)
    export_all_row_metrics: bool = True
    export_supported_row_metrics: bool = True


class NegativeControlConfig(OWLBaseModel):
    """Required leakage and label-perturbation control switches."""

    action_shuffle: bool = True
    target_shuffle: bool = True
    temporal_break: bool = True
    repeat_mismatch: bool = True
    mechanism_only: bool = True
    oracle_leakage_guard: bool = True
    random_seed: int = 4404


class CertificationConfig(OWLBaseModel):
    """Define a fail-closed certificate policy that keeps confirmatory evaluation locked."""

    classification: str = "PHASE4_DEVELOPMENT_CANDIDATE"
    phase5_unlock_requested: bool = False
    require_target_gpu: bool = True
    require_negative_control_collapse: bool = True
    require_all_ensemble_members: bool = True


class FeatureConfig(OWLBaseModel):
    """Perspective isolation and causal-history length policy."""

    primary_view: Literal["agent_primary"] = "agent_primary"
    oracle_diagnostic_enabled: bool = True
    mechanism_mediation_enabled: bool = True
    execution_postchoice_enabled: bool = True
    history_length: int = Field(default=8, ge=0, le=256)


class ScalarizationConfig(OWLBaseModel):
    """Frozen scalar profiles, urgency, quantile, and lower-tail policy."""

    profiles: tuple[str, ...] = (
        "agent_risk_neutral",
        "agent_risk_averse",
        "oracle_diagnostic",
        "collective_balanced",
    )
    urgency_beta: float = Field(default=2.0, ge=0.0, le=100.0)
    cvar_alpha: float = Field(default=0.1, gt=0.0, le=0.5)
    quantiles: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)

    @model_validator(mode="after")
    def validate_quantiles(self) -> ScalarizationConfig:
        """Require a sorted interior grid that identifies the CVaR tail."""
        if tuple(sorted(set(self.quantiles))) != self.quantiles:
            raise ValueError("quantiles must be strictly increasing")
        if any(value <= 0.0 or value >= 1.0 for value in self.quantiles):
            raise ValueError("quantiles must lie inside (0,1)")
        if not any(value <= self.cvar_alpha for value in self.quantiles):
            raise ValueError("quantile grid must contain a lower-tail CVaR point")
        return self


class SplitConfig(OWLBaseModel):
    """Grouped nested-cross-fit fold counts and world grouping fields."""

    outer_folds: int = Field(default=5, ge=2)
    inner_folds: int = Field(default=3, ge=2)
    group_fields: tuple[str, ...] = ("seed", "run_id")
    time_block_diagnostic: bool = True


class ModelConfig(OWLBaseModel):
    """Complete CADC-MORE 2 architecture and comparator switches."""

    ensemble_members: int = Field(default=5, ge=1, le=20)
    hidden_width: int = Field(default=192, ge=16, le=4096)
    depth: int = Field(default=3, ge=1, le=16)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    ranker_enabled: bool = True
    listwise_enabled: bool = True
    survival_enabled: bool = True
    epistemic_enabled: bool = True
    externality_enabled: bool = True
    xgboost_enabled: bool = True


class CalibrationConfig(OWLBaseModel):
    """Conformal and isotonic calibration support requirements."""

    interval_level: float = Field(default=0.9, gt=0.5, lt=1.0)
    mondrian_minimum: int = Field(default=100, ge=10)
    isotonic_minimum: int = Field(default=500, ge=20)


class SupportConfig(OWLBaseModel):
    """kNN/OOD/uncertainty thresholds for mandatory abstention."""

    abstain: bool = True
    minimum_seeds: int = Field(default=3, ge=2)
    minimum_decisions: int = Field(default=100, ge=1)
    minimum_repeats: int = Field(default=4, ge=1)
    maximum_ensemble_disagreement: float = Field(default=1.0, gt=0.0)
    maximum_conformal_width: float = Field(default=10.0, gt=0.0)
    knn_k: int = Field(default=32, ge=1)


class RuntimeConfig(OWLBaseModel):
    """CPU or H100/H200/B200 backend, precision, and memory policy."""

    target: HardwareTarget = HardwareTarget.CPU
    backend: Literal["numpy", "cupy"] = "numpy"
    precision: Literal["fp32", "bf16", "fp8"] = "fp32"
    batch_size: int = Field(default=256, ge=1)
    gradient_accumulation: int = Field(default=1, ge=1)
    compile: bool = False
    cuda_graphs: bool = False
    deterministic: bool = True
    max_device_bytes: int = Field(default=48 * 1024**3, ge=1)
    workers: int = Field(default=0, ge=0)
    prefetch: int = Field(default=2, ge=1, le=64)
    corpus_workers: int = Field(default=1, ge=1, le=32)
    corpus_transfer_mode: Literal["immediate_reference", "deferred_bounded"] = (
        "immediate_reference"
    )
    training_horizon_batching: Literal["sequential", "flattened"] = "sequential"
    training_checkpoint_interval: int = Field(default=1, ge=1, le=1000)
    runtime_calibration_units: int = Field(default=6, ge=2, le=64)
    require_runtime_decision: bool = False
    corpus_target_seconds: int = Field(default=0, ge=0)
    total_target_seconds: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_hardware_policy(self) -> RuntimeConfig:
        """Reject unsupported backend, FP8, and CUDA-graph combinations."""
        if self.target is HardwareTarget.CPU and self.backend != "numpy":
            raise ValueError("CPU target requires numpy backend")
        if self.target is not HardwareTarget.B200 and self.precision == "fp8":
            raise ValueError("FP8 is restricted to separately gated B200 profiles")
        if self.cuda_graphs and not self.compile:
            raise ValueError("CUDA Graphs require a compiled fixed-shape profile")
        if self.target is HardwareTarget.CPU and self.corpus_workers != 1:
            raise ValueError("CPU reference profiles require corpus_workers=1")
        if self.target is HardwareTarget.CPU and self.corpus_transfer_mode != "immediate_reference":
            raise ValueError("CPU reference profiles require immediate branch transfers")
        if self.total_target_seconds and self.corpus_target_seconds > self.total_target_seconds:
            raise ValueError("corpus time target cannot exceed total time target")
        return self


class ArtifactConfig(OWLBaseModel):
    """Output-root, overwrite, and compact-export bounds."""

    output_root: str = "phase4_output"
    overwrite: bool = False
    local_export_max_bytes: int = Field(default=2 * 1024**3, ge=1)


class CADCPhase4Config(OWLBaseModel):
    """Top-level source, science, model, runtime, and lock configuration."""

    phase3_input: Phase3InputConfig = Field(default_factory=Phase3InputConfig)
    corpus: CorpusConfig = Field(default_factory=CorpusConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    scalarization: ScalarizationConfig = Field(default_factory=ScalarizationConfig)
    splits: SplitConfig = Field(default_factory=SplitConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    support: SupportConfig = Field(default_factory=SupportConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    negative_controls: NegativeControlConfig = Field(default_factory=NegativeControlConfig)
    certification: CertificationConfig = Field(default_factory=CertificationConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    artifacts: ArtifactConfig = Field(default_factory=ArtifactConfig)
    master_seed: int = 20260719

    @model_validator(mode="after")
    def preserve_phase5_lock(self) -> CADCPhase4Config:
        """Keep confirmatory evaluation locked until all modeling prerequisites are complete."""
        if self.certification.phase5_unlock_requested:
            raise ValueError("Phase 4 development configuration cannot unlock Phase 5")
        if self.features.oracle_diagnostic_enabled and not self.models.xgboost_enabled:
            raise ValueError("oracle diagnostic currently requires the XGBoost rank baseline")
        if self.models.ensemble_members != len(self.training.member_seeds):
            raise ValueError(
                "models.ensemble_members must equal the number of training.member_seeds"
            )
        if len(self.corpus.development_seeds) < self.splits.outer_folds:
            raise ValueError(
                "development seeds must cover every grouped outer fold"
            )
        if not self.corpus.validation_seeds or not self.corpus.calibration_seeds:
            raise ValueError("validation and calibration seed roles must be nonempty")
        if len(self.corpus.development_seeds) < self.corpus.minimum_independent_seeds:
            raise ValueError(
                "development seeds are below corpus.minimum_independent_seeds"
            )
        maximum_sources = (
            len(
                (
                    *self.corpus.development_seeds,
                    *self.corpus.validation_seeds,
                    *self.corpus.calibration_seeds,
                )
            )
            * len(self.corpus.source_ticks)
            * self.corpus.max_source_decisions_per_seed
        )
        if maximum_sources < self.corpus.minimum_source_decisions:
            raise ValueError("planned corpus cannot reach minimum_source_decisions")
        return self

    def canonical_digest(self) -> str:
        """Return the full hardware/path-sensitive configuration identity."""
        encoded = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def corpus_digest(self) -> str:
        """Hardware/path-independent identity of the scientific corpus contract."""

        payload = {
            "phase3_source_sha256": self.phase3_input.required_source_sha256,
            "factual_v2_digest": self.phase3_input.factual_v2_digest,
            "counterfactual_schema_digest": self.phase3_input.counterfactual_schema_digest,
            "rng_registry_digest": self.phase3_input.rng_registry_digest,
            "corpus": self.corpus.model_dump(mode="json"),
            "master_seed": self.master_seed,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def model_spec_digest(self) -> str:
        """Hardware-independent identity of features, targets, and model policy."""

        from owl.cadc.features import FeatureRegistry
        from owl.cadc.outcomes import OutcomeRegistry
        from owl.cadc.schema import PHASE4_SCHEMA_DIGEST

        payload = {
            "corpus_sha256": self.corpus_digest(),
            "phase4_schema_digest": PHASE4_SCHEMA_DIGEST,
            "feature_registry_digest": FeatureRegistry().digest,
            "outcome_registry_digest": OutcomeRegistry().digest,
            "features": self.features.model_dump(mode="json"),
            "scalarization": self.scalarization.model_dump(mode="json"),
            "splits": self.splits.model_dump(mode="json"),
            "models": self.models.model_dump(mode="json"),
            "training": self.training.model_dump(mode="json"),
            "calibration": self.calibration.model_dump(mode="json"),
            "support": self.support.model_dump(mode="json"),
            "evaluation": self.evaluation.model_dump(mode="json"),
            "negative_controls": self.negative_controls.model_dump(mode="json"),
            "master_seed": self.master_seed,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def load_phase4_config(path: str | Path) -> CADCPhase4Config:
    """Load one config with at most one deterministic inheritance layer."""
    source = Path(path)
    payload: Any = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Phase 4 configuration must contain a mapping")
    parent_name = payload.pop("extends", None)
    if parent_name is not None:
        if not isinstance(parent_name, str) or not parent_name:
            raise TypeError("Phase 4 configuration 'extends' must be a nonempty path")
        parent_path = Path(parent_name)
        if not parent_path.is_absolute():
            parent_path = source.parent / parent_path
        parent_payload: Any = yaml.safe_load(parent_path.read_text(encoding="utf-8"))
        if not isinstance(parent_payload, dict):
            raise TypeError("extended Phase 4 configuration must contain a mapping")
        if "extends" in parent_payload:
            raise ValueError("nested Phase 4 configuration inheritance is forbidden")
        payload = _deep_merge(parent_payload, payload)
    return CADCPhase4Config.model_validate(payload)


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Merge hardware-specific overrides without mutating either parsed YAML mapping."""

    result = dict(base)
    for key, value in overlay.items():
        prior = result.get(key)
        if isinstance(prior, dict) and isinstance(value, dict):
            result[key] = _deep_merge(prior, value)
        else:
            result[key] = value
    return result
