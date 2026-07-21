"""Define versioned CADC-MORE 2 schemas, registries, and stable identifiers."""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from owl.core.actions import Action

PHASE4_DATASET_SCHEMA_VERSION = "owl.cadc.phase4.dataset.v1"
PHASE4_FEATURE_SCHEMA_VERSION = "owl.cadc.phase4.features.v1"
PHASE4_OUTCOME_REGISTRY_VERSION = "owl.cadc.phase4.outcomes.v1"
PHASE4_MODEL_ARTIFACT_VERSION = "owl.cadc.phase4.model-artifact.v1"
PHASE4_SCORE_SCHEMA_VERSION = "owl.cadc.phase4.scores.v1"
PHASE4_CERTIFICATE_VERSION = "owl.cadc.phase4-development-certificate.v1"
PHASE4_ID_VERSION = "owl.cadc.phase4-id-sha256.v1"

EXPECTED_PHASE3_SOURCE_SHA256 = (
    "d17ef58692c7663eb0cc87ab4cdf7e74ca9b529091fcab4f15b6fe28e2a607a3"
)
EXPECTED_PHASE3_CLASSIFICATION = "H100_PHASE3_TARGET_CERTIFIED"
EXPECTED_PHASE25_SOURCE_SHA256 = (
    "c48b19939583e8c63033f96a7bd147f6f11e7a67cb1a2e957d91922bdc55a15a"
)
EXPECTED_PHASE25_CERTIFICATE_SHA256 = (
    "59ac1ea607863a8b9457273006dcda53e9141eea1bead1eedc1bb1f1d21a5aba"
)
EXPECTED_FACTUAL_V2_DIGEST = (
    "14beff051fc94f6069444355d109ef77a82f092d00e98f25fe3fcefa782edec5"
)
EXPECTED_COUNTERFACTUAL_DIGEST = (
    "c00db34ad9cbaaa0fa589261a9b6718860bbe1a9f01fe7549f0ba9f90c2da4f1"
)
EXPECTED_RNG_REGISTRY_DIGEST = (
    "4ce20f4ce0500206d8e93c73db0006f58defa2c2ac3543d2aa8246585509a00a"
)


class FeaturePerspective(StrEnum):
    """Mutually isolated feature evidence perspectives."""

    AGENT_PRIMARY = "agent_primary"
    ORACLE_DIAGNOSTIC = "oracle_diagnostic"
    MECHANISM_MEDIATION = "mechanism_mediation"
    EXECUTION_POSTCHOICE = "execution_postchoice"


class FeatureStage(StrEnum):
    """Temporal availability stage for a registered feature."""

    PRE_CHOICE = "pre_choice"
    POST_CHOICE = "post_choice"
    HISTORY = "history"
    DIAGNOSTIC = "diagnostic"


class OutcomeFamily(StrEnum):
    """Scientific family of a raw vector outcome."""

    HOMEOSTASIS = "homeostasis"
    SURVIVAL = "survival"
    ACTION_ENDPOINT = "action_endpoint"
    INFORMATION = "information"
    EXTERNALITY = "externality"
    LINEAGE = "lineage"


class ActionFamily(StrEnum):
    """Stable family routing for all immutable actions."""

    LOCOMOTION = "locomotion_navigation"
    FEEDING = "feeding_acquisition"
    DEFENSE = "defense_predation"
    SENSING = "sensing_communication"
    MAINTENANCE = "maintenance_integration"
    REPRODUCTION = "reproduction_topology"


class SupportStatus(StrEnum):
    """Typed support result used by mandatory abstention."""

    SUPPORTED = "supported"
    LIMITED = "limited_support"
    INSUFFICIENT = "insufficient_counterfactual_support"
    OOD = "out_of_distribution"


class CalibrationStatus(StrEnum):
    """Typed probability/interval calibration state."""

    UNCALIBRATED = "uncalibrated"
    CALIBRATED = "calibrated"
    INSUFFICIENT = "insufficient_calibration_support"


class ModelRole(StrEnum):
    """Registered CADC-MORE 2 component roles."""

    VIABILITY_BASELINE = "viability_baseline"
    STRUCTURAL_TRANSITION = "structural_transition"
    FAMILY_EXPERT = "family_expert"
    RANKER = "ranker"
    SURVIVAL_RISK = "survival_risk"
    EPISTEMIC_VALUE = "epistemic_value"
    EXTERNALITY = "externality"
    SUPPORT_CALIBRATOR = "support_calibrator"


class SplitRole(StrEnum):
    """Define modeling, calibration, and sealed confirmatory roles."""

    TRAIN = "train"
    VALIDATION = "validation"
    CALIBRATION = "calibration"
    TEST = "test"
    PHASE5_SEALED = "phase5_sealed"
    PHASE6_SEALED = "phase6_sealed"


class ArtifactStatus(StrEnum):
    """Lifecycle status for independently persisted artifacts."""

    STARTED = "started"
    PASSED = "passed"
    FAILED = "failed"
    INSUFFICIENT_DATA = "insufficient_data"
    UNSUPPORTED_EVIDENCE = "unsupported_evidence"


class AbstentionReason(StrEnum):
    """Typed reason why a candidate score must not be treated as supported."""

    NONE = "none"
    LOW_SEED_COVERAGE = "low_seed_coverage"
    LOW_ACTION_SUPPORT = "low_action_support"
    LOW_REPEAT_SUPPORT = "low_repeat_support"
    PROPENSITY_OVERLAP = "propensity_overlap"
    FEATURE_OOD = "feature_ood"
    HIGH_DISAGREEMENT = "high_ensemble_disagreement"
    WIDE_INTERVAL = "wide_conformal_interval"
    MISSING_ENDPOINT = "missing_endpoint"


class HardwareTarget(StrEnum):
    """Supported CPU and target-GPU certification profiles."""

    CPU = "cpu"
    H100 = "h100"
    H200 = "h200"
    B200 = "b200"


@dataclass(frozen=True)
class ActionFamilyDefinition:
    """Immutable action index/name, family route, and endpoint tags."""

    action_index: int
    action_name: str
    primary_family: ActionFamily
    endpoint_tags: tuple[str, ...]


_LOCOMOTION = tuple(range(int(Action.MOVE_N), int(Action.MOVE_SW) + 1))
ACTION_FAMILY_REGISTRY: tuple[ActionFamilyDefinition, ...] = tuple(
    ActionFamilyDefinition(
        int(action),
        action.name,
        (
            ActionFamily.LOCOMOTION
            if int(action) in _LOCOMOTION
            else ActionFamily.FEEDING
            if action in {Action.FEED, Action.INGEST}
            else ActionFamily.SENSING
            if action in {Action.SENSE, Action.COMMUNICATE}
            else ActionFamily.REPRODUCTION
            if action in {Action.REPRODUCE, Action.SPLIT, Action.MERGE}
            else ActionFamily.DEFENSE
            if action in {Action.INHIBIT, Action.EXPEL, Action.FLEE, Action.PURSUE}
            else ActionFamily.MAINTENANCE
        ),
        (
            ("semantic_target", "compiled_movement", "survival_risk")
            if action in {Action.FLEE, Action.PURSUE}
            else ("topology", "metabolic")
            if action is Action.EXPEL
            else ("lineage", "topology")
            if action in {Action.REPRODUCE, Action.SPLIT, Action.MERGE}
            else ("information", "control_value")
            if action in {Action.SENSE, Action.COMMUNICATE}
            else ("movement",)
            if int(action) in _LOCOMOTION
            else ("homeostasis",)
        ),
    )
    for action in Action
)


def _canonical(value: Any) -> bytes:
    if value is None:
        return b"n"
    if isinstance(value, bool):
        return b"b1" if value else b"b0"
    if isinstance(value, int):
        return b"i" + str(value).encode("ascii")
    if isinstance(value, float):
        return b"f" + struct.pack(">d", value)
    if isinstance(value, bytes):
        return b"y" + value
    if isinstance(value, str):
        return b"s" + value.encode("utf-8")
    if isinstance(value, Mapping):
        return b"m" + _length_prefixed(
            tuple(
                _length_prefixed((_canonical(str(key)), _canonical(value[key])))
                for key in sorted(value, key=str)
            )
        )
    if isinstance(value, Sequence):
        return b"q" + _length_prefixed(tuple(_canonical(item) for item in value))
    raise TypeError(f"unsupported Phase 4 ID component: {type(value).__name__}")


def _length_prefixed(parts: Sequence[bytes]) -> bytes:
    return b"".join(struct.pack(">Q", len(part)) + part for part in parts)


def stable_id(kind: str, *components: Any) -> str:
    """Return a stable length-prefixed SHA-256 identifier."""
    payload = _length_prefixed(
        (_canonical(PHASE4_ID_VERSION), _canonical(kind), *map(_canonical, components))
    )
    return hashlib.sha256(payload).hexdigest()


def action_family(action: int | Action) -> ActionFamilyDefinition:
    """Resolve one immutable action-family definition by exact index."""
    index = int(action)
    if index < 0 or index >= len(ACTION_FAMILY_REGISTRY):
        raise ValueError(f"action outside immutable axis: {index}")
    return ACTION_FAMILY_REGISTRY[index]


def schema_manifest() -> dict[str, Any]:
    """Return versioned schema and action-family identities."""
    payload: dict[str, Any] = {
        "dataset_schema_version": PHASE4_DATASET_SCHEMA_VERSION,
        "feature_schema_version": PHASE4_FEATURE_SCHEMA_VERSION,
        "outcome_registry_version": PHASE4_OUTCOME_REGISTRY_VERSION,
        "model_artifact_version": PHASE4_MODEL_ARTIFACT_VERSION,
        "score_schema_version": PHASE4_SCORE_SCHEMA_VERSION,
        "certificate_version": PHASE4_CERTIFICATE_VERSION,
        "id_version": PHASE4_ID_VERSION,
        "action_axis": [{"index": int(action), "name": action.name} for action in Action],
        "action_families": [asdict(item) for item in ACTION_FAMILY_REGISTRY],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    payload["schema_digest"] = hashlib.sha256(encoded).hexdigest()
    payload["family_registry_digest"] = stable_id(
        "action_family_registry", payload["action_families"]
    )
    return payload


PHASE4_SCHEMA_DIGEST = str(schema_manifest()["schema_digest"])
ACTION_FAMILY_REGISTRY_DIGEST = str(schema_manifest()["family_registry_digest"])
