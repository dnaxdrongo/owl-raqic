"""Read-only interfaces for CADC-MORE 2 model building and evaluation."""

from owl.cadc.config import CADCPhase4Config, load_phase4_config
from owl.cadc.schema import (
    PHASE4_CERTIFICATE_VERSION,
    PHASE4_DATASET_SCHEMA_VERSION,
    PHASE4_FEATURE_SCHEMA_VERSION,
    PHASE4_MODEL_ARTIFACT_VERSION,
    PHASE4_OUTCOME_REGISTRY_VERSION,
    PHASE4_SCORE_SCHEMA_VERSION,
    ActionFamily,
    FeaturePerspective,
    SupportStatus,
)

__all__ = [
    "CADCPhase4Config",
    "PHASE4_CERTIFICATE_VERSION",
    "PHASE4_DATASET_SCHEMA_VERSION",
    "PHASE4_FEATURE_SCHEMA_VERSION",
    "PHASE4_MODEL_ARTIFACT_VERSION",
    "PHASE4_OUTCOME_REGISTRY_VERSION",
    "PHASE4_SCORE_SCHEMA_VERSION",
    "ActionFamily",
    "FeaturePerspective",
    "SupportStatus",
    "load_phase4_config",
]
