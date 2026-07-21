from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from owl.core.actions import Action
from owl.core.config import CADCRecordingConfig, SimulationConfig
from owl.record.cadc_schema import (
    CADC_ACTION_COUNT,
    CADC_SCHEMA_DIGEST,
    CADC_SCHEMA_VERSION,
    ReasonCode,
    action_names,
    schema_manifest,
)


def test_cadc_defaults_are_disabled_and_round_trip_in_normalized_config() -> None:
    cfg = SimulationConfig()
    assert cfg.recording.cadc.enabled is False
    assert cfg.recording.cadc.host_queue_depth == 1

    restored = SimulationConfig.model_validate_json(cfg.model_dump_json())
    assert restored == cfg
    assert restored.recording.cadc.model_dump() == cfg.recording.cadc.model_dump()


def test_cadc_config_rejects_claims_without_required_evidence() -> None:
    with pytest.raises(ValidationError, match="information capture requires"):
        CADCRecordingConfig(capture_agent_context=False, capture_information=True)
    with pytest.raises(ValidationError, match="dense context requires"):
        CADCRecordingConfig(capture_oracle_context=False, include_dense_context=True)
    with pytest.raises(ValidationError, match="exact profile requires"):
        CADCRecordingConfig(profile="exact")
    with pytest.raises(ValidationError, match="exact profile requires"):
        CADCRecordingConfig(
            profile="exact", include_dense_context=True, strict_overflow=False
        )
    with pytest.raises(ValidationError, match="must not exceed"):
        CADCRecordingConfig(max_pending_bytes=1024 * 1024, max_batch_bytes=2 * 1024 * 1024)


def test_cadc_schema_preserves_action_order_and_has_stable_digest() -> None:
    assert CADC_ACTION_COUNT == 22
    assert action_names() == tuple(action.name for action in Action)
    manifest = schema_manifest()
    assert manifest["schema_version"] == CADC_SCHEMA_VERSION
    assert manifest["schema_digest"] == CADC_SCHEMA_DIGEST
    assert len(manifest["action_names"]) == 22
    assert len(json.dumps(manifest, sort_keys=True)) > 100


def test_reason_registry_is_numeric_unique_and_reserves_zero_for_none() -> None:
    codes = [int(item) for item in ReasonCode]
    assert len(codes) == len(set(codes))
    assert int(ReasonCode.NONE) == 0
    assert int(ReasonCode.NO_EXECUTION_CONTRACT) == 11
