from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from owl.cadc.config import CADCPhase4Config, load_phase4_config
from owl.cadc.schema import ACTION_FAMILY_REGISTRY, schema_manifest, stable_id
from owl.core.actions import Action

EXPECTED_ACTIONS = (
    "REST",
    "SENSE",
    "MOVE_N",
    "MOVE_S",
    "MOVE_E",
    "MOVE_W",
    "MOVE_NE",
    "MOVE_NW",
    "MOVE_SE",
    "MOVE_SW",
    "FEED",
    "COMMUNICATE",
    "INHIBIT",
    "INTEGRATE",
    "REPAIR",
    "REPRODUCE",
    "INGEST",
    "EXPEL",
    "SPLIT",
    "MERGE",
    "FLEE",
    "PURSUE",
)


def test_immutable_action_axis_and_family_registry() -> None:
    assert tuple(action.name for action in Action) == EXPECTED_ACTIONS
    assert tuple(value.action_index for value in ACTION_FAMILY_REGISTRY) == tuple(range(22))
    assert schema_manifest()["action_axis"] == [
        {"index": index, "name": name} for index, name in enumerate(EXPECTED_ACTIONS)
    ]


def test_stable_id_is_length_delimited_and_deterministic() -> None:
    assert stable_id("test", "ab", "c") == stable_id("test", "ab", "c")
    assert stable_id("test", "ab", "c") != stable_id("test", "a", "bc")
    assert stable_id("test", 1) != stable_id("test", "1")


def test_seed_overlap_and_phase5_unlock_fail_closed() -> None:
    with pytest.raises(ValueError, match="overlaps"):
        CADCPhase4Config.model_validate(
            {
                "corpus": {
                    "development_seeds": [1],
                    "validation_seeds": [1],
                }
            }
        )
    with pytest.raises(ValueError, match="cannot unlock Phase 5"):
        CADCPhase4Config.model_validate(
            {"certification": {"phase5_unlock_requested": True}}
        )


def test_hardware_overlays_share_science_but_not_runtime_identity() -> None:
    root = Path(__file__).resolve().parents[1] / "configs"
    configs = [
        load_phase4_config(root / "cadc_phase4_development.yaml"),
        load_phase4_config(root / "cadc_phase4_h200_development.yaml"),
        load_phase4_config(root / "cadc_phase4_b200_development.yaml"),
    ]
    assert len({value.corpus_digest() for value in configs}) == 1
    assert len({value.model_spec_digest() for value in configs}) == 1
    assert len({value.canonical_digest() for value in configs}) == 3
    assert [value.runtime.target.value for value in configs] == ["h100", "h200", "b200"]
    assert all(value.runtime.precision == "bf16" for value in configs)


def test_nested_config_inheritance_is_rejected(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text("extends: earlier.yaml\n", encoding="utf-8")
    child = tmp_path / "child.yaml"
    child.write_text("extends: base.yaml\n", encoding="utf-8")
    with pytest.raises(ValueError, match="nested"):
        load_phase4_config(child)


def test_source_tick_bound_is_enforced() -> None:
    payload = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs/cadc_phase4_cpu_smoke.yaml").read_text()
    )
    payload["corpus"]["max_source_ticks_per_seed"] = 1
    payload["corpus"]["source_ticks"] = [1, 2]
    with pytest.raises(ValueError, match="source_ticks exceeds"):
        CADCPhase4Config.model_validate(payload)
