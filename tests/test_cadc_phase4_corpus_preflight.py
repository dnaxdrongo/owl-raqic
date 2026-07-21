from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from owl.cadc.corpus import validate_corpus_qiskit_evidence
from owl.core.config import load_config


def _unit_module() -> Any:
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts/_run_cadc_phase4_corpus_unit.py"
    spec = importlib.util.spec_from_file_location("phase4_corpus_unit", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _off_evidence() -> dict[str, object]:
    return {
        "passed": True,
        "exercised": False,
        "mode": "off",
        "evidence_status": "not_exercised",
        "runtime_binding_required": False,
        "runtime_binding_used": False,
        "automatic_execution_fallback": False,
    }


def test_non_qiskit_corpus_preflight_does_not_invoke_qiskit(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs/cadc_phase3_phase25_h100_acceptance.yaml"
    cfg = load_config(config_path)
    calls = 0

    def forbidden_qiskit_validator(**_: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        raise AssertionError("Qiskit validator must not run for mode=off")

    module = _unit_module()
    receipt, qiskit, exercised = module._prepare_corpus_preflight(
        engine=root,
        config_path=config_path,
        output=tmp_path,
        cfg=cfg,
        backend="numpy",
        phase3_source="phase3-source",
        qiskit_validator=forbidden_qiskit_validator,
    )
    assert calls == 0
    assert exercised is False
    assert qiskit == _off_evidence()
    assert receipt["scientific_ticks_started"] == 0
    persisted = json.loads(
        (tmp_path / "preflight/preflight_receipt.json").read_text(encoding="utf-8")
    )
    assert persisted["qiskit_execution"] == _off_evidence()


@pytest.mark.parametrize(
    ("mutations", "expected"),
    [
        ({}, []),
        (
            {"qiskit_exercised": None},
            ["qiskit_applicability_missing:unit"],
        ),
        (
            {"qiskit_gpu_runtime_required": True},
            ["qiskit_runtime_required_when_not_exercised:unit"],
        ),
        (
            {"qiskit_execution": None},
            ["qiskit_evidence_missing:unit"],
        ),
    ],
)
def test_non_qiskit_inventory_contract(
    mutations: dict[str, object], expected: list[str]
) -> None:
    inventory: dict[str, object] = {
        "qiskit_exercised": False,
        "qiskit_gpu_runtime_required": False,
        "qiskit_execution": _off_evidence(),
    }
    inventory.update(mutations)
    failures = validate_corpus_qiskit_evidence(inventory, unit_id="unit")
    for failure in expected:
        assert failure in failures
    if not expected:
        assert failures == []


def test_exercised_target_gpu_qiskit_requires_runtime_binding() -> None:
    inventory = {
        "qiskit_exercised": True,
        "qiskit_gpu_runtime_required": True,
        "qiskit_execution": {
            "passed": True,
            "exercised": True,
            "mode": "validation_sample",
            "evidence_status": "executed",
            "runtime_binding_required": True,
            "runtime_binding_used": False,
            "automatic_execution_fallback": False,
        },
    }
    assert validate_corpus_qiskit_evidence(inventory, unit_id="unit") == [
        "qiskit_runtime_binding_missing:unit"
    ]


def test_effective_qiskit_mode_preserves_legacy_force_all_alias() -> None:
    module = _unit_module()
    cfg = SimpleNamespace(
        raqic=SimpleNamespace(
            qiskit_decision_mode="off",
            use_qiskit_for_all=True,
        )
    )
    assert module._effective_qiskit_mode(cfg) == "every_ow_static_exact"


def test_action_family_lexsort_matches_numpy_tuple_semantics() -> None:
    module = _unit_module()
    sequence = np.asarray([13, 10, 12, 11, 14], dtype=np.int64)
    actions = np.asarray([2, 1, 2, 1, 0], dtype=np.int16)
    expected = np.lexsort((sequence, actions))
    actual = module._action_family_lexsort(np, sequence, actions)
    np.testing.assert_array_equal(actual, expected)
