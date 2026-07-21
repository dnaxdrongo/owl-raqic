from __future__ import annotations

from types import SimpleNamespace

import pytest

from owl.gpu.distributed.launch import _certify_collective_ledgers
from owl.gpu.graph_certification import certify_graph_status
from owl.runtime.production_guard import evaluate_production_readiness
from owl_raqic.qiskit_backend.parameterized_templates import (
    build_raw_feature_template,
    supports_runtime_parameter_binding,
)
from owl_raqic.qiskit_backend.per_ow_executor import _metadata_reports_gpu


def test_runtime_parameter_binding_is_static_only():
    assert supports_runtime_parameter_binding("static")
    for family in ("deferred", "dynamic_recursive", "walk", "density_noise"):
        assert not supports_runtime_parameter_binding(family)
        with pytest.raises(ValueError, match="not certified"):
            build_raw_feature_template(8, family=family)


def test_aer_gpu_metadata_requires_positive_evidence():
    assert _metadata_reports_gpu({"device": "GPU"})
    assert _metadata_reports_gpu({"nested": {"gpu_parallel_shots_": 2}})
    assert not _metadata_reports_gpu({"device": "CPU"})
    assert not _metadata_reports_gpu({})


def test_full_graph_certificate_requires_pointer_contract():
    status = {
        "coverage": {
            "required_segments": ["tick"],
            "captured_segments": ["tick"],
            "replay_counts": {"tick": 2},
            "uncovered_reasons": {},
        },
        "invalidation_count": 0,
    }
    assert not certify_graph_status(status, requirement="full_tick").passed
    status["capture_contract"] = {
        "pointer_snapshot_enabled": True,
        "captured_segments_pointer_stable": True,
    }
    assert certify_graph_status(status, requirement="full_tick").passed


def _ledger_report(rank: int, ledger: list[dict], boundary_checks: int = 2):
    return {
        "rank": rank,
        "collective_ledger": ledger,
        "halo_stats": {
            "boundary_checks": boundary_checks,
            "boundary_elements": 20 if boundary_checks else 0,
        },
    }


def test_distributed_ledger_certificate_pairs_p2p_and_collectives():
    rank0 = _ledger_report(
        0,
        [
            {"operation": "send", "count": 4, "dtype": "float32", "peer_or_root": 1, "tick": 1},
            {"operation": "recv", "count": 4, "dtype": "float32", "peer_or_root": 1, "tick": 1},
            {
                "operation": "all_reduce:sum",
                "count": 2,
                "dtype": "float64",
                "peer_or_root": -1,
                "tick": 1,
            },
        ],
    )
    rank1 = _ledger_report(
        1,
        [
            {"operation": "recv", "count": 4, "dtype": "float32", "peer_or_root": 0, "tick": 1},
            {"operation": "send", "count": 4, "dtype": "float32", "peer_or_root": 0, "tick": 1},
            {
                "operation": "all_reduce:sum",
                "count": 2,
                "dtype": "float64",
                "peer_or_root": -1,
                "tick": 1,
            },
        ],
    )
    certificate = _certify_collective_ledgers([rank0, rank1])
    assert certificate["passed"]
    assert certificate["point_to_point_pairs_match"]
    assert certificate["collective_sequences_match"]


def test_distributed_ledger_certificate_rejects_unmatched_send():
    certificate = _certify_collective_ledgers(
        [
            _ledger_report(
                0,
                [
                    {
                        "operation": "send",
                        "count": 4,
                        "dtype": "float32",
                        "peer_or_root": 1,
                        "tick": 1,
                    }
                ],
            ),
            _ledger_report(1, []),
        ]
    )
    assert not certificate["passed"]
    assert any("unmatched NCCL" in item for item in certificate["failures"])


def test_production_guard_requires_strict_qiskit_gpu_proof_and_distributed_certificate():
    plan = SimpleNamespace(
        graph_requirement="allow_partial",
        qiskit_policy=SimpleNamespace(per_ow=True, strict_gpu=True),
        multi_gpu=True,
    )
    base = {
        "fallback_count": 0,
        "per_ow_qiskit": {
            "processed_count": 3,
            "expected_count": 3,
            "metadata": {"gpu_execution_verified": False},
        },
        "distributed_certification": {"passed": False},
    }
    readiness = evaluate_production_readiness(
        plan=plan,
        execution_metadata=base,
        all_configs_valid=True,
        config_usage_clean=True,
        memory_preflight_passed=True,
        certification_compatible=True,
    )
    assert not readiness.passed
    base["per_ow_qiskit"]["metadata"]["gpu_execution_verified"] = True
    base["distributed_certification"]["passed"] = True
    readiness = evaluate_production_readiness(
        plan=plan,
        execution_metadata=base,
        all_configs_valid=True,
        config_usage_clean=True,
        memory_preflight_passed=True,
        certification_compatible=True,
    )
    assert readiness.passed
