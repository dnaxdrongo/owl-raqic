from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from owl.gpu.distributed.launch import _certify_collective_ledgers
from owl.gpu.graph_safety import CaptureAllocationGuard, MemoryPoolSnapshot
from owl.gpu.memory_model import AllocationSpec, MemoryPlan
from owl.record.gpu_async_writer import AsyncGPUWriter
from owl.runtime.certificates import EnvironmentIdentity, compare_identities
from owl.viz.event_bus import VisualEvent, VisualEventBuffer, VisualEventType
from owl_raqic.qiskit_backend.aer_evidence import parse_aer_gpu_evidence
from owl_raqic.qiskit_backend.gpu_execution import run_statevector_probabilities_gpu


def test_aer_gpu_evidence_requires_positive_result_metadata():
    assert not parse_aer_gpu_evidence({"device": "CPU"})["verified"]
    assert parse_aer_gpu_evidence({"device": "GPU"})["verified"]
    assert parse_aer_gpu_evidence({"chunk_parallel_gpus": 2})["verified"]


def test_generic_statevector_extraction_requires_action_layout_for_ancillas():
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(2)
    circuit.h(0)
    with pytest.raises(ValueError, match="action_qubits"):
        run_statevector_probabilities_gpu(circuit, n_actions=2, device="CPU")
    result = run_statevector_probabilities_gpu(
        circuit, n_actions=2, device="CPU", action_qubits=(0,)
    )
    assert np.allclose(result.probabilities, [0.5, 0.5])


def test_capture_allocation_guard_fails_on_pool_growth():
    guard = CaptureAllocationGuard(xp=object())
    passed, reason = guard.compare(
        "decision", MemoryPoolSnapshot(10, 20), MemoryPoolSnapshot(11, 30)
    )
    assert not passed
    assert "unplanned capture allocation" in reason


def test_distributed_certificate_rejects_missing_rank_and_boundary_mismatch():
    reports = [
        {
            "rank": 0,
            "halo_stats": {
                "boundary_checks": 1,
                "boundary_elements": 1,
                "boundary_mismatch_count": 1,
            },
            "collective_ledger": [],
        }
    ]
    cert = _certify_collective_ledgers(reports, expected_rank_count=2)
    assert not cert["passed"]
    assert any("rank report count" in item for item in cert["failures"])
    assert not cert["boundary_consistency_passed"]


def test_visual_and_writer_evidence_fail_closed(tmp_path: Path):
    events = VisualEventBuffer(capacity=1)
    events.add(VisualEvent(1, VisualEventType.AUDIT_FAILURE, 0, 0))
    with pytest.raises(OverflowError):
        events.add(VisualEvent(1, VisualEventType.DEATH, 1, 1))
    assert events.critical_drop_count == 1

    writer = AsyncGPUWriter(tmp_path / "records.jsonl", max_queue=1, overflow_policy="raise")
    writer.write({"tick": 1})
    with pytest.raises(RuntimeError, match="queue is full"):
        writer.write({"tick": 2})
    assert writer.overflow_count == 1
    writer.close()


def test_memory_and_environment_identity_evidence_are_exact():
    plan = MemoryPlan(
        allocations=[AllocationSpec("state", 256, "gpu", "run")],
        steady_state_bytes=256,
        peak_bytes=512,
        allowed_bytes=128,
    )
    assert not plan.evaluate()

    identity = EnvironmentIdentity(
        source_sha256="source",
        config_sha256="config",
        plan_sha256="plan",
        scientific_contract_version="contract",
        scientific_contract_sha256="contract-hash",
        python_version="3.13",
        package_versions={},
        platform="test",
        driver_version="driver",
        cuda_runtime_version="cuda",
        gpu_devices=(),
        nccl_version=None,
    )
    assert compare_identities(identity, replace(identity, source_sha256="changed"))
