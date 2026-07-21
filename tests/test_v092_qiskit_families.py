# ruff: noqa: E402 -- approved source-tree bootstrap or optional import gate
from __future__ import annotations

import numpy as np
import pytest

qiskit = pytest.importorskip("qiskit")
qiskit_aer = pytest.importorskip("qiskit_aer")

from qiskit import transpile
from qiskit_aer import AerSimulator

from owl_raqic.qiskit_backend.aer_runtime import run_aer_job
from owl_raqic.qiskit_backend.circuit_families import (
    CIRCUIT_FAMILIES,
    build_circuit_family,
    circuit_family_structure,
)
from owl_raqic.qiskit_backend.parameterized_templates import (
    statevector_action_probabilities,
    supports_runtime_parameter_binding,
)
from owl_raqic.qiskit_backend.per_ow_executor import parse_aer_gpu_evidence

P = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
PHASE = np.array([0.0, 0.2, -0.3, 0.5], dtype=np.float64)


def _statevector_probs(family: str, **kwargs):
    built = build_circuit_family(family, P, PHASE, measure=False, **kwargs)
    circuit = built.circuit.remove_final_measurements(inplace=False).copy()
    circuit.save_statevector()
    simulator = AerSimulator(method="statevector", device="CPU")
    circuit = transpile(circuit, backend=simulator, optimization_level=0)
    result = run_aer_job(simulator, circuit)
    state = np.asarray(result.get_statevector(0), dtype=np.complex128)
    return built, statevector_action_probabilities(
        state,
        action_qubits=tuple(built.layout.action_qubits),
        action_count=P.size,
    )


def test_static_and_deferred_have_declared_action_marginals():
    static, static_probs = _statevector_probs("static")
    deferred, deferred_probs = _statevector_probs("deferred", feedback_phases=np.array([0.1, 0.2]))
    assert np.allclose(static_probs, P, atol=1e-10)
    assert np.allclose(deferred_probs, CIRCUIT_FAMILIES["deferred"].oracle(P), atol=1e-10)
    structure = circuit_family_structure(deferred.circuit)
    assert "cx" in structure["operation_names"]
    assert "crz" in structure["operation_names"]
    assert deferred.layout.record_qubits


def test_walk_is_multistep_coin_and_conditional_shift_with_oracle():
    built, actual = _statevector_probs("walk", steps=3, n_positions=4)
    expected = CIRCUIT_FAMILIES["walk"].oracle(P, steps=3, n_positions=4)
    assert np.allclose(actual, expected, atol=1e-10)
    structure = circuit_family_structure(built.circuit)
    assert structure["semantic_operation_names"].count("legal_coin_operator") == 3
    assert structure["semantic_operation_names"].count("conditional_ring_shift") == 3
    assert built.layout.position_qubits


def test_dynamic_uses_real_measurement_and_control_flow():
    built = build_circuit_family("dynamic_recursive", P, PHASE, rounds=2)
    structure = circuit_family_structure(built.circuit)
    assert structure["has_mid_circuit_measurement"]
    assert structure["has_control_flow"]
    assert "if_else" in structure["control_flow_operations"]
    # The feedback is diagonal after re-preparation, so it preserves the legal
    # action distribution. Run a shot-based CPU Aer smoke of the real circuit.
    simulator = AerSimulator(
        method="statevector",
        device="CPU",
        max_parallel_threads=1,
        max_parallel_experiments=1,
    )
    compiled = transpile(built.circuit, backend=simulator, optimization_level=0)
    result = run_aer_job(simulator, compiled, shots=2048, seed_simulator=17)
    counts = result.get_counts(0)
    decoded = CIRCUIT_FAMILIES["dynamic_recursive"].decode_counts(counts, built, len(P))
    assert np.max(np.abs(decoded - P)) < 0.06


def test_density_noise_attaches_real_noise_declaration_and_shot_oracle():
    built = build_circuit_family(
        "density_noise", P, PHASE, measure=True, depolarizing_probability=0.001
    )
    assert built.metadata["noise_model"] == "depolarizing"
    expected = CIRCUIT_FAMILIES["density_noise"].oracle(P, depolarizing_probability=0.001)
    assert np.isclose(expected.sum(), 1.0)
    assert np.all(expected >= 0.0)


def test_runtime_parameter_binding_is_static_only():
    assert supports_runtime_parameter_binding("static")
    for family in ("deferred", "dynamic_recursive", "walk", "density_noise"):
        assert not supports_runtime_parameter_binding(family)


def test_gpu_evidence_parser_requires_positive_metadata():
    assert not parse_aer_gpu_evidence({"device": "CPU"})["verified"]
    assert parse_aer_gpu_evidence({"device": "GPU"})["verified"]
    assert parse_aer_gpu_evidence({"batched_shots_optimization_parallel_gpus": 2})["verified"]
    assert parse_aer_gpu_evidence({"chunk_parallel_gpus": 1})["verified"]


def test_walk_preserves_sparse_authority_subspace():
    from owl_raqic.qiskit_backend.per_ow_executor import PerOWQiskitExecutor
    from owl_raqic.qiskit_backend.qiskit_policy import (
        QiskitDecisionMode,
        QiskitExecutionPolicy,
    )

    p = np.asarray([[0.7, 0.0, 0.3, 0.0, 0.0]], dtype=np.float64)
    phase = np.zeros_like(p)
    authority = np.asarray([[True, False, True, False, False]], dtype=bool)
    policy = QiskitExecutionPolicy(
        mode=QiskitDecisionMode.EVERY_OW_CIRCUIT_FAMILY,
        circuit_families=("walk",),
        authoritative_family="walk",
        method="statevector",
        device="CPU",
        strict_gpu=False,
        shots=2048,
        chunk_size=1,
        confirm_expensive=True,
    )
    result = PerOWQiskitExecutor(policy, seed=7).execute(
        p, phase, authority, np.asarray([9], dtype=np.int64), tick=1
    )
    row = result.authoritative.probabilities[0]
    assert np.allclose(row[~authority[0]], 0.0, atol=1e-12)
    assert np.isclose(row.sum(), 1.0)
    audit = result.authoritative.metadata["authority_audit"]
    assert audit["all_passed"]
    assert audit["max_illegal_probability"] <= 1e-12
    assert audit["rows"][0]["legal_basis"] == (0, 2)
    assert audit["projection_policy"] == "legal_subspace_required"
