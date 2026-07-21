from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .backend_profiles import require_qiskit
from .register_builder import bits_needed
from .result_decode import RegisterLayout


@dataclass(frozen=True)
class RAQICCircuitBuildResult:
    circuit: object
    parameter_map: dict[str, Any]
    registers: dict[str, Any]
    metadata: dict[str, Any]
    recovery_gates: dict[str, Any]
    layout: RegisterLayout | None = None


def _pad_amplitudes(amplitudes: Any) -> Any:
    amplitudes = np.asarray(amplitudes, dtype=np.complex128).reshape(-1)
    if amplitudes.size < 2:
        amplitudes = np.pad(amplitudes, (0, 2 - amplitudes.size))
    n = 1 << (len(amplitudes) - 1).bit_length()
    out = np.zeros(n, dtype=np.complex128)
    out[: len(amplitudes)] = amplitudes
    norm = float(np.linalg.norm(out))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("amplitudes must have positive finite norm")
    return out / norm


def _append_state_preparation(qc: Any, qubits: Any, vec: Any) -> Any:
    try:
        from qiskit.circuit.library import StatePreparation

        qc.append(StatePreparation(vec), list(qubits))
    except Exception:
        qc.initialize(vec, list(qubits))


def build_static_action_circuit(amplitudes: Any, measure: bool = True) -> RAQICCircuitBuildResult:
    require_qiskit()
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

    vec = _pad_amplitudes(amplitudes)
    n_qubits = int(np.log2(len(vec)))
    q = QuantumRegister(n_qubits, "action")
    c = ClassicalRegister(n_qubits, "c_action")
    qc = QuantumCircuit(q, c, name="raqic_static_action")
    _append_state_preparation(qc, q, vec)
    if measure:
        qc.measure(q, c)
    return RAQICCircuitBuildResult(
        circuit=qc,
        parameter_map={"amplitudes": vec.tolist()},
        registers={"action": n_qubits, "c_action": n_qubits},
        metadata={"mode": "static", "finite_complex_projection": True},
        recovery_gates={"normalization": float(np.vdot(vec, vec).real)},
        layout=RegisterLayout(
            action_qubits=tuple(range(n_qubits)),
            classical_action_bits=tuple(range(n_qubits)),
        ),
    )


def build_interference_action_circuit(
    amplitudes: Any,
    unitary: np.ndarray,
    measure: bool = True,
    *,
    action_graph_hash: str,
) -> RAQICCircuitBuildResult:
    """Prepare action amplitudes, apply the certified padded mixer, and measure."""
    require_qiskit()
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    from qiskit.circuit.library import UnitaryGate

    if not action_graph_hash:
        raise ValueError("interference circuit requires a nonempty action graph hash")
    raw = np.asarray(amplitudes, dtype=np.complex128).reshape(-1)
    action_count = int(raw.size)
    vec = _pad_amplitudes(raw)
    supplied = np.asarray(unitary, dtype=np.complex128)
    if supplied.shape != (action_count, action_count):
        raise ValueError("unitary shape must match the unpadded action count")
    padded = np.eye(vec.size, dtype=np.complex128)
    padded[:action_count, :action_count] = supplied
    residual = float(np.max(np.abs(padded.conj().T @ padded - np.eye(vec.size))))
    if not np.isfinite(residual) or residual > 1e-10:
        raise ValueError(f"interference unitary is not unitary: residual={residual}")
    n_qubits = int(np.log2(vec.size))
    q = QuantumRegister(n_qubits, "action")
    c = ClassicalRegister(n_qubits, "c_action")
    qc = QuantumCircuit(q, c, name="raqic_interference_action")
    _append_state_preparation(qc, q, vec)
    qc.append(UnitaryGate(padded, label="semantic_interference_mixer"), list(q))
    if measure:
        qc.measure(q, c)
    return RAQICCircuitBuildResult(
        circuit=qc,
        parameter_map={
            "amplitudes": vec.tolist(),
            "action_count": action_count,
        },
        registers={"action": n_qubits, "c_action": n_qubits},
        metadata={
            "mode": "interference",
            "finite_complex_projection": True,
            "authority_preserving_unitary": True,
            "unused_basis_states": int(vec.size - action_count),
            "action_graph_hash": str(action_graph_hash),
        },
        recovery_gates={
            "normalization": float(np.vdot(vec, vec).real),
            "unitarity_residual": residual,
        },
        layout=RegisterLayout(
            action_qubits=tuple(range(n_qubits)),
            classical_action_bits=tuple(range(n_qubits)),
        ),
    )


def build_deferred_control_circuit(
    amplitudes: Any,
    feedback_phases: Any | None = None,
    *,
    measure: bool = True,
) -> RAQICCircuitBuildResult:
    """Coherently defer a record and use it as a feedback control.

    The record register copies the action computational basis without
    measurement.  Controlled phase feedback then acts on the action register.
    Because the feedback is diagonal, it preserves authority-zero amplitudes
    and the action marginal while realizing a genuine coherent deferred record.
    """
    require_qiskit()
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

    vec = _pad_amplitudes(amplitudes)
    n = int(np.log2(len(vec)))
    action = QuantumRegister(n, "action")
    record = QuantumRegister(n, "record")
    c_action = ClassicalRegister(n, "c_action")
    qc = QuantumCircuit(action, record, c_action, name="raqic_deferred_control")
    _append_state_preparation(qc, action, vec)
    for aq, rq in zip(action, record, strict=True):
        qc.cx(aq, rq)
    phases = np.zeros(n, dtype=float)
    if feedback_phases is not None:
        raw = np.asarray(feedback_phases, dtype=float).reshape(-1)
        phases[: min(n, raw.size)] = raw[:n]
    for j in range(n):
        if phases[j] != 0.0:
            qc.crz(float(phases[j]), record[j], action[j])
    if measure:
        qc.measure(action, c_action)
    return RAQICCircuitBuildResult(
        circuit=qc,
        parameter_map={"amplitudes": vec.tolist(), "feedback_phases": phases.tolist()},
        registers={"action": n, "record": n, "c_action": n},
        metadata={
            "mode": "deferred",
            "coherent_record": True,
            "controlled_feedback": True,
            "feedback_preserves_action_marginal": True,
        },
        recovery_gates={"normalization": float(np.vdot(vec, vec).real)},
        layout=RegisterLayout(
            action_qubits=tuple(range(n)),
            record_qubits=tuple(range(n, 2 * n)),
            classical_action_bits=tuple(range(n)),
        ),
    )


def _conditional_feedback_block(qc: Any, q: Any, c: Any, outcome: int, phases: np.ndarray) -> Any:
    """Apply legal-subspace-preserving dynamic phase feedback."""
    phases = np.asarray(phases, dtype=float)
    if hasattr(qc, "if_test"):
        with qc.if_test((c, int(outcome))):
            for j, qb in enumerate(q):
                qc.rz(float(phases[j % len(phases)]), qb)
    else:  # pragma: no cover - alternate Qiskit API path
        for j, qb in enumerate(q):
            inst = qc.rz(float(phases[j % len(phases)]), qb)
            with contextlib.suppress(Exception):
                inst.c_if(c, int(outcome))


def build_dynamic_recursive_circuit(
    amplitudes: Any,
    rounds: int = 1,
    feedback_phases: Any | None = None,
) -> RAQICCircuitBuildResult:
    """Build a measured-record, reset/reprepare, classically-fed-back circuit."""
    require_qiskit()
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

    if rounds < 1:
        raise ValueError("rounds must be at least 1 for a dynamic circuit")
    vec = _pad_amplitudes(amplitudes)
    n = int(np.log2(len(vec)))
    q = QuantumRegister(n, "action")
    c = ClassicalRegister(n, "c_action")
    qc = QuantumCircuit(q, c, name="raqic_dynamic_recursive")
    outcome_count = 1 << n
    if feedback_phases is None:
        feedback_phases = [
            np.array([0.05 * (y + 1) * (j + 1) for j in range(n)], dtype=float)
            for y in range(outcome_count)
        ]
    else:
        feedback_phases = [np.asarray(row, dtype=float) for row in feedback_phases]
        if len(feedback_phases) < outcome_count:
            raise ValueError("feedback_phases must provide one phase vector per outcome")

    _append_state_preparation(qc, q, vec)
    for _round in range(rounds):
        qc.measure(q, c)
        qc.reset(q)
        _append_state_preparation(qc, q, vec)
        for y in range(outcome_count):
            _conditional_feedback_block(qc, q, c, y, feedback_phases[y])
    # Final authoritative measurement after the requested feedback rounds.
    qc.measure(q, c)
    return RAQICCircuitBuildResult(
        circuit=qc,
        parameter_map={
            "amplitudes": vec.tolist(),
            "rounds": rounds,
            "feedback_phases": [row.tolist() for row in feedback_phases],
        },
        registers={"action": n, "c_action": n},
        metadata={
            "mode": "dynamic_recursive",
            "dynamic_feedback": True,
            "feedforward_api": "if_test",
            "authority_preserving_feedback": "diagonal_phase",
        },
        recovery_gates={"normalization": float(np.vdot(vec, vec).real), "rounds": rounds},
        layout=RegisterLayout(
            action_qubits=tuple(range(n)),
            classical_action_bits=tuple(range(n)),
        ),
    )


def _legal_coin_matrix(coin_qubits: int, legal_basis: Any | None = None) -> np.ndarray:
    """Unitary coin that mixes only authority-approved computational states."""
    dim = 1 << int(coin_qubits)
    legal = list(range(dim)) if legal_basis is None else sorted({int(x) for x in legal_basis})
    if any(index < 0 or index >= dim for index in legal):
        raise ValueError("legal_basis contains an out-of-range coin state")
    matrix = np.eye(dim, dtype=np.complex128)
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    for offset in range(0, len(legal) - 1, 2):
        a, b = legal[offset], legal[offset + 1]
        matrix[a, a] = inv_sqrt2
        matrix[a, b] = inv_sqrt2
        matrix[b, a] = inv_sqrt2
        matrix[b, b] = -inv_sqrt2
    return matrix


def _conditional_ring_shift_matrix(coin_qubits: int, position_qubits: int) -> np.ndarray:
    total = coin_qubits + position_qubits
    dim = 1 << total
    position_modulus = 1 << position_qubits
    matrix = np.zeros((dim, dim), dtype=np.complex128)
    coin_mask = (1 << coin_qubits) - 1
    for basis in range(dim):
        coin = basis & coin_mask
        position = (basis >> coin_qubits) & (position_modulus - 1)
        delta = 1 if (coin & 1) == 0 else -1
        new_position = (position + delta) % position_modulus
        target = coin | (new_position << coin_qubits)
        matrix[target, basis] = 1.0
    return matrix


def build_quantum_walk_variant(
    coin_amplitudes: Any,
    n_positions: int = 4,
    *,
    steps: int = 2,
    measure: bool = True,
    legal_basis: Any | None = None,
) -> RAQICCircuitBuildResult:
    """Build a discrete-time coined walk on a periodic position register."""
    require_qiskit()
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    from qiskit.circuit.library import UnitaryGate

    if steps < 1:
        raise ValueError("walk steps must be positive")
    coin = _pad_amplitudes(coin_amplitudes)
    cqb = int(np.log2(len(coin)))
    pqb = bits_needed(max(2, int(n_positions)))
    actual_positions = 1 << pqb
    qcoin = QuantumRegister(cqb, "coin")
    qpos = QuantumRegister(pqb, "position")
    c_action = ClassicalRegister(cqb, "c_action")
    c_position = ClassicalRegister(pqb, "c_position")
    qc = QuantumCircuit(qcoin, qpos, c_action, c_position, name="raqic_quantum_walk")
    _append_state_preparation(qc, qcoin, coin)
    legal = (
        tuple(range(len(np.asarray(coin_amplitudes).reshape(-1))))
        if legal_basis is None
        else tuple(int(x) for x in legal_basis)
    )
    coin_gate = UnitaryGate(
        _legal_coin_matrix(cqb, legal),
        label="legal_coin_operator",
    )
    shift = UnitaryGate(
        _conditional_ring_shift_matrix(cqb, pqb),
        label="conditional_ring_shift",
    )
    for _ in range(steps):
        qc.append(coin_gate, list(qcoin))
        qc.append(shift, list(qcoin) + list(qpos))
    if measure:
        qc.measure(qcoin, c_action)
        qc.measure(qpos, c_position)
    return RAQICCircuitBuildResult(
        circuit=qc,
        parameter_map={
            "coin": coin.tolist(),
            "steps": int(steps),
            "n_positions": int(actual_positions),
        },
        registers={
            "coin": cqb,
            "position": pqb,
            "c_action": cqb,
            "c_position": pqb,
        },
        metadata={
            "mode": "walk",
            "walk_steps": int(steps),
            "coin_operator": "authority_subspace_pairwise_hadamard",
            "legal_basis": list(legal),
            "shift": "conditional_periodic_plus_minus_one",
            "authority_preserving": True,
        },
        recovery_gates={"coin_norm": float(np.vdot(coin, coin).real)},
        layout=RegisterLayout(
            action_qubits=tuple(range(cqb)),
            position_qubits=tuple(range(cqb, cqb + pqb)),
            classical_action_bits=tuple(range(cqb)),
            classical_position_bits=tuple(range(cqb, cqb + pqb)),
        ),
    )


def build_scale_boundary_circuit(
    child_amplitudes: Any, parent_amplitudes: Any
) -> RAQICCircuitBuildResult:
    require_qiskit()
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

    child = _pad_amplitudes(child_amplitudes)
    parent = _pad_amplitudes(parent_amplitudes)
    cq = int(np.log2(len(child)))
    pq = int(np.log2(len(parent)))
    qc_child = QuantumRegister(cq, "child_record")
    qc_parent = QuantumRegister(pq, "parent_record")
    c = ClassicalRegister(cq + pq, "c_scale")
    qc = QuantumCircuit(qc_child, qc_parent, c, name="raqic_scale_boundary")
    _append_state_preparation(qc, qc_child, child)
    _append_state_preparation(qc, qc_parent, parent)
    for k in range(min(cq, pq)):
        qc.cx(qc_child[k], qc_parent[k])
    qc.measure(qc_child[:] + qc_parent[:], c)
    return RAQICCircuitBuildResult(
        circuit=qc,
        parameter_map={"child": child.tolist(), "parent": parent.tolist()},
        registers={"child": cq, "parent": pq, "c_scale": cq + pq},
        metadata={"mode": "scale_boundary"},
        recovery_gates={
            "child_norm": float(np.vdot(child, child).real),
            "parent_norm": float(np.vdot(parent, parent).real),
        },
        layout=RegisterLayout(
            action_qubits=tuple(range(cq)),
            record_qubits=tuple(range(cq, cq + pq)),
            classical_action_bits=tuple(range(cq)),
            classical_record_bits=tuple(range(cq, cq + pq)),
        ),
    )
