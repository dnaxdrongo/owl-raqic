from __future__ import annotations

from typing import Any

import numpy as np

from .aer_runtime import run_aer_job
from .backend_profiles import qiskit_aer_available, require_qiskit
from .memory_guard import ensure_memory_within_limit


def statevector_probabilities_from_circuit(
    circuit: Any, n_actions: int | None = None, memory_limit_mb: float = 512.0
) -> Any:
    require_qiskit()
    from qiskit.quantum_info import Statevector

    circ = circuit.remove_final_measurements(inplace=False)
    ensure_memory_within_limit(circ.num_qubits, "statevector", memory_limit_mb)
    sv = Statevector.from_instruction(circ)
    probs = np.asarray(sv.probabilities(), dtype=float)
    if n_actions is not None:
        probs = probs[:n_actions]
        probs = probs / probs.sum()
    return probs


def aer_counts(circuit: Any, shots: int = 1024, seed: int | None = None) -> Any:
    require_qiskit()
    if not qiskit_aer_available():
        raise ImportError("qiskit-aer is not installed")
    from qiskit_aer import AerSimulator

    sim = AerSimulator(method="automatic", seed_simulator=seed)
    result = run_aer_job(sim, circuit, shots=shots)
    return result.get_counts()
