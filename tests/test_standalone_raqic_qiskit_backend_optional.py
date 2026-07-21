import numpy as np
import pytest

from owl_raqic.math.instruments import action_amplitudes
from owl_raqic.qiskit_backend.backend_profiles import (
    MissingQiskitError,
    gpu_aer_available,
    qiskit_available,
)
from owl_raqic.qiskit_backend.memory_guard import (
    density_matrix_memory_bytes,
    ensure_memory_within_limit,
    statevector_memory_bytes,
)
from owl_raqic.qiskit_backend.register_builder import bits_needed, build_register_map


def test_register_map_builds():
    rm = build_register_map(10, 2, include_scale=True, include_place=True)
    assert rm.action_qubits == 4
    assert rm.total_qubits == 7


def test_bits_needed():
    assert bits_needed(10) == 4
    assert bits_needed(1) == 1


def test_memory_guard_values():
    assert statevector_memory_bytes(4) == 16 * 16
    assert density_matrix_memory_bytes(2) == 16 * 16


def test_memory_guard_blocks_large_statevector():
    with pytest.raises(MemoryError):
        ensure_memory_within_limit(30, "statevector", limit_mb=1)


def test_gpu_not_required():
    assert isinstance(gpu_aer_available(), bool)


def test_qiskit_static_builder_skip_or_build():
    from owl_raqic.qiskit_backend.circuit_templates import build_static_action_circuit

    amps, probs = action_amplitudes(np.array([0.0, 1.0, -0.2]))
    if not qiskit_available():
        with pytest.raises(MissingQiskitError):
            build_static_action_circuit(amps)
    else:
        build = build_static_action_circuit(amps)
        assert build.recovery_gates["normalization"] == pytest.approx(1)


@pytest.mark.skipif(not qiskit_available(), reason="Qiskit not installed in this environment")
def test_qiskit_statevector_matches_numpy():
    from owl_raqic.qiskit_backend.circuit_templates import build_static_action_circuit
    from owl_raqic.qiskit_backend.execution import statevector_probabilities_from_circuit

    amps, probs = action_amplitudes(np.array([0.0, 1.0, -0.2]))
    build = build_static_action_circuit(amps)
    got = statevector_probabilities_from_circuit(build.circuit, n_actions=3)
    assert np.allclose(got, probs)
