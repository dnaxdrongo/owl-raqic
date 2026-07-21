from __future__ import annotations

import numpy as np
import pytest

from owl_raqic.qiskit_backend.native_state_preparation import (
    amplitudes_to_native_angles,
    make_native_preparation_layout,
    preflight_native_runtime_binding,
    transpile_native_template,
)


def test_native_layout_preserves_22_action_padding() -> None:
    layout = make_native_preparation_layout(22)
    assert layout.qubit_count == 5
    assert layout.padded_dimension == 32
    assert layout.action_to_basis == tuple(range(22))
    assert layout.unused_basis == tuple(range(22, 32))
    assert layout.magnitude_parameter_count == 31
    assert layout.phase_parameter_count == 31


def test_all_zero_state_is_rejected() -> None:
    with pytest.raises(FloatingPointError, match="all-zero"):
        amplitudes_to_native_angles(np.zeros(32, dtype=np.complex128))


def test_compiled_template_contains_no_placeholder_instruction() -> None:
    pytest.importorskip("qiskit_aer")
    from qiskit_aer import AerSimulator

    simulator = AerSimulator(method="statevector", device="CPU")
    compiled = transpile_native_template(
        make_native_preparation_layout(22),
        simulator,
        device="CPU",
    )
    lowered = {name.lower() for name in compiled.instruction_inventory}
    assert "parameterizedinitialize" not in lowered
    assert "initialize" not in lowered
    assert "state_preparation" not in lowered
    assert lowered <= {"ry", "rz", "cx", "save_statevector"}


def test_cpu_runtime_binding_preflight_matches_dense_and_oracle() -> None:
    pytest.importorskip("qiskit_aer")
    report = preflight_native_runtime_binding(
        action_count=22,
        device="CPU",
        strict_gpu=False,
        batch_size=8,
        tolerance=1e-10,
    )
    assert report["passed"] is True
    assert report["runtime_binding_used"] is True
    assert report["automatic_fallback_used"] is False
    assert report["max_probability_error"] < 1e-10
    assert report["max_reference_error"] < 1e-10
    assert report["max_unused_basis_mass"] < 1e-10


def test_production_qiskit_backend_has_no_machine_learning_import() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src" / "owl_raqic" / "qiskit_backend"
    production = "\n".join(
        path.read_text(encoding="utf-8")
        for path in root.glob("*.py")
        if path.name != "native_state_preparation.py"
    )
    assert "from qiskit_machine_learning" not in production
    assert "import qiskit_machine_learning" not in production


def test_forced_validation_matrix_uses_native_runtime_binding() -> None:
    from owl_raqic.validation.circuit_matrix import validate_circuit_matrix

    probabilities: np.ndarray = np.zeros((2, 22), dtype=np.float64)
    probabilities[:, 0] = 0.6
    probabilities[:, 1] = 0.4
    report = validate_circuit_matrix(
        probabilities,
        families=("static",),
        strict_gpu=False,
        simulator_options={"runtime_parameter_bind_enable": True},
        tolerance=1e-10,
        kl_tolerance=1e-9,
    )
    assert report.passed is True
    execution = report.rows[0].metadata["execution"]
    assert execution["runtime_parameter_binding_used"] is True
    assert execution["automatic_fallback_used"] is False
    preflight = execution["runtime_parameter_binding_preflight"]
    assert preflight["passed"] is True
    assert "ParameterizedInitialize" not in preflight["instruction_inventory"]
