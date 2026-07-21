"""Exact native parameterized state preparation for Aer runtime binding.

The production runtime-binding path deliberately avoids ``RawFeatureVector``
and ``ParameterizedInitialize``.  A padded complex amplitude row is converted
into a fixed-topology Möttönen-style magnitude tree plus a Walsh phase
polynomial.  The resulting circuit contains only ordinary parameterized
``ry``/``rz`` gates and ``cx`` gates before it is submitted to Aer.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from typing import Any, cast

import numpy as np


class RequiredNativeRuntimeBindingError(RuntimeError):
    """Raised when the required-native runtime-binding contract is unavailable."""


@dataclass(frozen=True)
class NativePreparationLayout:
    action_count: int
    padded_dimension: int
    qubit_count: int
    magnitude_parameter_count: int
    phase_parameter_count: int
    parameter_order: tuple[str, ...]
    action_to_basis: tuple[int, ...]
    unused_basis: tuple[int, ...]


@dataclass(frozen=True)
class NativePreparationAngles:
    magnitude_angles: np.ndarray
    phase_angles: np.ndarray
    normalized_amplitudes: np.ndarray


@dataclass(frozen=True)
class CompiledRuntimeBindingTemplate:
    circuit: Any
    layout: NativePreparationLayout
    ordered_parameters: tuple[Any, ...]
    action_qubits: tuple[int, ...]
    classical_action_bits: tuple[int, ...]
    template_sha256: str
    target_fingerprint: str
    instruction_inventory: tuple[str, ...]
    qiskit_version: str
    aer_version: str
    method: str
    device: str
    precision: str
    measure: bool


def _package_version(name: str) -> str:
    try:
        return str(importlib_metadata.version(name))
    except importlib_metadata.PackageNotFoundError:
        return "unavailable"


def make_native_preparation_layout(action_count: int) -> NativePreparationLayout:
    count = int(action_count)
    if count < 1:
        raise ValueError("action_count must be positive")
    qubits = max(1, int(math.ceil(math.log2(max(2, count)))))
    dimension = 1 << qubits
    magnitude_names = tuple(f"owl_mag[{index}]" for index in range(dimension - 1))
    # One coefficient for each nonempty Z-string. The omitted empty-string
    # coefficient is a physically irrelevant global phase.
    phase_names = tuple(f"owl_phase[{index}]" for index in range(dimension - 1))
    return NativePreparationLayout(
        action_count=count,
        padded_dimension=dimension,
        qubit_count=qubits,
        magnitude_parameter_count=dimension - 1,
        phase_parameter_count=dimension - 1,
        parameter_order=(*magnitude_names, *phase_names),
        action_to_basis=tuple(range(count)),
        unused_basis=tuple(range(count, dimension)),
    )


def probabilities_and_phases_to_amplitudes(
    probabilities: Any,
    phases: Any,
    *,
    padded_dimension: int,
) -> np.ndarray:
    p = np.asarray(probabilities, dtype=np.float64)
    phase = np.asarray(phases, dtype=np.float64)
    if p.ndim != 2 or p.shape != phase.shape:
        raise ValueError("probabilities and phases must have equal [N,A] shape")
    if p.shape[1] > int(padded_dimension):
        raise ValueError("action dimension exceeds padded state dimension")
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(phase)):
        raise FloatingPointError("probabilities and phases must be finite")
    clipped = np.maximum(p, 0.0)
    totals = clipped.sum(axis=1, keepdims=True)
    if np.any(totals <= 0.0):
        raise FloatingPointError("probability rows must contain positive mass")
    clipped /= totals
    rows = np.zeros((p.shape[0], int(padded_dimension)), dtype=np.complex128)
    rows[:, : p.shape[1]] = np.sqrt(clipped) * np.exp(1j * phase)
    norms = np.linalg.norm(rows, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
        raise FloatingPointError("amplitude rows could not be normalized")
    rows /= norms[:, None]
    return rows


def _parity_sign(index: int, mask: int) -> float:
    return -1.0 if (int(index & mask).bit_count() & 1) else 1.0


def amplitudes_to_native_angles(
    amplitudes: Any,
    *,
    zero_tolerance: float = 1e-15,
) -> NativePreparationAngles:
    """Convert one complex state into magnitude-tree and phase-polynomial angles."""

    vector = np.asarray(amplitudes, dtype=np.complex128).reshape(-1)
    if vector.size < 2 or vector.size & (vector.size - 1):
        raise ValueError("amplitude dimension must be a power of two >= 2")
    if not np.all(np.isfinite(vector.real)) or not np.all(np.isfinite(vector.imag)):
        raise FloatingPointError("amplitudes must be finite")
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= float(zero_tolerance):
        raise FloatingPointError("all-zero amplitude input is invalid")
    normalized = vector / norm
    probabilities = np.abs(normalized) ** 2
    qubits = int(round(math.log2(vector.size)))

    magnitude_angles: list[float] = []
    for depth in range(qubits):
        target_qubit = qubits - 1 - depth
        half = 1 << target_qubit
        for prefix in range(1 << depth):
            base = prefix << (target_qubit + 1)
            mass_zero = float(probabilities[base : base + half].sum())
            mass_one = float(probabilities[base + half : base + 2 * half].sum())
            if mass_zero <= zero_tolerance and mass_one <= zero_tolerance:
                angle = 0.0
            else:
                angle = 2.0 * math.atan2(
                    math.sqrt(max(0.0, mass_one)),
                    math.sqrt(max(0.0, mass_zero)),
                )
            magnitude_angles.append(float(angle))

    # Expand desired basis phases into a Walsh basis. For nonempty mask S, a
    # parity RZ(theta_S) contributes -theta_S/2 * (-1)^parity(x & S).
    # The omitted Walsh coefficient for mask zero is global phase only.
    basis_phases = np.angle(normalized).astype(np.float64, copy=False)
    basis_phases[np.abs(normalized) <= zero_tolerance] = 0.0
    dimension = int(vector.size)
    phase_angles: list[float] = []
    for mask in range(1, dimension):
        coefficient = sum(
            float(basis_phases[index]) * _parity_sign(index, mask) for index in range(dimension)
        ) / float(dimension)
        phase_angles.append(float(-2.0 * coefficient))

    return NativePreparationAngles(
        magnitude_angles=np.asarray(magnitude_angles, dtype=np.float64),
        phase_angles=np.asarray(phase_angles, dtype=np.float64),
        normalized_amplitudes=np.asarray(normalized, dtype=np.complex128),
    )


def _transform_uniform_rotation_angles(angles: list[Any]) -> None:
    """Apply Qiskit's Shende uniformly-controlled-rotation transform in place."""

    def recurse(start: int, end: int, reversed_decomposition: bool) -> None:
        half = (end - start) // 2
        for index in range(start, start + half):
            left = angles[index]
            right = angles[index + half]
            average = (left + right) / 2.0
            difference = (left - right) / 2.0
            if reversed_decomposition:
                angles[index + half], angles[index] = average, difference
            else:
                angles[index], angles[index + half] = average, difference
        if half > 1:
            recurse(start, start + half, False)
            recurse(start + half, end, True)

    if len(angles) > 1:
        recurse(0, len(angles), False)


def _append_uniformly_controlled_ry(
    circuit: Any,
    *,
    target: int,
    controls: list[int],
    desired_angles: list[Any],
) -> None:
    if not controls:
        circuit.ry(desired_angles[0], target)
        return
    transformed = list(desired_angles)
    _transform_uniform_rotation_angles(transformed)
    for index, angle in enumerate(transformed):
        circuit.ry(angle, target)
        if index != len(transformed) - 1:
            binary = np.binary_repr(index + 1)
            control_index = len(binary) - len(binary.rstrip("0"))
        else:
            control_index = len(controls) - 1
        circuit.cx(controls[control_index], target)


def _append_parity_phase(circuit: Any, *, mask: int, angle: Any, qubits: int) -> None:
    members = [qubit for qubit in range(qubits) if mask & (1 << qubit)]
    target = members[-1]
    controls = members[:-1]
    for control in controls:
        circuit.cx(control, target)
    circuit.rz(angle, target)
    for control in reversed(controls):
        circuit.cx(control, target)


def build_native_parameterized_state_preparation(
    layout: NativePreparationLayout,
) -> tuple[Any, tuple[Any, ...]]:
    """Build an exact fixed topology from ordinary RY/RZ/CX instructions."""

    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector

    qubits = int(layout.qubit_count)
    dimension = int(layout.padded_dimension)
    magnitude = ParameterVector("owl_mag", layout.magnitude_parameter_count)
    phase = ParameterVector("owl_phase", layout.phase_parameter_count)
    circuit = QuantumCircuit(qubits, name="owl_exact_native_state_preparation")

    parameter_index = 0
    for depth in range(qubits):
        target = qubits - 1 - depth
        controls = list(range(target + 1, qubits))
        width = 1 << depth
        desired = [magnitude[parameter_index + offset] for offset in range(width)]
        _append_uniformly_controlled_ry(
            circuit,
            target=target,
            controls=controls,
            desired_angles=desired,
        )
        parameter_index += width

    for mask in range(1, dimension):
        _append_parity_phase(
            circuit,
            mask=mask,
            angle=phase[mask - 1],
            qubits=qubits,
        )

    return circuit, (*tuple(magnitude), *tuple(phase))


def _template_payload(circuit: Any, layout: NativePreparationLayout) -> dict[str, Any]:
    operations: list[dict[str, Any]] = []
    for instruction in circuit.data:
        operation = instruction.operation
        operations.append(
            {
                "name": str(operation.name),
                "qubits": [int(circuit.find_bit(bit).index) for bit in instruction.qubits],
                "clbits": [int(circuit.find_bit(bit).index) for bit in instruction.clbits],
                "params": [str(value) for value in operation.params],
            }
        )
    return {
        "layout": {
            "action_count": layout.action_count,
            "padded_dimension": layout.padded_dimension,
            "qubit_count": layout.qubit_count,
            "parameter_order": layout.parameter_order,
        },
        "operations": operations,
    }


def _target_fingerprint(simulator: Any, *, method: str, device: str, precision: str) -> str:
    payload = {
        "backend": type(simulator).__name__,
        "method": str(method),
        "device": str(device).upper(),
        "precision": str(precision),
        "available_methods": sorted(str(item) for item in simulator.available_methods()),
        "available_devices": sorted(str(item).upper() for item in simulator.available_devices()),
        "operations": sorted(str(item) for item in simulator.target.operation_names),
        "qiskit": _package_version("qiskit"),
        "qiskit_aer": _package_version("qiskit-aer"),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def assert_native_instruction_inventory(
    circuit: Any,
    simulator: Any,
) -> tuple[str, ...]:
    inventory = tuple(sorted({str(item.operation.name) for item in circuit.data}))
    forbidden = {
        "ParameterizedInitialize",
        "parameterized_initialize",
        "initialize",
        "state_preparation",
        "StatePreparation",
        "raw_feature_vector",
    }
    present_forbidden = sorted(set(inventory) & forbidden)
    if present_forbidden:
        raise RequiredNativeRuntimeBindingError(
            f"forbidden placeholder instructions remain after compilation: {present_forbidden}"
        )
    supported = {str(item) for item in simulator.target.operation_names}
    unsupported = sorted(set(inventory) - supported)
    if unsupported:
        raise RequiredNativeRuntimeBindingError(
            f"compiled runtime-binding circuit contains unsupported operations: {unsupported}"
        )
    return inventory


def transpile_native_template(
    layout: NativePreparationLayout,
    simulator: Any,
    *,
    method: str = "statevector",
    device: str = "GPU",
    precision: str = "double",
    optimization_level: int = 0,
    measure: bool = False,
) -> CompiledRuntimeBindingTemplate:
    """Transpile and verify the fixed native symbolic template before binding."""

    from qiskit import ClassicalRegister, transpile

    source, ordered_source_parameters = build_native_parameterized_state_preparation(layout)
    compiled = transpile(source, backend=simulator, optimization_level=int(optimization_level))
    classical_bits: tuple[int, ...] = ()
    if measure:
        register = ClassicalRegister(layout.qubit_count, "action")
        compiled.add_register(register)
        compiled.measure(range(layout.qubit_count), register)
        classical_bits = tuple(range(layout.qubit_count))
    else:
        compiled.save_statevector(label="statevector")

    by_name = {str(parameter): parameter for parameter in compiled.parameters}
    ordered_parameters = tuple(by_name[str(parameter)] for parameter in ordered_source_parameters)
    if tuple(str(parameter) for parameter in ordered_parameters) != layout.parameter_order:
        raise RequiredNativeRuntimeBindingError("compiled parameter order changed unexpectedly")
    expected_count = layout.magnitude_parameter_count + layout.phase_parameter_count
    if len(ordered_parameters) != expected_count:
        raise RequiredNativeRuntimeBindingError("compiled parameter count is incomplete")

    inventory = assert_native_instruction_inventory(compiled, simulator)
    payload = _template_payload(compiled, layout)
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return CompiledRuntimeBindingTemplate(
        circuit=compiled,
        layout=layout,
        ordered_parameters=ordered_parameters,
        action_qubits=tuple(range(layout.qubit_count)),
        classical_action_bits=classical_bits,
        template_sha256=digest,
        target_fingerprint=_target_fingerprint(
            simulator,
            method=method,
            device=device,
            precision=precision,
        ),
        instruction_inventory=inventory,
        qiskit_version=_package_version("qiskit"),
        aer_version=_package_version("qiskit-aer"),
        method=str(method),
        device=str(device).upper(),
        precision=str(precision),
        measure=bool(measure),
    )


def make_parameter_bind_batch(
    compiled: CompiledRuntimeBindingTemplate,
    probabilities: Any,
    phases: Any,
) -> tuple[dict[Any, list[float]], np.ndarray]:
    amplitudes = probabilities_and_phases_to_amplitudes(
        probabilities,
        phases,
        padded_dimension=compiled.layout.padded_dimension,
    )
    angle_rows = [amplitudes_to_native_angles(row) for row in amplitudes]
    magnitude = np.stack([row.magnitude_angles for row in angle_rows], axis=0)
    phase = np.stack([row.phase_angles for row in angle_rows], axis=0)
    values = np.concatenate([magnitude, phase], axis=1)
    if values.shape[1] != len(compiled.ordered_parameters):
        raise RequiredNativeRuntimeBindingError("native binding value count mismatch")
    binds = {
        parameter: values[:, index].astype(float).tolist()
        for index, parameter in enumerate(compiled.ordered_parameters)
    }
    return binds, amplitudes


def _statevector_probabilities(statevector: Any, *, action_count: int) -> np.ndarray:
    state = np.asarray(statevector, dtype=np.complex128).reshape(-1)
    probabilities = np.abs(state) ** 2
    out = probabilities[: int(action_count)].astype(np.float64, copy=True)
    total = float(out.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise FloatingPointError("native statevector action probabilities are invalid")
    return cast(np.ndarray, out / total)


def _align_global_phase(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    overlap = np.vdot(expected, actual)
    if abs(overlap) == 0.0:
        return actual
    return actual * np.exp(-1j * np.angle(overlap))


def preflight_native_runtime_binding(
    *,
    action_count: int,
    method: str = "statevector",
    device: str = "GPU",
    precision: str = "double",
    strict_gpu: bool = True,
    tolerance: float = 1e-10,
    batch_size: int = 8,
    seed: int = 9303,
    simulator_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Prove the exact production binding path before scientific tick zero."""

    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import StatePreparation
    from qiskit_aer import AerSimulator

    from .aer_evidence import parse_aer_gpu_evidence

    requested_device = str(device).upper()
    options: dict[str, Any] = {
        "method": str(method),
        "device": requested_device,
        "precision": str(precision),
        "seed_simulator": int(seed),
        "runtime_parameter_bind_enable": True,
    }
    options.update(dict(simulator_options or {}))
    simulator = AerSimulator(**options)
    available_devices = tuple(str(item).upper() for item in simulator.available_devices())
    if requested_device not in available_devices:
        raise RequiredNativeRuntimeBindingError(
            f"Aer device {requested_device!r} unavailable; available={available_devices}"
        )
    if strict_gpu and requested_device != "GPU":
        raise RequiredNativeRuntimeBindingError("required-native flagship preflight requires GPU")

    layout = make_native_preparation_layout(action_count)
    compiled = transpile_native_template(
        layout,
        simulator,
        method=method,
        device=requested_device,
        precision=precision,
        measure=False,
    )

    count = max(1, min(int(batch_size), 8))
    rng = np.random.default_rng(int(seed))
    vectors: list[np.ndarray] = []
    for basis in (0, action_count - 1):
        row: np.ndarray = np.zeros((layout.padded_dimension,), dtype=np.complex128)
        row[int(basis)] = 1.0
        vectors.append(row)
    uniform: np.ndarray = np.zeros((layout.padded_dimension,), dtype=np.complex128)
    uniform[:action_count] = 1.0 / math.sqrt(action_count)
    vectors.append(uniform)
    pair_real = np.zeros_like(uniform)
    pair_real[0] = math.sqrt(0.4)
    pair_real[min(1, action_count - 1)] = math.sqrt(0.6)
    vectors.append(pair_real)
    pair_complex = np.zeros_like(uniform)
    pair_complex[0] = math.sqrt(0.35)
    pair_complex[min(1, action_count - 1)] = math.sqrt(0.65) * np.exp(0.73j)
    vectors.append(pair_complex)
    random_row = np.zeros_like(uniform)
    random_row[:action_count] = rng.normal(size=action_count) + 1j * rng.normal(size=action_count)
    random_row /= np.linalg.norm(random_row)
    vectors.append(random_row)
    sparse = np.zeros_like(uniform)
    legal = np.asarray(sorted({0, min(3, action_count - 1), action_count - 1}), dtype=int)
    sparse[legal] = rng.normal(size=legal.size) + 1j * rng.normal(size=legal.size)
    sparse /= np.linalg.norm(sparse)
    vectors.append(sparse)
    while len(vectors) < count:
        row = np.zeros_like(uniform)
        row[:action_count] = rng.normal(size=action_count) + 1j * rng.normal(size=action_count)
        row /= np.linalg.norm(row)
        vectors.append(row)
    vectors = vectors[:count]

    probabilities = np.stack([np.abs(row[:action_count]) ** 2 for row in vectors])
    phases = np.stack([np.angle(row[:action_count]) for row in vectors])
    binds, expected_amplitudes = make_parameter_bind_batch(compiled, probabilities, phases)
    result = simulator.run(compiled.circuit, parameter_binds=[binds]).result()
    if not bool(result.success):
        raise RequiredNativeRuntimeBindingError(f"Aer preflight failed: {result.status}")
    if len(result.results) != len(vectors):
        raise RequiredNativeRuntimeBindingError(
            "Aer runtime binding did not emit one result per binding row"
        )

    max_statevector_error = 0.0
    max_probability_error = 0.0
    max_reference_error = 0.0
    max_unused_mass = 0.0
    gpu_evidence: list[dict[str, Any]] = []
    for index, expected in enumerate(expected_amplitudes):
        actual = np.asarray(result.get_statevector(index), dtype=np.complex128)
        aligned = _align_global_phase(actual, expected)
        max_statevector_error = max(
            max_statevector_error,
            float(np.max(np.abs(aligned - expected))),
        )
        actual_probability = _statevector_probabilities(actual, action_count=action_count)
        expected_probability = np.abs(expected[:action_count]) ** 2
        expected_probability /= expected_probability.sum()
        max_probability_error = max(
            max_probability_error,
            float(np.max(np.abs(actual_probability - expected_probability))),
        )
        max_unused_mass = max(
            max_unused_mass,
            float(np.sum(np.abs(actual[action_count:]) ** 2)),
        )

        reference = QuantumCircuit(layout.qubit_count)
        reference.append(StatePreparation(expected), range(layout.qubit_count))
        reference.save_statevector(label="statevector")
        reference = transpile(reference, backend=simulator, optimization_level=0)
        reference_result = simulator.run(reference).result()
        reference_state = np.asarray(
            reference_result.get_statevector(0),
            dtype=np.complex128,
        )
        reference_aligned = _align_global_phase(actual, reference_state)
        max_reference_error = max(
            max_reference_error,
            float(np.max(np.abs(reference_aligned - reference_state))),
        )
        evidence = parse_aer_gpu_evidence(dict(result.results[index].metadata or {}))
        gpu_evidence.append(evidence)

    threshold = float(tolerance)
    if max_statevector_error > threshold or max_probability_error > threshold:
        raise RequiredNativeRuntimeBindingError(
            "native runtime-binding output disagrees with dense reference: "
            f"statevector={max_statevector_error:.3e}, probability={max_probability_error:.3e}"
        )
    if max_reference_error > threshold:
        raise RequiredNativeRuntimeBindingError(
            "native runtime-binding output disagrees with concrete Qiskit oracle: "
            f"{max_reference_error:.3e}"
        )
    if max_unused_mass > threshold:
        raise RequiredNativeRuntimeBindingError(
            f"native runtime-binding produced unused-basis mass {max_unused_mass:.3e}"
        )
    if strict_gpu and not all(bool(item.get("verified")) for item in gpu_evidence):
        raise RequiredNativeRuntimeBindingError(
            "required-native preflight lacks positive Aer GPU execution metadata"
        )

    return {
        "schema_version": "owl.raqic.required-native-preflight.v1",
        "passed": True,
        "scientific_ticks_started": 0,
        "runtime_binding_policy": "required_native",
        "state_preparation_strategy": "exact_native_rotation_tree",
        "runtime_binding_used": True,
        "automatic_fallback_allowed": False,
        "automatic_fallback_used": False,
        "device": requested_device,
        "method": str(method),
        "precision": str(precision),
        "batch_size": len(vectors),
        "template_sha256": compiled.template_sha256,
        "target_fingerprint": compiled.target_fingerprint,
        "instruction_inventory": list(compiled.instruction_inventory),
        "max_statevector_error": max_statevector_error,
        "max_probability_error": max_probability_error,
        "max_reference_error": max_reference_error,
        "max_unused_basis_mass": max_unused_mass,
        "gpu_evidence": gpu_evidence,
        "qiskit_version": compiled.qiskit_version,
        "qiskit_aer_version": compiled.aer_version,
    }
