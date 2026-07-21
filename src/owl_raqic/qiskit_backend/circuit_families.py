from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from owl_raqic.math.action_graph import (
    action_family_edges,
    action_graph_hash,
    legal_subspace_unitary,
)

from .circuit_templates import (
    RAQICCircuitBuildResult,
    build_deferred_control_circuit,
    build_dynamic_recursive_circuit,
    build_interference_action_circuit,
    build_quantum_walk_variant,
    build_static_action_circuit,
)
from .result_decode import ActionBitLayout, counts_to_action_probabilities

Oracle = Callable[..., np.ndarray]


@dataclass(frozen=True)
class CircuitFamilySpec:
    name: str
    builder: Callable[..., RAQICCircuitBuildResult]
    oracle: Oracle
    exact: bool
    shot_based: bool
    dynamic: bool
    noisy: bool
    compatible_methods: tuple[str, ...]
    action_register: str
    classical_action_register: str | None
    qubit_estimator: Callable[..., int]
    memory_estimator: Callable[..., int]
    structural_validator: Callable[[RAQICCircuitBuildResult], None]

    def decode_counts(
        self,
        counts: Mapping[str, int],
        built: RAQICCircuitBuildResult,
        action_count: int,
        *,
        authority: np.ndarray | None = None,
    ) -> np.ndarray:
        if built.layout is None or not built.layout.classical_action_bits:
            raise ValueError(f"{self.name} did not declare classical action bits")
        return counts_to_action_probabilities(
            counts,
            ActionBitLayout(tuple(built.layout.classical_action_bits)),
            action_count,
            authority=authority,
            project_illegal_to_rest=bool(self.noisy),
            rest_index=0,
        )


def _normalize(probabilities: Any) -> np.ndarray:
    p = np.maximum(np.asarray(probabilities, dtype=np.float64), 0.0)
    total = float(p.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("probability row has invalid normalization")
    return p / total


def _static_oracle(probabilities: Any, **_kwargs: Any) -> np.ndarray:
    return _normalize(probabilities)


def _walk_oracle(
    probabilities: Any,
    *,
    steps: int = 2,
    n_positions: int = 4,
    legal_basis: Any | None = None,
    **_kwargs: Any,
) -> np.ndarray:
    """Independent exact oracle for the authority-preserving coined ring walk."""
    p = _normalize(probabilities)
    action_count = p.size
    cqb = max(1, int(math.ceil(math.log2(max(2, action_count)))))
    coin_dim = 1 << cqb
    pqb = max(1, int(math.ceil(math.log2(max(2, n_positions)))))
    pos_dim = 1 << pqb
    legal = (
        tuple(range(action_count)) if legal_basis is None else tuple(int(x) for x in legal_basis)
    )
    coin = np.eye(coin_dim, dtype=np.complex128)
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    for offset in range(0, len(legal) - 1, 2):
        a, b = legal[offset], legal[offset + 1]
        coin[a, a] = inv_sqrt2
        coin[a, b] = inv_sqrt2
        coin[b, a] = inv_sqrt2
        coin[b, b] = -inv_sqrt2
    state = np.zeros((coin_dim, pos_dim), dtype=np.complex128)
    state[:action_count, 0] = np.sqrt(p)
    for _ in range(max(1, int(steps))):
        state = coin @ state
        shifted = np.zeros_like(state)
        for coin_index in range(coin_dim):
            delta = 1 if (coin_index & 1) == 0 else -1
            shifted[coin_index] = np.roll(state[coin_index], delta)
        state = shifted
    out = np.sum(np.abs(state) ** 2, axis=1)[:action_count]
    return _normalize(out)


def _interference_oracle(
    probabilities: Any,
    *,
    phases: Any,
    authority_mask: Any,
    mixer_strength: float,
    mixer_trotter_steps: int,
    action_names: tuple[str, ...],
    **_kwargs: Any,
) -> np.ndarray:
    p = _normalize(probabilities)
    phase = np.asarray(phases, dtype=np.float64).reshape(-1)
    if phase.shape != p.shape:
        raise ValueError("phases must match probabilities")
    authority = np.asarray(authority_mask, dtype=bool).reshape(-1)
    edges = action_family_edges(tuple(action_names))
    unitary = legal_subspace_unitary(
        p.size,
        authority,
        edges,
        strength=float(mixer_strength),
        trotter_steps=int(mixer_trotter_steps),
    )
    amplitude = np.sqrt(p) * np.exp(1j * phase)
    out = np.abs(unitary @ amplitude) ** 2
    out = np.where(authority, out, 0.0)
    return _normalize(out)


def _noise_oracle(
    probabilities: Any, *, depolarizing_probability: float = 0.001, **_kwargs: Any
) -> np.ndarray:
    p = _normalize(probabilities)
    q = float(np.clip(depolarizing_probability, 0.0, 1.0))
    return (1.0 - q) * p + q / float(p.size)


def _qbits(action_count: int, **_kwargs: Any) -> int:
    return max(1, int(math.ceil(math.log2(max(2, int(action_count))))))


def _qbits_deferred(action_count: int, **kwargs: Any) -> int:
    return 2 * _qbits(action_count, **kwargs)


def _qbits_walk(action_count: int, *, n_positions: int = 4, **_kwargs: Any) -> int:
    return _qbits(action_count) + max(1, int(math.ceil(math.log2(max(2, int(n_positions))))))


def _state_bytes(action_count: int, *, precision_bytes: int = 16, **kwargs: Any) -> int:
    return (1 << _qbits(action_count, **kwargs)) * int(precision_bytes)


def _deferred_bytes(action_count: int, *, precision_bytes: int = 16, **kwargs: Any) -> int:
    return (1 << _qbits_deferred(action_count, **kwargs)) * int(precision_bytes)


def _walk_bytes(action_count: int, *, precision_bytes: int = 16, **kwargs: Any) -> int:
    return (1 << _qbits_walk(action_count, **kwargs)) * int(precision_bytes)


def _density_bytes(action_count: int, *, precision_bytes: int = 16, **kwargs: Any) -> int:
    qubits = _qbits(action_count, **kwargs)
    return (1 << (2 * qubits)) * int(precision_bytes)


def circuit_family_structure(circuit: Any) -> dict[str, Any]:
    names = [str(item.operation.name) for item in circuit.data]
    labels = [str(getattr(item.operation, "label", "") or "") for item in circuit.data]
    semantic_names = [label if label else name for name, label in zip(names, labels, strict=True)]
    measurements = [i for i, name in enumerate(names) if name == "measure"]
    control_flow = [
        name for name in names if name in {"if_else", "while_loop", "for_loop", "switch_case"}
    ]
    return {
        "operation_names": names,
        "operation_labels": labels,
        "semantic_operation_names": semantic_names,
        "measurement_indices": measurements,
        "has_mid_circuit_measurement": bool(measurements and min(measurements) < len(names) - 1),
        "has_control_flow": bool(control_flow),
        "control_flow_operations": control_flow,
        "num_qubits": int(circuit.num_qubits),
        "num_clbits": int(circuit.num_clbits),
        "depth": int(circuit.depth()),
    }


def _validate_static(built: RAQICCircuitBuildResult) -> None:
    if built.layout is None or not built.layout.action_qubits:
        raise ValueError("static family lacks action layout")


def _validate_deferred(built: RAQICCircuitBuildResult) -> None:
    structure = circuit_family_structure(built.circuit)
    if "cx" not in structure["operation_names"] or "crz" not in structure["operation_names"]:
        raise ValueError("deferred family requires coherent record copy and controlled feedback")
    if built.layout is None or not built.layout.record_qubits:
        raise ValueError("deferred family lacks record register")


def _validate_dynamic(built: RAQICCircuitBuildResult) -> None:
    structure = circuit_family_structure(built.circuit)
    if not structure["has_mid_circuit_measurement"] or not structure["has_control_flow"]:
        raise ValueError("dynamic family requires mid-circuit measurement and control flow")


def _validate_walk(built: RAQICCircuitBuildResult) -> None:
    structure = circuit_family_structure(built.circuit)
    names = structure["semantic_operation_names"]
    if "conditional_ring_shift" not in names or "legal_coin_operator" not in names:
        raise ValueError(
            "walk family requires authority-preserving coin and conditional shift operations"
        )
    if built.layout is None or not built.layout.position_qubits:
        raise ValueError("walk family lacks position register")


def _validate_interference(built: RAQICCircuitBuildResult) -> None:
    structure = circuit_family_structure(built.circuit)
    if "semantic_interference_mixer" not in structure["semantic_operation_names"]:
        raise ValueError("interference family lacks the semantic mixer unitary")
    if built.recovery_gates.get("unitarity_residual", 1.0) > 1e-10:
        raise ValueError("interference family failed its unitarity audit")
    if not built.metadata.get("action_graph_hash"):
        raise ValueError("interference family lacks action-graph provenance")


def _validate_noise(built: RAQICCircuitBuildResult) -> None:
    if built.metadata.get("noise_model") != "depolarizing":
        raise ValueError("density/noise family lacks an Aer noise model declaration")


def build_density_noise_circuit(
    amplitudes: Any, *, depolarizing_probability: float = 0.001, measure: bool = True
) -> RAQICCircuitBuildResult:
    built = build_static_action_circuit(amplitudes, measure=measure)
    metadata = dict(built.metadata)
    metadata.update(
        {
            "mode": "density_noise",
            "noise_model": "depolarizing",
            "depolarizing_probability": float(depolarizing_probability),
        }
    )
    return RAQICCircuitBuildResult(
        circuit=built.circuit,
        parameter_map=built.parameter_map,
        registers=built.registers,
        metadata=metadata,
        recovery_gates=built.recovery_gates,
        layout=built.layout,
    )


def _static(amplitudes: Any, **kwargs: Any) -> Any:
    return build_static_action_circuit(amplitudes, measure=bool(kwargs.get("measure", False)))


def _deferred(amplitudes: Any, **kwargs: Any) -> Any:
    return build_deferred_control_circuit(
        amplitudes,
        feedback_phases=kwargs.get("feedback_phases"),
        measure=bool(kwargs.get("measure", False)),
    )


def _dynamic(amplitudes: Any, **kwargs: Any) -> Any:
    return build_dynamic_recursive_circuit(
        amplitudes,
        rounds=max(1, int(kwargs.get("rounds", 1))),
        feedback_phases=kwargs.get("feedback_phases"),
    )


def _walk(amplitudes: Any, **kwargs: Any) -> Any:
    return build_quantum_walk_variant(
        amplitudes,
        n_positions=max(2, int(kwargs.get("n_positions", 4))),
        steps=max(1, int(kwargs.get("steps", 2))),
        measure=bool(kwargs.get("measure", True)),
        legal_basis=kwargs.get("legal_basis"),
    )


def _interference(amplitudes: Any, **kwargs: Any) -> Any:
    action_names = tuple(kwargs["action_names"])
    authority = np.asarray(kwargs["authority_mask"], dtype=bool)
    edges = action_family_edges(action_names)
    graph_hash = action_graph_hash(action_names)
    expected_hash = kwargs.get("action_graph_hash")
    if expected_hash is not None and str(expected_hash) != graph_hash:
        raise ValueError("Qiskit interference action graph hash does not match the dense graph")
    unitary = legal_subspace_unitary(
        len(action_names),
        authority,
        edges,
        strength=float(kwargs.get("mixer_strength", 0.0)),
        trotter_steps=max(1, int(kwargs.get("mixer_trotter_steps", 1))),
    )
    return build_interference_action_circuit(
        amplitudes,
        unitary,
        measure=bool(kwargs.get("measure", False)),
        action_graph_hash=graph_hash,
    )


def _density(amplitudes: Any, **kwargs: Any) -> Any:
    return build_density_noise_circuit(
        amplitudes,
        depolarizing_probability=float(kwargs.get("depolarizing_probability", 0.001)),
        measure=bool(kwargs.get("measure", True)),
    )


CIRCUIT_FAMILIES: dict[str, CircuitFamilySpec] = {
    "static": CircuitFamilySpec(
        "static",
        _static,
        _static_oracle,
        True,
        False,
        False,
        False,
        ("statevector", "density_matrix", "tensor_network"),
        "action",
        "c_action",
        _qbits,
        _state_bytes,
        _validate_static,
    ),
    "deferred": CircuitFamilySpec(
        "deferred",
        _deferred,
        _static_oracle,
        True,
        False,
        False,
        False,
        ("statevector", "density_matrix", "tensor_network"),
        "action",
        "c_action",
        _qbits_deferred,
        _deferred_bytes,
        _validate_deferred,
    ),
    "dynamic_recursive": CircuitFamilySpec(
        "dynamic_recursive",
        _dynamic,
        _static_oracle,
        False,
        True,
        True,
        False,
        ("statevector", "density_matrix", "tensor_network"),
        "action",
        "c_action",
        _qbits,
        _state_bytes,
        _validate_dynamic,
    ),
    "walk": CircuitFamilySpec(
        "walk",
        _walk,
        _walk_oracle,
        True,
        False,
        False,
        False,
        ("statevector", "density_matrix", "tensor_network"),
        "coin",
        "c_action",
        _qbits_walk,
        _walk_bytes,
        _validate_walk,
    ),
    "interference": CircuitFamilySpec(
        "interference",
        _interference,
        _interference_oracle,
        True,
        False,
        False,
        False,
        ("statevector", "density_matrix", "tensor_network"),
        "action",
        "c_action",
        _qbits,
        _state_bytes,
        _validate_interference,
    ),
    "density_noise": CircuitFamilySpec(
        "density_noise",
        _density,
        _noise_oracle,
        False,
        True,
        False,
        True,
        ("density_matrix", "statevector"),
        "action",
        "c_action",
        _qbits,
        _density_bytes,
        _validate_noise,
    ),
}


def build_circuit_family(
    family: str, probabilities: Any, phases: Any | None = None, **kwargs: Any
) -> RAQICCircuitBuildResult:
    if family not in CIRCUIT_FAMILIES:
        raise ValueError(f"unknown RAQIC circuit family: {family}")
    p = _normalize(probabilities)
    phase = np.zeros_like(p) if phases is None else np.asarray(phases, dtype=np.float64)
    if phase.shape != p.shape:
        raise ValueError("phase and probability rows must have equal shapes")
    amplitudes = np.sqrt(p) * np.exp(1j * phase)
    if family == "interference":
        kwargs = {**kwargs, "phases": phase}
    built = CIRCUIT_FAMILIES[family].builder(amplitudes, **kwargs)
    CIRCUIT_FAMILIES[family].structural_validator(built)
    return built
