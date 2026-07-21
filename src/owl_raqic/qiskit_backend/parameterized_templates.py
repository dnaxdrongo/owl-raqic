"""Build runtime-parameterized Qiskit templates.

Static-family runtime binding uses the project-owned native rotation-tree
state preparation in :mod:`native_state_preparation` and does not construct
``RawFeatureVector`` objects.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from .native_state_preparation import (
    CompiledRuntimeBindingTemplate,
    NativePreparationLayout,
    make_native_preparation_layout,
    make_parameter_bind_batch,
    probabilities_and_phases_to_amplitudes,
    transpile_native_template,
)

RUNTIME_PARAMETERIZED_FAMILIES = frozenset({"static"})


def supports_runtime_parameter_binding(family: str) -> bool:
    """Return structural eligibility; executable support requires preflight."""

    return str(family) in RUNTIME_PARAMETERIZED_FAMILIES


# These aliases forward existing callers and tests to the native implementation.
# RawFeatureVector implementation from the execution graph.
ParameterizedCircuitTemplate = CompiledRuntimeBindingTemplate


def build_native_feature_template(
    action_count: int,
    *,
    simulator: Any,
    family: str = "static",
    measure: bool = False,
    method: str = "statevector",
    device: str = "GPU",
    precision: str = "double",
) -> CompiledRuntimeBindingTemplate:
    if not supports_runtime_parameter_binding(family):
        raise ValueError(
            f"runtime parameter binding is not certified for circuit family {family!r}"
        )
    layout = make_native_preparation_layout(action_count)
    return transpile_native_template(
        layout,
        simulator,
        method=method,
        device=device,
        precision=precision,
        measure=measure,
    )


def build_raw_feature_template(
    action_count: int,
    *,
    family: str = "static",
    measure: bool = False,
    simulator: Any | None = None,
    method: str = "statevector",
    device: str = "CPU",
    precision: str = "double",
) -> CompiledRuntimeBindingTemplate:
    """Forward compatible calls to the native implementation.

        
    """

    if simulator is None:
        from qiskit_aer import AerSimulator

        simulator = AerSimulator(
            method=str(method),
            device=str(device).upper(),
            precision=str(precision),
            runtime_parameter_bind_enable=True,
        )
    return build_native_feature_template(
        action_count,
        simulator=simulator,
        family=family,
        measure=measure,
        method=method,
        device=device,
        precision=precision,
    )


def amplitude_bindings(
    template: CompiledRuntimeBindingTemplate,
    probabilities: Any,
    phases: Any,
) -> tuple[dict[Any, list[float]], np.ndarray]:
    return make_parameter_bind_batch(template, probabilities, phases)


def statevector_action_probabilities(
    statevector: Any,
    *,
    action_qubits: tuple[int, ...],
    action_count: int,
) -> np.ndarray:
    """Marginalize an arbitrary statevector onto the action register."""

    state = np.asarray(statevector, dtype=np.complex128).reshape(-1)
    probabilities = np.abs(state) ** 2
    out: np.ndarray = np.zeros((int(action_count),), dtype=np.float64)
    for basis, probability in enumerate(probabilities):
        action = 0
        for output_bit, qubit in enumerate(action_qubits):
            action |= ((basis >> qubit) & 1) << output_bit
        if action < action_count:
            out[action] += float(probability)
    total = out.sum()
    if not np.isfinite(total) or total <= 0:
        raise FloatingPointError("statevector action marginal is not normalized")
    return cast(np.ndarray, out / total)


__all__ = [
    "NativePreparationLayout",
    "ParameterizedCircuitTemplate",
    "RUNTIME_PARAMETERIZED_FAMILIES",
    "amplitude_bindings",
    "build_native_feature_template",
    "build_raw_feature_template",
    "make_native_preparation_layout",
    "probabilities_and_phases_to_amplitudes",
    "statevector_action_probabilities",
    "supports_runtime_parameter_binding",
]
