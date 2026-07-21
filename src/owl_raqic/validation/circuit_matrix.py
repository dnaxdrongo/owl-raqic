from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from owl_raqic.gpu.instrument_checks import explicit_instrument_audit
from owl_raqic.qiskit_backend.circuit_families import CIRCUIT_FAMILIES
from owl_raqic.qiskit_backend.per_ow_executor import PerOWQiskitExecutor
from owl_raqic.qiskit_backend.qiskit_policy import (
    QiskitDecisionMode,
    QiskitExecutionPolicy,
)
from owl_raqic.validation.statistics import kl_divergence, shot_validation_pass, total_variation


@dataclass
class CircuitValidationRow:
    row: int
    family: str
    max_abs_error: float
    kl_divergence: float
    total_variation: float
    qiskit_used_gpu: bool
    passed: bool
    dense: list[float]
    qiskit: list[float]
    metadata: dict[str, Any]


@dataclass
class CircuitValidationReport:
    rows: list[CircuitValidationRow]
    method: str
    device: str
    tolerance: float

    @property
    def passed(self) -> bool:
        return bool(self.rows) and all(row.passed for row in self.rows)

    def to_dict(self) -> Any:
        return {
            "passed": self.passed,
            "method": self.method,
            "device": self.device,
            "tolerance": self.tolerance,
            "max_abs_error": max((r.max_abs_error for r in self.rows), default=0.0),
            "rows": [asdict(r) for r in self.rows],
        }

    def write(self, path: str | Path) -> Any:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return path


def validate_circuit_matrix(
    probabilities: Any,
    phases: Any | None = None,
    authority_mask: Any | None = None,
    parent_intention: Any | None = None,
    *,
    ow_ids: Any | None = None,
    families: tuple[str, ...] = ("static",),
    authoritative_family: str = "static",
    limit: int = 16,
    method: str = "statevector",
    strict_gpu: bool = True,
    allow_cpu_fallback: bool = False,
    tolerance: float = 1e-8,
    kl_tolerance: float = 1e-7,
    shots: int = 4096,
    simulator_options: dict[str, Any] | None = None,
    expected_probabilities: Any | None = None,
    interference_mixer_strength: float = 0.0,
    interference_trotter_steps: int = 1,
    action_names: tuple[str, ...] = (),
) -> CircuitValidationReport:
    """Validate distinct circuit families on an already-selected bounded slab."""

    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError("probabilities must have shape [N,A]")
    if p.shape[0] > int(limit):
        p = p[: int(limit)]
    expected = (
        p.copy()
        if expected_probabilities is None
        else np.asarray(expected_probabilities, dtype=np.float64)[: p.shape[0]]
    )
    if expected.shape != p.shape:
        raise ValueError("expected_probabilities must match probabilities")
    phase = (
        np.zeros_like(p) if phases is None else np.asarray(phases, dtype=np.float64)[: p.shape[0]]
    )
    authority = (
        np.ones_like(p, dtype=bool)
        if authority_mask is None
        else np.asarray(authority_mask, dtype=bool)[: p.shape[0]]
    )
    ids = (
        np.arange(p.shape[0], dtype=np.int64)
        if ow_ids is None
        else np.asarray(ow_ids, dtype=np.int64)[: p.shape[0]]
    )
    if allow_cpu_fallback and strict_gpu:
        # The production validator never labels fallback as GPU validation.
        # Static-template compatibility is handled by the dedicated helper.
        allow_cpu_fallback = False

    policy = QiskitExecutionPolicy(
        mode=QiskitDecisionMode.EVERY_OW_CIRCUIT_FAMILY,
        circuit_families=tuple(families),
        authoritative_family=authoritative_family,
        method=method,
        shots=int(shots),
        chunk_size=max(1, min(64, p.shape[0] or 1)),
        strict_gpu=strict_gpu,
        device="GPU" if strict_gpu else "CPU",
        runtime_parameter_binding=bool(
            (simulator_options or {}).get("runtime_parameter_bind_enable", False)
        ),
        batched_shots_gpu=bool((simulator_options or {}).get("batched_shots_gpu", False)),
        shot_branching=bool((simulator_options or {}).get("shot_branching_enable", False)),
        confirm_expensive=True,
        interference_mixer_strength=float(interference_mixer_strength),
        interference_trotter_steps=int(interference_trotter_steps),
        action_names=tuple(action_names),
    )
    executor = PerOWQiskitExecutor(policy)
    result = executor.execute(
        p,
        phase,
        authority,
        ids,
        tick=0,
        tolerance=tolerance,
    )
    rows: list[CircuitValidationRow] = []
    for family, family_result in result.families.items():
        target_rows = expected if family == "interference" else p
        for row_index, (dense, qiskit) in enumerate(
            zip(target_rows, family_result.probabilities, strict=True)
        ):
            dense = dense / max(float(dense.sum()), 1e-15)
            error = float(np.max(np.abs(dense - qiskit)))
            spec_shot_based = bool(CIRCUIT_FAMILIES[family].shot_based)
            statistical = (
                shot_validation_pass(
                    dense,
                    qiskit,
                    shots,
                    alpha=0.01,
                    max_tv=max(0.02, 4.0 * tolerance),
                )
                if spec_shot_based
                else None
            )
            rows.append(
                CircuitValidationRow(
                    row=int(row_index),
                    family=family,
                    max_abs_error=error,
                    kl_divergence=kl_divergence(dense, qiskit),
                    total_variation=total_variation(dense, qiskit),
                    qiskit_used_gpu=bool(
                        family_result.metadata.get("gpu_execution_verified", False)
                    ),
                    passed=bool(
                        (
                            statistical.passed
                            and kl_divergence(dense, qiskit)
                            <= max(
                                kl_tolerance,
                                statistical.allowance,
                            )
                        )
                        if statistical is not None
                        else (error <= tolerance and kl_divergence(dense, qiskit) <= kl_tolerance)
                    ),
                    dense=dense.tolist(),
                    qiskit=qiskit.tolist(),
                    metadata={
                        "execution": family_result.metadata,
                        "statistical_validation": (
                            None
                            if statistical is None
                            else {
                                "passed": statistical.passed,
                                "allowance": statistical.allowance,
                                "shots": statistical.shots,
                                "alpha": statistical.alpha,
                            }
                        ),
                        "instrument": explicit_instrument_audit(
                            p[row_index : row_index + 1],
                            phase[row_index : row_index + 1],
                            limit=1,
                            tol=tolerance,
                        ),
                    },
                )
            )
    requested_device = str(policy.device).upper()
    return CircuitValidationReport(rows, method, requested_device, tolerance)
