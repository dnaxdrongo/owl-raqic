from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from owl_raqic.qiskit_backend.aer_evidence import parse_aer_gpu_evidence
from owl_raqic.qiskit_backend.aer_runtime import run_aer_job
from owl_raqic.qiskit_backend.backend_profiles import qiskit_aer_available, require_qiskit
from owl_raqic.qiskit_backend.execution import statevector_probabilities_from_circuit
from owl_raqic.qiskit_backend.parameterized_templates import statevector_action_probabilities


@dataclass(frozen=True)
class QiskitGPUInfo:
    available: bool
    method: str = "statevector"
    device: str = "GPU"
    metadata: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QiskitProbabilityResult:
    probabilities: np.ndarray
    backend: str
    method: str
    device: str
    used_gpu: bool
    used_cpu_fallback: bool
    metadata: dict[str, Any]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["probabilities"] = np.asarray(self.probabilities, dtype=float).tolist()
        return out


def build_aer_simulator_from_config(
    method: str = "statevector",
    device: str = "GPU",
    *,
    batched_shots_gpu: bool | None = None,
    cuStateVec_enable: bool | None = None,
    **options: Any,
) -> Any:
    """Construct an Aer simulator without importing Qiskit on CPU-only imports.

    Options are passed only when explicitly requested.  This is important across
    Aer versions because unsupported backend options should fail clearly rather
    than be silently ignored.
    """
    require_qiskit()
    if not qiskit_aer_available():
        raise ImportError("qiskit-aer or qiskit-aer-gpu is not installed")
    from qiskit_aer import AerSimulator

    probe = AerSimulator()
    devices = tuple(str(item).upper() for item in probe.available_devices())
    methods = tuple(str(item) for item in probe.available_methods())
    if str(device).upper() not in devices:
        raise RuntimeError(f"Aer device {device!r} is unavailable; available_devices={devices}")
    if str(method) not in methods and str(method) != "automatic":
        raise RuntimeError(f"Aer method {method!r} is unavailable; available_methods={methods}")

    opts = dict(options)
    opts.setdefault("method", method)
    opts.setdefault("device", device)
    if batched_shots_gpu is not None:
        opts["batched_shots_gpu"] = bool(batched_shots_gpu)
    if cuStateVec_enable is not None:
        opts["cuStateVec_enable"] = bool(cuStateVec_enable)
    return AerSimulator(**opts)


def _result_metadata(result: Any) -> dict[str, Any]:
    try:
        rows = getattr(result, "results", None) or []
        if rows:
            return dict(getattr(rows[0], "metadata", {}) or {})
    except Exception:
        pass
    return {}


def aer_gpu_available_runtime(method: str = "statevector", **options: Any) -> QiskitGPUInfo:
    """Run a real one-qubit GPU job; import success alone is insufficient."""
    try:
        from qiskit import QuantumCircuit

        sim = build_aer_simulator_from_config(method=method, device="GPU", **options)
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.save_statevector()
        from qiskit import transpile

        compiled = transpile(qc, backend=sim, optimization_level=0)
        result = run_aer_job(sim, compiled)
        if not result.success:
            raise RuntimeError(getattr(result, "status", "Aer GPU smoke test failed"))
        metadata = _result_metadata(result)
        evidence = parse_aer_gpu_evidence(metadata)
        if not evidence["verified"]:
            raise RuntimeError("Aer GPU smoke test completed without positive GPU result metadata")
        return QiskitGPUInfo(
            available=True,
            method=method,
            device="GPU",
            metadata={"result": metadata, "gpu_evidence": evidence},
        )
    except Exception as exc:
        return QiskitGPUInfo(
            available=False,
            method=method,
            device="GPU",
            error=f"{type(exc).__name__}: {exc}",
        )


def _cpu_action_probabilities(circuit: Any, n_actions: Any, action_qubits: Any) -> Any:
    if action_qubits is None:
        if n_actions is not None:
            required = max(1, int(np.ceil(np.log2(max(2, int(n_actions))))))
            if int(circuit.num_qubits) > required:
                raise ValueError(
                    "action_qubits must be declared when a circuit contains non-action qubits"
                )
        return statevector_probabilities_from_circuit(circuit, n_actions=n_actions)
    from qiskit.quantum_info import Statevector

    circ = circuit.remove_final_measurements(inplace=False)
    state = Statevector.from_instruction(circ).data
    if n_actions is None:
        n_actions = 1 << len(action_qubits)
    return statevector_action_probabilities(
        state, action_qubits=tuple(action_qubits), action_count=int(n_actions)
    )


def run_statevector_probabilities_gpu(
    circuit: Any,
    n_actions: int | None = None,
    method: str = "statevector",
    device: str = "GPU",
    *,
    action_qubits: tuple[int, ...] | None = None,
    strict_gpu: bool = True,
    allow_cpu_fallback: bool = False,
    **options: Any,
) -> QiskitProbabilityResult:
    """Execute a circuit and return probabilities plus auditable backend metadata.

    A CPU fallback is never represented as GPU validation.  In strict mode any
    GPU error raises immediately.  Compatibility callers may explicitly enable
    CPU fallback, in which case ``used_cpu_fallback`` is true.
    """
    requested_gpu = device.upper() == "GPU"
    if not requested_gpu:
        probs = _cpu_action_probabilities(circuit, n_actions, action_qubits)
        return QiskitProbabilityResult(
            probabilities=np.asarray(probs, dtype=float),
            backend="qiskit.quantum_info.Statevector",
            method=method,
            device="CPU",
            used_gpu=False,
            used_cpu_fallback=False,
            metadata={},
        )

    try:
        sim = build_aer_simulator_from_config(method=method, device="GPU", **options)
        from qiskit import transpile

        circ = circuit.remove_final_measurements(inplace=False).copy()
        circ.save_statevector()
        compiled = transpile(circ, backend=sim, optimization_level=0)
        result = run_aer_job(sim, compiled)
        if not result.success:
            raise RuntimeError(getattr(result, "status", "Aer GPU simulation failed"))
        sv = np.asarray(result.get_statevector(compiled), dtype=np.complex128)
        if action_qubits is not None:
            if n_actions is None:
                n_actions = 1 << len(action_qubits)
            probs = statevector_action_probabilities(
                sv, action_qubits=tuple(action_qubits), action_count=int(n_actions)
            )
        else:
            probabilities = np.abs(sv) ** 2
            if n_actions is not None:
                required = max(1, int(np.ceil(np.log2(max(2, int(n_actions))))))
                if int(circuit.num_qubits) > required:
                    raise ValueError(
                        "action_qubits must be declared when a circuit contains non-action qubits"
                    )
                probabilities = probabilities[: int(n_actions)]
            total = float(np.sum(probabilities))
            if not np.isfinite(total) or total <= 0:
                raise FloatingPointError(f"invalid Aer probability normalization: {total}")
            probs = np.asarray(probabilities / total, dtype=float)
        metadata = _result_metadata(result)
        evidence = parse_aer_gpu_evidence(metadata)
        if not evidence["verified"]:
            raise RuntimeError("Aer job completed without positive GPU execution metadata")
        return QiskitProbabilityResult(
            probabilities=np.asarray(probs, dtype=float),
            backend="qiskit_aer.AerSimulator",
            method=method,
            device="GPU",
            used_gpu=True,
            used_cpu_fallback=False,
            metadata={"result": metadata, "gpu_evidence": evidence},
        )
    except Exception as exc:
        if strict_gpu or not allow_cpu_fallback:
            raise RuntimeError(
                f"strict Qiskit-Aer-GPU execution failed ({method}): {type(exc).__name__}: {exc}"
            ) from exc
        probs = _cpu_action_probabilities(circuit, n_actions, action_qubits)
        return QiskitProbabilityResult(
            probabilities=np.asarray(probs, dtype=float),
            backend="qiskit.quantum_info.Statevector",
            method=method,
            device="CPU",
            used_gpu=False,
            used_cpu_fallback=True,
            metadata={},
            error=f"{type(exc).__name__}: {exc}",
        )


def statevector_probabilities_gpu(
    circuit: Any,
    n_actions: int | None = None,
    method: str = "statevector",
    device: str = "GPU",
    *,
    action_qubits: tuple[int, ...] | None = None,
    strict_gpu: bool = True,
    allow_cpu_fallback: bool = False,
    return_metadata: bool = False,
    **options: Any,
) -> Any:
    """Backward-compatible probability API with strict, explicit fallback."""
    result = run_statevector_probabilities_gpu(
        circuit,
        n_actions=n_actions,
        method=method,
        device=device,
        action_qubits=action_qubits,
        strict_gpu=strict_gpu,
        allow_cpu_fallback=allow_cpu_fallback,
        **options,
    )
    return result if return_metadata else result.probabilities


def validate_dense_against_qiskit(
    circuit: Any,
    dense_probabilities: np.ndarray,
    n_actions: int,
    method: str = "statevector",
    device: str = "GPU",
    tol: float = 1e-8,
    *,
    action_qubits: tuple[int, ...] | None = None,
    strict_gpu: bool = True,
    allow_cpu_fallback: bool = False,
    **options: Any,
) -> dict[str, Any]:
    result = run_statevector_probabilities_gpu(
        circuit,
        n_actions=n_actions,
        method=method,
        device=device,
        action_qubits=action_qubits,
        strict_gpu=strict_gpu,
        allow_cpu_fallback=allow_cpu_fallback,
        **options,
    )
    q = np.asarray(result.probabilities, dtype=float)
    d = np.asarray(dense_probabilities, dtype=float)[:n_actions]
    d_total = float(d.sum())
    if not np.isfinite(d_total) or d_total <= 0:
        raise ValueError("dense probabilities have invalid normalization")
    d = d / d_total
    err = float(np.max(np.abs(q - d)))
    eps = 1e-15
    kl = float(np.sum(d * np.log(np.maximum(d, eps) / np.maximum(q, eps))))
    tv = float(0.5 * np.sum(np.abs(q - d)))
    return {
        "passed": bool(err <= tol and (result.used_gpu or device.upper() == "CPU")),
        "max_abs_error": err,
        "kl_divergence": kl,
        "total_variation": tv,
        "qiskit": q.tolist(),
        "dense": d.tolist(),
        "execution": result.to_dict(),
    }
