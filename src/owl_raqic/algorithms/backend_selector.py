from __future__ import annotations

from owl_raqic.config import BackendProfile
from owl_raqic.qiskit_backend.backend_profiles import qiskit_aer_available, qiskit_available


def select_backend(policy: str = "auto") -> BackendProfile:
    if policy == "cpu_audit":
        return BackendProfile(
            name="cpu_audit", method="cpu_audit", device="NONE", qiskit_required=False
        )
    if policy in ("statevector", "auto"):
        if qiskit_available() and qiskit_aer_available():
            return BackendProfile(
                name="cpu_statevector", method="statevector", device="CPU", qiskit_required=True
            )
        return BackendProfile(
            name="cpu_audit", method="cpu_audit", device="NONE", qiskit_required=False
        )
    if policy == "density_matrix":
        if qiskit_available() and qiskit_aer_available():
            return BackendProfile(
                name="cpu_density_matrix",
                method="density_matrix",
                device="CPU",
                qiskit_required=True,
            )
        return BackendProfile(
            name="cpu_audit", method="cpu_audit", device="NONE", qiskit_required=False
        )
    if policy == "gpu":
        return BackendProfile(
            name="gpu_statevector",
            method="statevector",
            device="GPU",
            optional=True,
            qiskit_required=True,
            gpu_required=True,
        )
    raise ValueError(f"unknown backend policy: {policy}")
