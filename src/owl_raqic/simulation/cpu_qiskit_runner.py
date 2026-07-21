from __future__ import annotations

from owl_raqic.qiskit_backend.backend_profiles import qiskit_available


def available() -> bool:
    return qiskit_available()
