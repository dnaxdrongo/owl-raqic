from __future__ import annotations

from owl_raqic.qiskit_backend.backend_profiles import gpu_aer_available


def available() -> bool:
    return gpu_aer_available()
