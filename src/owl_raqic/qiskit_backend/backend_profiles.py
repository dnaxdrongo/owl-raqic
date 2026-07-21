from __future__ import annotations

import importlib.util
from typing import Any


def qiskit_available() -> bool:
    return importlib.util.find_spec("qiskit") is not None


def qiskit_aer_available() -> bool:
    return importlib.util.find_spec("qiskit_aer") is not None


def gpu_aer_available() -> bool:
    return (
        importlib.util.find_spec("qiskit_aer") is not None
        and importlib.util.find_spec("cupy") is not None
    )


class MissingQiskitError(ImportError):
    pass


def require_qiskit() -> Any:
    if not qiskit_available():
        raise MissingQiskitError(
            "Qiskit is not installed. Install with `pip install owl-raqic[qiskit]`."
        )


def qiskit_aer_gpu_runtime_available(method: str = "statevector") -> bool:
    try:
        from owl_raqic.qiskit_backend.gpu_execution import aer_gpu_available_runtime

        return bool(aer_gpu_available_runtime(method=method).available)
    except Exception:
        return False
