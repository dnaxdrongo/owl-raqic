from __future__ import annotations

from typing import Any


def statevector_memory_bytes(n_qubits: int, complex_bytes: int = 16) -> int:
    return (1 << n_qubits) * complex_bytes


def density_matrix_memory_bytes(n_qubits: int, complex_bytes: int = 16) -> int:
    return (1 << (2 * n_qubits)) * complex_bytes


def ensure_memory_within_limit(
    n_qubits: int, method: str, limit_mb: float, allow_override: bool = False
) -> dict[str, Any]:
    if method == "density_matrix":
        need = density_matrix_memory_bytes(n_qubits)
    else:
        need = statevector_memory_bytes(n_qubits)
    limit = int(limit_mb * 1024 * 1024)
    ok = need <= limit or allow_override
    if not ok:
        raise MemoryError(
            f"estimated {method} memory {need / 1024 / 1024:.2f} MB exceeds limit {limit_mb:.2f} MB"
        )
    return {
        "n_qubits": n_qubits,
        "method": method,
        "estimated_bytes": need,
        "limit_mb": limit_mb,
        "allowed": ok,
    }
