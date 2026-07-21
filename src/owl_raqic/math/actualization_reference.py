"""Provide explicit reference calculations for actualization validation.

These helpers intentionally use NumPy matrices. They are suitable for unit,
SymPy, and Qiskit validation on bounded action spaces and are never imported by
the all-OW GPU hot path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from owl_raqic.math.action_graph import legal_subspace_unitary


@dataclass(frozen=True)
class ReferenceInterferenceResult:
    input_amplitudes: np.ndarray
    unitary: np.ndarray
    output_amplitudes: np.ndarray
    probabilities: np.ndarray


def amplitudes_from_probability_phase(
    probabilities: Any,
    phases: Any,
    authority_mask: Any | None = None,
) -> np.ndarray:
    """Build normalized complex amplitudes from action probabilities/phases."""
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    phi = np.asarray(phases, dtype=np.float64).reshape(-1)
    if p.shape != phi.shape:
        raise ValueError("probabilities and phases must have the same one-dimensional shape")
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(phi)):
        raise ValueError("probabilities and phases must be finite")
    if authority_mask is not None:
        legal = np.asarray(authority_mask, dtype=bool).reshape(-1)
        if legal.shape != p.shape:
            raise ValueError("authority_mask must match probabilities")
        p = np.where(legal, p, 0.0)
    if np.any(p < 0.0):
        raise ValueError("probabilities must be nonnegative")
    total = float(np.sum(p, dtype=np.float64))
    if total <= 0.0:
        raise ValueError("probability mass must be positive")
    p = p / total
    return np.asarray(np.sqrt(p) * np.exp(1j * phi), dtype=np.complex128)


def explicit_interference_reference(
    probabilities: Any,
    phases: Any,
    authority_mask: Any,
    edges: tuple[tuple[int, int], ...],
    *,
    strength: float,
    trotter_steps: int,
) -> ReferenceInterferenceResult:
    """Apply the exact legal-subspace unitary to one action row."""
    amplitudes = amplitudes_from_probability_phase(
        probabilities,
        phases,
        authority_mask,
    ).astype(np.complex128, copy=False)
    unitary = legal_subspace_unitary(
        amplitudes.size,
        authority_mask,
        edges,
        strength=float(strength),
        trotter_steps=int(trotter_steps),
    )
    output = unitary @ amplitudes
    legal = np.asarray(authority_mask, dtype=bool).reshape(-1)
    final = np.where(legal, np.abs(output) ** 2, 0.0)
    final /= max(float(np.sum(final, dtype=np.float64)), np.finfo(np.float64).tiny)
    return ReferenceInterferenceResult(
        input_amplitudes=amplitudes,
        unitary=unitary,
        output_amplitudes=output,
        probabilities=final,
    )


def two_action_probability_law(
    p_left: float,
    p_right: float,
    phase_left: float,
    phase_right: float,
    angle: float,
) -> tuple[float, float]:
    """Closed-form probabilities for one ``exp(-i angle sigma_x)`` pair."""
    p1 = float(p_left)
    p2 = float(p_right)
    if p1 < 0.0 or p2 < 0.0:
        raise ValueError("two-action probabilities must be nonnegative")
    c = float(np.cos(float(angle)))
    s = float(np.sin(float(angle)))
    interference = 2.0 * c * s * np.sqrt(p1 * p2) * np.sin(float(phase_right) - float(phase_left))
    return (
        c * c * p1 + s * s * p2 + interference,
        s * s * p1 + c * c * p2 - interference,
    )


def unitary_residual(unitary: Any) -> float:
    """Return the maximum absolute ``U†U-I`` residual."""
    matrix = np.asarray(unitary, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("unitary must be a square matrix")
    identity = np.eye(matrix.shape[0], dtype=np.complex128)
    return float(np.max(np.abs(matrix.conj().T @ matrix - identity)))
