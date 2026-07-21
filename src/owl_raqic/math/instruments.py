from __future__ import annotations

from typing import Any, cast

import numpy as np
from scipy.linalg import expm

from .intentions import apply_top_down_bias, stable_softmax


def action_amplitudes(
    scores: np.ndarray,
    phases: np.ndarray | None = None,
    intention: np.ndarray | None = None,
    beta_intention: float = 1.0,
    temperature: float = 1.0,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=float)
    biased = apply_top_down_bias(scores, intention, beta=beta_intention)
    probs = stable_softmax(biased, temperature=temperature, mask=mask)
    phases = (
        np.zeros_like(scores, dtype=float) if phases is None else np.asarray(phases, dtype=float)
    )
    if phases.shape != scores.shape:
        raise ValueError("phases shape must match scores")
    amps = np.sqrt(probs) * np.exp(1j * phases)
    amps = amps / np.linalg.norm(amps)
    return amps, np.abs(amps) ** 2


def householder_unitary_from_state(target: np.ndarray) -> np.ndarray:
    target = np.asarray(target, dtype=complex)
    n = target.size
    norm = np.linalg.norm(target)
    if norm == 0:
        raise ValueError("target cannot be zero")
    psi = target / norm
    e0 = np.zeros(n, dtype=complex)
    e0[0] = 1.0
    if np.linalg.norm(psi - e0) < 1e-14:
        return np.eye(n, dtype=complex)
    # Build a numerically stable orthonormal basis anchored at the target state.
    rng_basis = np.eye(n, dtype=complex)
    rng_basis[:, 0] = psi
    Q, R = np.linalg.qr(rng_basis)
    phase = np.vdot(Q[:, 0], psi)
    if abs(phase) > 1e-14:
        Q[:, 0] *= phase / abs(phase)
    # If QR degeneracy caused mismatch, use Gram-Schmidt manually.
    if np.linalg.norm(Q[:, 0] - psi) > 1e-8:
        cols = [psi]
        for j in range(n):
            v = np.eye(n, dtype=complex)[:, j]
            for c in cols:
                v = v - c * np.vdot(c, v)
            if np.linalg.norm(v) > 1e-10:
                cols.append(v / np.linalg.norm(v))
            if len(cols) == n:
                break
        Q = np.column_stack(cols)
    return Q


def projector_partition(dim: int) -> list[np.ndarray]:
    return [
        np.diag([1.0 if i == j else 0.0 for i in range(dim)]).astype(complex) for j in range(dim)
    ]


def preparation_kraus_from_amplitudes(
    amplitudes: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray, list[np.ndarray]]:
    U = householder_unitary_from_state(amplitudes)
    projectors = projector_partition(len(amplitudes))
    kraus = [P @ U for P in projectors]
    return kraus, U, projectors


def outcome_probabilities_from_kraus(kraus_ops: list[np.ndarray], rho: np.ndarray) -> np.ndarray:
    return np.array([np.trace(K @ rho @ K.conjugate().T).real for K in kraus_ops], dtype=float)


def post_measurement_state(K: np.ndarray, rho: np.ndarray, prob: float | None = None) -> np.ndarray:
    num = K @ rho @ K.conjugate().T
    p = float(np.trace(num).real) if prob is None else prob
    if p <= 0:
        raise ValueError("cannot normalize zero-probability branch")
    return cast(np.ndarray, num / p)


def feedback_unitaries(dim: int, phase_scale: float = 0.1) -> list[np.ndarray]:
    outs = []
    for y in range(dim):
        phases = np.array([phase_scale * (y + 1) * (j + 1) for j in range(dim)])
        H = np.diag(phases)
        outs.append(expm(-1j * H))
    return outs


def recursive_channel(
    kraus_ops: list[np.ndarray], feedback_ops: list[np.ndarray], rho: np.ndarray
) -> np.ndarray:
    if len(kraus_ops) != len(feedback_ops):
        raise ValueError("kraus and feedback lists must have same length")
    out = np.zeros_like(rho, dtype=complex)
    for K, U in zip(kraus_ops, feedback_ops, strict=True):
        out += U @ K @ rho @ K.conjugate().T @ U.conjugate().T
    return out


def simulate_recursive_ensemble(
    amplitudes: np.ndarray, rho0: np.ndarray | None = None, rounds: int = 1
) -> dict[str, Any]:
    kraus, Uprep, projectors = preparation_kraus_from_amplitudes(amplitudes)
    feedback = feedback_unitaries(len(amplitudes))
    rho = np.zeros((len(amplitudes), len(amplitudes)), dtype=complex)
    rho[0, 0] = 1.0
    if rho0 is not None:
        rho = np.asarray(rho0, dtype=complex)
    traces = []
    min_eigs = []
    probs_by_round = []
    for _ in range(rounds):
        probs = outcome_probabilities_from_kraus(kraus, rho)
        probs_by_round.append(probs)
        rho = recursive_channel(kraus, feedback, rho)
        rho = (rho + rho.conjugate().T) / 2
        traces.append(np.trace(rho).real)
        min_eigs.append(np.min(np.linalg.eigvalsh(rho)).real)
    return {
        "rho": rho,
        "traces": np.array(traces),
        "min_eigenvalues": np.array(min_eigs),
        "probabilities": probs_by_round,
    }
