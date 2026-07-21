from __future__ import annotations

from fractions import Fraction
from typing import Any

import numpy as np

from owl_raqic.adelic.padic import product_formula_rational
from owl_raqic.adelic.phases import diagonal_character_cancels


def check_state_normalization(state: Any, tol: float = 1e-10) -> bool:
    state = np.asarray(state, dtype=complex)
    return bool(abs(np.vdot(state, state) - 1.0) <= tol)


def check_density_matrix(rho: Any, tol: float = 1e-10) -> dict[str, Any]:
    rho = np.asarray(rho, dtype=complex)
    herm = np.linalg.norm(rho - rho.conjugate().T)
    tr = np.trace(rho)
    evals = np.linalg.eigvalsh((rho + rho.conjugate().T) / 2)
    return {
        "hermitian": bool(herm <= tol),
        "trace_one": bool(abs(tr - 1.0) <= tol),
        "positive": bool(np.min(evals) >= -tol),
        "min_eigenvalue": float(np.min(evals).real),
        "trace_real": float(np.real(tr)),
        "trace_imag": float(np.imag(tr)),
        "hermitian_residual": float(herm),
        "passed": bool(herm <= tol and abs(tr - 1.0) <= tol and np.min(evals) >= -tol),
    }


def check_unitarity(U: Any, tol: float = 1e-10) -> dict[str, Any]:
    U = np.asarray(U, dtype=complex)
    intention = np.eye(U.shape[0], dtype=complex)
    res = np.linalg.norm(U.conjugate().T @ U - intention)
    return {"unitary": bool(res <= tol), "residual": float(res)}


def check_projector_partition(projectors: Any, tol: float = 1e-10) -> dict[str, Any]:
    matrices = [np.asarray(P, dtype=complex) for P in projectors]
    if not matrices:
        raise ValueError("projectors must not be empty")
    Psum = np.sum(np.stack(matrices, axis=0), axis=0)
    dim = Psum.shape[0]
    res_sum = np.linalg.norm(Psum - np.eye(dim))
    res_idem = max(float(np.linalg.norm(P @ P - P)) for P in matrices)
    res_herm = max(float(np.linalg.norm(P.conjugate().T - P)) for P in matrices)
    return {
        "partition": bool(res_sum <= tol and res_idem <= tol and res_herm <= tol),
        "sum_residual": float(res_sum),
        "idempotence_residual": float(res_idem),
        "hermitian_residual": float(res_herm),
    }


def check_kraus_completeness(kraus_ops: Any, tol: float = 1e-10) -> dict[str, Any]:
    acc = sum(K.conjugate().T @ K for K in kraus_ops)
    dim = acc.shape[0]
    res = np.linalg.norm(acc - np.eye(dim))
    return {"complete": bool(res <= tol), "residual": float(res)}


def check_trace_preservation(channel: Any, rho: Any, tol: float = 1e-10) -> dict[str, Any]:
    out = channel(rho)
    tr = np.trace(out)
    return {
        "trace_preserved": bool(abs(tr - np.trace(rho)) <= tol),
        "trace_real": float(np.real(tr)),
        "trace_imag": float(np.imag(tr)),
    }


def check_born_probabilities(rho: Any, projectors: Any, tol: float = 1e-10) -> dict[str, Any]:
    probs = np.array([np.trace(P @ rho).real for P in projectors], dtype=float)
    return {
        "nonnegative": bool(np.all(probs >= -tol)),
        "normalized": bool(abs(probs.sum() - 1.0) <= tol),
        "probabilities": probs,
        "passed": bool(np.all(probs >= -tol) and abs(probs.sum() - 1.0) <= tol),
    }


def check_adelic_product_formula(num: int, den: int = 1) -> dict[str, Any]:
    val = product_formula_rational(num, den)
    return {"value": val, "passed": bool(val == Fraction(1, 1))}


def check_adelic_character_cancellation(
    num: int, den: int = 1, primes: Any = (2, 3, 5)
) -> dict[str, Any]:
    return {
        "passed": diagonal_character_cancels(num, den, primes),
        "num": num,
        "den": den,
        "primes": tuple(primes),
    }


def check_intention_simplex(intention: Any, tol: float = 1e-10) -> dict[str, Any]:
    intention = np.asarray(intention, dtype=float)
    return {
        "nonnegative": bool(np.all(-tol <= intention)),
        "normalized": bool(abs(intention.sum() - 1) <= tol),
        "passed": bool(np.all(-tol <= intention) and abs(intention.sum() - 1) <= tol),
    }


def check_bottom_up_weights(W: Any, tol: float = 1e-10) -> dict[str, Any]:
    W = np.asarray(W, dtype=float)
    return {
        "nonnegative": bool(np.all(-tol <= W)),
        "normalized": bool(abs(W.sum() - 1) <= tol),
        "passed": bool(np.all(-tol <= W) and abs(W.sum() - 1) <= tol),
    }


def check_top_down_bias(P0: Any, PI: Any, target_action: int, tol: float = 1e-12) -> dict[str, Any]:
    P0 = np.asarray(P0, dtype=float)
    PI = np.asarray(PI, dtype=float)
    changed_up = PI[target_action] > P0[target_action] + tol
    not_forced = PI[target_action] < 1.0 - tol
    return {
        "bias_increased": bool(changed_up),
        "not_forced": bool(not_forced),
        "passed": bool(changed_up and not_forced),
    }
