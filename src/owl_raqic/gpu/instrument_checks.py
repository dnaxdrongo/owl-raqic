from __future__ import annotations

from typing import Any

import numpy as np

from owl_raqic.math.checks import check_density_matrix, check_kraus_completeness
from owl_raqic.math.instruments import (
    feedback_unitaries,
    preparation_kraus_from_amplitudes,
    recursive_channel,
)
from owl_raqic.math.states import density_from_state, ket0


def analytic_probability_checks(
    probabilities: np.ndarray, mask: np.ndarray | None = None, tol: float = 1e-10
) -> dict[str, Any]:
    p = np.asarray(probabilities, dtype=float)
    out = {
        "rows": int(p.shape[0]) if p.ndim == 2 else 0,
        "max_row_sum_error": float(np.max(np.abs(p.sum(axis=1) - 1.0))) if p.size else 0.0,
        "min_probability": float(np.min(p)) if p.size else 0.0,
        "max_probability": float(np.max(p)) if p.size else 0.0,
        "passed": True,
    }
    out["passed"] = bool(out["max_row_sum_error"] <= tol and out["min_probability"] >= -tol)
    if mask is not None and p.size:
        m = np.asarray(mask, dtype=bool)
        illegal_mass = float(np.max(np.where(m, 0.0, p))) if m.shape == p.shape else float("inf")
        out["max_illegal_probability"] = illegal_mass
        out["passed"] = bool(out["passed"] and illegal_mass <= tol)
    return out


def explicit_instrument_audit(
    probabilities: np.ndarray, phases: np.ndarray | None = None, limit: int = 8, tol: float = 1e-10
) -> dict[str, Any]:
    p = np.asarray(probabilities, dtype=float)
    ph = np.zeros_like(p) if phases is None else np.asarray(phases, dtype=float)
    rows = min(int(limit), p.shape[0])
    max_kraus = 0.0
    max_trace = 0.0
    min_eig = 0.0
    failures = 0
    for i in range(rows):
        amps = np.sqrt(np.maximum(p[i], 0.0)) * np.exp(1j * ph[i])
        amps = amps / np.linalg.norm(amps)
        kraus, _, _ = preparation_kraus_from_amplitudes(amps)
        kcheck = check_kraus_completeness(kraus)
        max_kraus = max(max_kraus, float(kcheck.get("residual", 0.0)))
        feedback = feedback_unitaries(len(amps))
        rho0 = density_from_state(ket0(len(amps)))
        rho1 = recursive_channel(kraus, feedback, rho0)
        tr_res = float(abs(np.trace(rho1) - 1.0))
        dcheck = check_density_matrix(rho1, tol=tol)
        max_trace = max(max_trace, tr_res)
        min_eig = min(min_eig, float(dcheck.get("min_eigenvalue", 0.0)))
        if max_kraus > tol or tr_res > tol or min_eig < -tol:
            failures += 1
    return {
        "audited_rows": rows,
        "max_kraus_residual": max_kraus,
        "max_trace_residual": max_trace,
        "min_eigenvalue": min_eig,
        "failures": failures,
        "passed": bool(failures == 0),
    }
