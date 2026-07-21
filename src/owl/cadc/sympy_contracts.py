"""Provide fail-closed wrappers for symbolic and numerical math checks."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from owl.cadc.scalarization import HomeostaticDrive, candidate_advantage, stabilized_softmax


def verify_math_contracts(reference: str | Path | None = None) -> dict[str, Any]:
    """Verify core identities and optionally bind the supplied SymPy receipt."""
    checks: dict[str, bool] = {}
    logits = np.asarray([[-1000.0, 0.0, 1000.0], [1.0, 1.0, 1.0]])
    probabilities = stabilized_softmax(logits)
    checks["softmax_finite"] = bool(np.isfinite(probabilities).all())
    checks["softmax_normalized"] = bool(
        np.allclose(probabilities.sum(axis=-1), 1.0, rtol=0.0, atol=1e-12)
    )
    checks["paired_antisymmetry"] = bool(
        np.array_equal(
            candidate_advantage(np.arange(5), np.arange(5)[::-1]),
            -candidate_advantage(np.arange(5)[::-1], np.arange(5)),
        )
    )
    drive = HomeostaticDrive(
        names=("health", "resource"),
        setpoints=(1.0, 1.0),
        scales=(1.0, 2.0),
        lower_asymmetry=(2.0, 1.0),
        upper_asymmetry=(1.0, 1.0),
    )
    source = np.asarray([[0.5, 0.5], [1.0, 1.0]])
    checks["homeostatic_setpoint_zero"] = bool(np.isclose(drive.drive(source)[1], 0.0))
    checks["homeostatic_improvement_positive"] = bool(
        drive.improvement(source, np.asarray([[0.25, 0.25], [0.0, 0.0]]))[0] > 0
    )
    supplied: Mapping[str, Any] | None = None
    if reference is not None:
        payload = json.loads(Path(reference).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("SymPy receipt must contain a mapping")
        supplied = payload
        supplied_checks = payload.get("checks")
        if isinstance(supplied_checks, list):
            checks["supplied_sympy_receipt_passed"] = all(
                bool(item.get("passed")) for item in supplied_checks if isinstance(item, dict)
            )
        else:
            checks["supplied_sympy_receipt_passed"] = bool(payload.get("passed"))
    failures = sorted(name for name, passed in checks.items() if not passed)
    return {
        "schema_version": "owl.cadc.phase4-math-contracts.v1",
        "passed": not failures,
        "checks": checks,
        "failures": failures,
        "supplied_reference": dict(supplied) if supplied is not None else None,
    }

