from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("qiskit")

from owl.core.actions import Action
from owl_raqic.qiskit_backend.circuit_families import build_circuit_family


def test_interference_family_builds_padded_authority_preserving_unitary() -> None:
    probabilities = np.linspace(1.0, 2.0, len(Action))
    probabilities /= probabilities.sum()
    phases = np.linspace(-0.5, 0.5, len(Action))
    authority = np.ones(len(Action), dtype=bool)
    built = build_circuit_family(
        "interference",
        probabilities,
        phases,
        authority_mask=authority,
        mixer_strength=0.2,
        mixer_trotter_steps=1,
        action_names=tuple(action.name for action in Action),
        measure=False,
    )
    assert built.metadata["mode"] == "interference"
    assert built.recovery_gates["unitarity_residual"] <= 1e-10
