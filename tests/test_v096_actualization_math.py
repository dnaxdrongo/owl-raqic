from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from owl.core.actions import Action
from owl_raqic.math.action_graph import (
    ACTION_GRAPH_VERSION,
    action_family_edges,
    action_graph_hash,
    legal_subspace_unitary,
)
from owl_raqic.math.actualization_reference import (
    explicit_interference_reference,
    two_action_probability_law,
)


def test_uploaded_sympy_audit_contract_passes() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "audit_v096_actualization_math.py"
    spec = importlib.util.spec_from_file_location("v096_sympy_audit", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.build_report()
    assert report["passed"] is True
    assert all(report["checks"].values())


def test_action_graph_is_versioned_deterministic_and_unitary() -> None:
    names = tuple(action.name for action in Action)
    edges = action_family_edges(names)
    assert ACTION_GRAPH_VERSION == "semantic_families_v1"
    assert edges == action_family_edges(names)
    assert len(edges) == len(set(edges)) == 22
    assert len(action_graph_hash(names)) == 64
    authority = np.ones(len(names), dtype=bool)
    unitary = legal_subspace_unitary(len(names), authority, edges, strength=0.31, trotter_steps=3)
    np.testing.assert_allclose(unitary.conj().T @ unitary, np.eye(len(names)), atol=3e-14)


def test_two_action_closed_form_matches_exact_matrix_and_global_phase() -> None:
    q1, q2 = 0.35, 0.65
    phi1, phi2, kappa = 0.2, -0.7, 0.29
    closed = np.asarray(two_action_probability_law(q1, q2, phi1, phi2, kappa))
    probabilities = np.asarray([q1, q2])
    phases = np.asarray([phi1, phi2])
    authority = np.asarray([True, True])
    exact = explicit_interference_reference(
        probabilities,
        phases,
        authority,
        ((0, 1),),
        strength=kappa,
        trotter_steps=1,
    ).probabilities
    np.testing.assert_allclose(exact, closed, atol=2e-15, rtol=2e-15)
    shifted = explicit_interference_reference(
        probabilities,
        phases + 1.2345,
        authority,
        ((0, 1),),
        strength=kappa,
        trotter_steps=1,
    ).probabilities
    np.testing.assert_allclose(shifted, exact, atol=2e-15, rtol=2e-15)
