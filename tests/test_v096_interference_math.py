from __future__ import annotations

import numpy as np

from owl_raqic.gpu.actualization_extensions import apply_legal_interference_mixer
from owl_raqic.math.action_graph import action_family_edges, legal_subspace_unitary

ACTIONS = (
    "REST",
    "SENSE",
    "MOVE_N",
    "MOVE_S",
    "MOVE_E",
    "MOVE_W",
    "MOVE_NE",
    "MOVE_NW",
    "MOVE_SE",
    "MOVE_SW",
    "FEED",
    "COMMUNICATE",
    "INHIBIT",
    "INTEGRATE",
    "REPAIR",
    "REPRODUCE",
    "INGEST",
    "EXPEL",
    "SPLIT",
    "MERGE",
    "FLEE",
    "PURSUE",
)


def test_action_graph_is_deterministic_and_duplicate_free() -> None:
    edges = action_family_edges(ACTIONS)
    assert edges == action_family_edges(ACTIONS)
    assert len(edges) == len(set(edges))
    assert len(edges) == 22


def test_dense_pair_sequence_matches_explicit_unitary() -> None:
    edges = action_family_edges(ACTIONS)
    authority = np.ones((1, len(ACTIONS)), dtype=bool)
    probabilities = np.linspace(1.0, 2.0, len(ACTIONS))
    probabilities /= probabilities.sum()
    phases = np.linspace(-1.0, 1.0, len(ACTIONS))
    amplitude = np.sqrt(probabilities) * np.exp(1j * phases)
    mixed = apply_legal_interference_mixer(
        amplitude[None, :], authority, edges, strength=0.4, trotter_steps=2, xp=np
    )[0]
    unitary = legal_subspace_unitary(
        len(ACTIONS), authority[0], edges, strength=0.4, trotter_steps=2
    )
    np.testing.assert_allclose(mixed, unitary @ amplitude, atol=2e-15, rtol=2e-15)
    np.testing.assert_allclose(np.vdot(mixed, mixed).real, 1.0, atol=2e-15)
    np.testing.assert_allclose(unitary.conj().T @ unitary, np.eye(len(ACTIONS)), atol=2e-15)


def test_illegal_actions_are_not_mixed() -> None:
    edges = action_family_edges(ACTIONS)
    authority = np.zeros((1, len(ACTIONS)), dtype=bool)
    authority[0, 0] = True
    amplitude = np.zeros((1, len(ACTIONS)), dtype=np.complex128)
    amplitude[0, 0] = 1.0
    mixed = apply_legal_interference_mixer(
        amplitude, authority, edges, strength=1.0, trotter_steps=3, xp=np
    )
    np.testing.assert_array_equal(mixed, amplitude)
