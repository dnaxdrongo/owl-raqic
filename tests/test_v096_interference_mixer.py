from __future__ import annotations

import numpy as np
import pytest

from owl.core.actions import Action
from owl_raqic.gpu.actualization_extensions import apply_legal_interference_mixer
from owl_raqic.math.action_graph import action_family_edges, legal_subspace_unitary


def _inputs() -> tuple[tuple[str, ...], tuple[tuple[int, int], ...]]:
    names = tuple(action.name for action in Action)
    return names, action_family_edges(names)


def test_zero_strength_is_exact_identity_and_one_legal_action_is_fixed() -> None:
    names, edges = _inputs()
    amplitude = np.zeros((1, len(names)), dtype=np.complex128)
    amplitude[0, 0] = 1.0
    one_legal = np.zeros_like(amplitude, dtype=bool)
    one_legal[0, 0] = True
    zero = apply_legal_interference_mixer(
        amplitude, one_legal, edges, strength=0.0, trotter_steps=1, xp=np
    )
    assert zero is amplitude
    nonzero = apply_legal_interference_mixer(
        amplitude, one_legal, edges, strength=0.7, trotter_steps=4, xp=np
    )
    np.testing.assert_array_equal(nonzero, amplitude)


def test_dense_sequence_matches_explicit_unitary_and_preserves_norm() -> None:
    names, edges = _inputs()
    rng = np.random.default_rng(184)
    p = rng.random(len(names))
    p /= p.sum()
    phases = rng.uniform(-np.pi, np.pi, len(names))
    amplitude = np.sqrt(p) * np.exp(1j * phases)
    authority = rng.random(len(names)) > 0.3
    authority[0] = True
    amplitude = np.where(authority, amplitude, 0.0)
    amplitude /= np.linalg.norm(amplitude)
    mixed = apply_legal_interference_mixer(
        amplitude[None, :], authority[None, :], edges, strength=0.19, trotter_steps=3, xp=np
    )[0]
    unitary = legal_subspace_unitary(len(names), authority, edges, strength=0.19, trotter_steps=3)
    np.testing.assert_allclose(mixed, unitary @ amplitude, atol=3e-15, rtol=3e-15)
    np.testing.assert_allclose(np.vdot(mixed, mixed).real, 1.0, atol=3e-15)
    np.testing.assert_array_equal(mixed[~authority], np.zeros(np.sum(~authority), complex))


def test_relative_phase_changes_probability_but_global_phase_does_not() -> None:
    edges = ((0, 1),)
    authority = np.asarray([[True, True]])
    base = np.sqrt(np.asarray([[0.5, 0.5]])).astype(np.complex128)
    plus = base * np.exp(1j * np.asarray([[0.0, np.pi / 2.0]]))
    minus = base * np.exp(1j * np.asarray([[0.0, -np.pi / 2.0]]))
    p_plus = (
        np.abs(
            apply_legal_interference_mixer(
                plus, authority, edges, strength=0.6, trotter_steps=1, xp=np
            )
        )
        ** 2
    )
    p_minus = (
        np.abs(
            apply_legal_interference_mixer(
                minus, authority, edges, strength=0.6, trotter_steps=1, xp=np
            )
        )
        ** 2
    )
    assert not np.allclose(p_plus, p_minus)
    shifted = plus * np.exp(1j * 0.93)
    p_shifted = (
        np.abs(
            apply_legal_interference_mixer(
                shifted, authority, edges, strength=0.6, trotter_steps=1, xp=np
            )
        )
        ** 2
    )
    np.testing.assert_allclose(p_shifted, p_plus, atol=2e-15)


def test_invalid_or_nonfinite_mixer_inputs_fail_closed() -> None:
    with pytest.raises(ValueError, match="finite"):
        apply_legal_interference_mixer(
            np.asarray([[np.nan + 0j, 1.0 + 0j]]),
            np.asarray([[True, True]]),
            ((0, 1),),
            strength=0.1,
            trotter_steps=1,
            xp=np,
        )
    with pytest.raises(ValueError, match="invalid edge"):
        apply_legal_interference_mixer(
            np.asarray([[1.0 + 0j, 0.0 + 0j]]),
            np.asarray([[True, True]]),
            ((0, 2),),
            strength=0.1,
            trotter_steps=1,
            xp=np,
        )


def test_preallocated_amplitude_and_pair_scratch_match_allocating_path() -> None:
    names, edges = _inputs()
    rng = np.random.default_rng(9303)
    rows = 5
    raw = rng.normal(size=(rows, len(names))) + 1j * rng.normal(size=(rows, len(names)))
    authority = rng.random((rows, len(names))) > 0.25
    authority[:, 0] = True
    raw = np.where(authority, raw, 0.0)
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    expected = apply_legal_interference_mixer(
        raw,
        authority,
        edges,
        strength=0.17,
        trotter_steps=3,
        xp=np,
    )
    output = np.empty_like(raw)
    left = np.empty((rows,), dtype=np.complex128)
    right = np.empty((rows,), dtype=np.complex128)
    actual = apply_legal_interference_mixer(
        raw,
        authority,
        edges,
        strength=0.17,
        trotter_steps=3,
        xp=np,
        output=output,
        left_scratch=left,
        right_scratch=right,
    )
    assert actual is output
    np.testing.assert_allclose(actual, expected, atol=3e-15, rtol=3e-15)


def test_preallocated_scratch_shape_mismatch_fails_closed() -> None:
    with pytest.raises(ValueError, match="left_scratch"):
        apply_legal_interference_mixer(
            np.asarray([[1.0 + 0j, 0.0 + 0j]]),
            np.asarray([[True, True]]),
            ((0, 1),),
            strength=0.1,
            trotter_steps=1,
            xp=np,
            left_scratch=np.empty((2,), dtype=np.complex128),
        )
