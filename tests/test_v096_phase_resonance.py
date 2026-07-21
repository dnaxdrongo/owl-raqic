from __future__ import annotations

import numpy as np

from owl_raqic.gpu.actualization_extensions import (
    aggregate_action_phase_context,
    phase_modulated_parent_intention,
)


def test_zero_resonance_exactly_recovers_normalized_parent() -> None:
    parent = np.asarray([[0.2, 0.8]])
    phase = np.asarray([[0.1, 1.2]])
    inherited = np.asarray([[0.4, 0.7]])
    coherence = np.asarray([[0.9, 0.5]])
    resonant, alignment = phase_modulated_parent_intention(
        parent,
        phase,
        inherited,
        coherence,
        0.0,
        xp=np,
        dtype=np.float64,
    )
    np.testing.assert_array_equal(resonant, parent)
    np.testing.assert_array_equal(alignment, np.zeros_like(parent))


def test_prior_tick_phasor_context_has_expected_direction_and_support() -> None:
    probabilities = np.zeros((2, 2, 2), dtype=np.float64)
    probabilities[..., 0] = 1.0
    phases = np.zeros_like(probabilities)
    phases[0, 0, 0] = np.pi / 2.0
    weights = np.full((2, 2), 0.25)
    patch_phase, patch_coherence, global_phase, global_coherence, parent_phase, parent_coh = (
        aggregate_action_phase_context(
            probabilities,
            phases,
            weights,
            patch_size=2,
            patch_weight=0.75,
            global_weight=0.25,
            support_epsilon=1e-10,
            rest_index=0,
            xp=np,
            dtype=np.float64,
        )
    )
    assert patch_phase.shape == (1, 1, 2)
    assert patch_coherence[0, 0, 0] < 1.0
    np.testing.assert_allclose(global_phase, patch_phase[0, 0])
    np.testing.assert_allclose(global_coherence, patch_coherence[0, 0])
    assert parent_phase.shape == probabilities.shape
    assert parent_coh.shape == probabilities.shape
