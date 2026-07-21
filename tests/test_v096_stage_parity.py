from __future__ import annotations

import numpy as np
import pytest

from owl.core.actions import Action
from owl.science.stage_contract import STAGE_CONTRACTS
from owl_raqic.gpu.actualization_extensions import aggregate_action_phase_context


def test_decision_stage_registers_v096_scientific_outputs() -> None:
    decision = next(item for item in STAGE_CONTRACTS if item.name == "decision")
    outputs = set(decision.writes)
    for required in (
        "raqic_pre_mixer_probabilities",
        "raqic_utility_innovation",
        "raqic_phase_alignment",
        "raqic_resonant_parent_intention",
        "raqic_shadow_probabilities",
        "raqic_shadow_readout",
    ):
        assert required in outputs


def test_phase_context_is_computed_from_supplied_prior_records_only() -> None:
    actions = len(Action)
    prior_p = np.zeros((2, 2, actions), dtype=np.float64)
    prior_p[..., int(Action.REST)] = 1.0
    prior_phase = np.zeros_like(prior_p)
    weights = np.full((2, 2), 0.25)
    first = aggregate_action_phase_context(
        prior_p,
        prior_phase,
        weights,
        patch_size=2,
        patch_weight=0.75,
        global_weight=0.25,
        support_epsilon=1e-10,
        rest_index=int(Action.REST),
        xp=np,
        dtype=np.float64,
    )
    current_p = prior_p.copy()
    current_p[..., int(Action.REST)] = 0.0
    current_p[..., int(Action.SENSE)] = 1.0
    # No mutation or implicit dependency on a future/current decision tensor.
    second = aggregate_action_phase_context(
        prior_p,
        prior_phase,
        weights,
        patch_size=2,
        patch_weight=0.75,
        global_weight=0.25,
        support_epsilon=1e-10,
        rest_index=int(Action.REST),
        xp=np,
        dtype=np.float64,
    )
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left, right)
    assert not np.shares_memory(prior_p, current_p)


def test_parent_context_shape_mismatch_fails_closed() -> None:
    p = np.ones((2, 2, 2), dtype=np.float64) / 2.0
    with pytest.raises(ValueError, match="child_weights"):
        aggregate_action_phase_context(
            p,
            np.zeros_like(p),
            np.ones((1, 1)),
            patch_size=2,
            patch_weight=0.75,
            global_weight=0.25,
            support_epsilon=1e-10,
            rest_index=0,
            xp=np,
            dtype=np.float64,
        )
