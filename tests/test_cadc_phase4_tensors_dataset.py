from __future__ import annotations

import numpy as np
import pytest

from owl.cadc.dataset import GroupedBatchIndex, join_source_decisions
from owl.cadc.tensors import assemble_fixed_action_tensors


def _batch() -> object:
    decisions = 2
    candidate_rows = decisions * 22
    direction_rows = decisions * 16
    branch_decision = np.repeat(np.arange(decisions), 2 * 22)
    branch_action = np.tile(np.repeat(np.arange(22), 2), decisions)
    branch_horizon = np.tile([1, 1], decisions * 22)
    return assemble_fixed_action_tensors(
        decision_ids=np.asarray(["d0", "d1"]),
        seeds=np.asarray([1, 2]),
        split_roles=np.asarray(["train", "train"]),
        outer_folds=np.asarray([0, 1]),
        context_columns={"health": np.asarray([0.5, 0.6])},
        oracle_context_columns={"oracle_food": np.asarray([1.0, 2.0])},
        candidate_decision_index=np.repeat(np.arange(decisions), 22),
        candidate_action_index=np.tile(np.arange(22), decisions),
        candidate_columns={"distance": np.arange(candidate_rows)},
        candidate_executable=np.ones(candidate_rows, dtype=bool),
        direction_decision_index=np.repeat(np.arange(decisions), 16),
        direction_family_index=np.tile(np.repeat(np.arange(2), 8), decisions),
        direction_index=np.tile(np.arange(8), decisions * 2),
        direction_columns={"direction_score": np.arange(direction_rows)},
        direction_executable=np.ones(direction_rows, dtype=bool),
        branch_decision_index=branch_decision,
        branch_action_index=branch_action,
        branch_horizon=branch_horizon,
        branch_repeat_index=np.tile([0, 1], decisions * 22),
        outcome_columns={"value": np.tile([1.0, 3.0], decisions * 22)},
        branch_scalar_target=np.tile([2.0, 4.0], decisions * 22),
        registered_horizons=[1],
        quantile_levels=[0.05, 0.5, 0.95],
        cvar_alpha=0.5,
        selected_actions=np.asarray([20, 21]),
    )


def test_fixed_axes_and_repeat_mean() -> None:
    batch = _batch()
    assert batch.candidates.shape[:2] == (2, 22)
    assert batch.oracle_context.shape == (2, 1)
    assert batch.selected_actions.tolist() == [20, 21]
    assert batch.directions.shape[:3] == (2, 2, 8)
    assert batch.outcomes.shape == (2, 1, 22, 1)
    assert np.all(batch.outcomes == 2.0)
    assert np.all(batch.scalar_targets == 3.0)
    assert np.allclose(batch.scalar_quantiles[..., 1], 3.0)
    assert np.all(batch.scalar_cvar == 2.0)
    assert np.all(batch.repeat_count == 2)


def test_bad_candidate_cardinality_fails() -> None:
    index = GroupedBatchIndex.build(["a"] * 21 + ["b"] * 22)
    with pytest.raises(ValueError, match="exactly 22"):
        index.validate_exact_size(22)


def test_direction_nonfinite_masks_are_preserved_at_input_boundary() -> None:
    batch = _batch()
    assert batch.directions.shape[-1] == 4


def test_source_join_requires_unique_exact_key_and_action() -> None:
    factual = {
        "tick": np.asarray([2]),
        "decision_sequence": np.asarray([7]),
        "ow_id": np.asarray([9]),
        "selected_action": np.asarray([20]),
    }
    source = {
        "tick": np.asarray([2]),
        "decision_sequence": np.asarray([7]),
        "ow_id": np.asarray([9]),
        "factual_selected_action": np.asarray([20]),
    }
    assert join_source_decisions(factual, source)["selected_action"].tolist() == [20]
    source["factual_selected_action"] = np.asarray([21])
    with pytest.raises(ValueError, match="selected actions"):
        join_source_decisions(factual, source)
