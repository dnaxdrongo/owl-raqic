from __future__ import annotations

from owl.core.actions import Action
from owl.counterfactual.scheduler import BranchStatus
from tests.counterfactual_phase25_helpers import execute_action


def test_sense_branch_uses_authoritative_transition_and_information_outcomes(tmp_path) -> None:
    _, run, _, result = execute_action(tmp_path, Action.SENSE)
    try:
        assert result.status == BranchStatus.COMPLETED
        outcome = result.outcomes[1].values
        assert "active_sense_new_cell_count" in outcome
        assert "resource_delta" in outcome
        assert float(outcome["resource_delta"][0]) < 0
    finally:
        run.close(checkpoint=False)
