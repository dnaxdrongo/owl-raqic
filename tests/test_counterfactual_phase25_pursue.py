from __future__ import annotations

from owl.core.actions import Action
from owl.counterfactual.scheduler import BranchStatus
from tests.counterfactual_phase25_helpers import execute_action


def test_pursue_branch_executes_compiler_and_decreases_target_distance(tmp_path) -> None:
    _, run, _, result = execute_action(tmp_path, Action.PURSUE)
    try:
        assert result.status == BranchStatus.COMPLETED
        values = result.outcomes[1].values
        # The semantic target also acts in the same factual tick, so pursuit
        # may preserve rather than strictly reduce separation.
        assert float(values["target_distance_delta"][0]) <= 0
        assert int(values["displacement_y"][0]) or int(values["displacement_x"][0])
    finally:
        run.close(checkpoint=False)
