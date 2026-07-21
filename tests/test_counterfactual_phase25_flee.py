from __future__ import annotations

from owl.core.actions import Action
from owl.counterfactual.scheduler import BranchStatus
from tests.counterfactual_phase25_helpers import execute_action


def test_flee_branch_executes_compiler_and_increases_target_distance(tmp_path) -> None:
    _, run, _, result = execute_action(tmp_path, Action.FLEE)
    try:
        assert result.status == BranchStatus.COMPLETED
        assert float(result.outcomes[1].values["target_distance_delta"][0]) > 0
    finally:
        run.close(checkpoint=False)
