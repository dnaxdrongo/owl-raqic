from __future__ import annotations

from owl.counterfactual.scheduler import BranchStatus, CounterfactualScheduler
from tests.counterfactual_phase25_helpers import source_run


def test_selected_action_anchor_matches_factual_horizon_one(tmp_path) -> None:
    cfg, run, source = source_run(tmp_path)
    try:
        scheduler = CounterfactualScheduler(run, cfg)
        decision_id = source.decisions.materialize_ids(run.ds.backend)[0]
        action = int(source.decisions.selected_action[0])
        result = scheduler._execute_branch(  # noqa: SLF001
            source, 0, decision_id, action, -1, int(cfg.world.seed), anchor=True
        )
        assert result.status == BranchStatus.COMPLETED
        assert result.anchor_matches == {1: True}
    finally:
        run.close(checkpoint=False)
