from __future__ import annotations

from owl.core.actions import Action
from owl.counterfactual.forcing import build_forced_action_batch, inject_forced_actions
from owl.counterfactual.state_hash import differing_leaves, hash_state
from tests.counterfactual_phase25_helpers import source_run


def test_force_changes_only_committed_high_level_fields(tmp_path) -> None:
    _, run, source = source_run(tmp_path)
    try:
        branch = source.state.branch_clone()
        branch.metadata["cfg"] = run.cfg
        batch = build_forced_action_batch(source.decisions, [0], [int(Action.FLEE)])
        before = hash_state(branch)
        assert bool(inject_forced_actions(branch, batch)[0])
        changed = set(differing_leaves(before, hash_state(branch)))
        assert changed <= {"arrays.readout", "arrays.raqic_readout"}
        branch.action_target_ow_id[5, 5, 0] = -123
        assert not bool(inject_forced_actions(branch, batch)[0])
    finally:
        run.close(checkpoint=False)
