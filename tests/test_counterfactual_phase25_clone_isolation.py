from __future__ import annotations

from owl.counterfactual.state_clone import assert_no_alias
from tests.counterfactual_phase25_helpers import source_run


def test_complete_clone_has_no_alias_and_mutation_is_isolated(tmp_path) -> None:
    _, run, source = source_run(tmp_path)
    try:
        branch = source.state.branch_clone()
        assert_no_alias(source.state, branch)
        before = source.state.arrays["health"].copy()
        branch.health.fill(0)
        assert (source.state.arrays["health"] == before).all()
        assert "compiled_execution_action" in branch.arrays
    finally:
        run.close(checkpoint=False)
