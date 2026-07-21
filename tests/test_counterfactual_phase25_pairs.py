from __future__ import annotations

from collections import defaultdict

from owl.counterfactual.scheduler import CounterfactualScheduler
from tests.counterfactual_phase25_helpers import source_run


def test_executable_candidates_share_repeat_seed_and_form_pairs(tmp_path) -> None:
    cfg, run, source = source_run(tmp_path)
    try:
        result = CounterfactualScheduler(run, cfg).run_source(source)
        seeds = defaultdict(set)
        for branch in result.branches:
            if not branch.anchor:
                seeds[branch.repeat_index].add(branch.branch_seed)
        assert all(len(values) == 1 for values in seeds.values())
        assert result.pairs
        assert all(pair.branch_a != pair.branch_b for pair in result.pairs)
    finally:
        run.close(checkpoint=False)
