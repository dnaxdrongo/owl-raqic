from __future__ import annotations

import numpy as np
from scripts._phase4_counterfactual_runtime import (
    DeferredTransferCounterfactualScheduler,
    _rehash_forced_state,
)

from owl.counterfactual.rng_registry import branch_seed
from owl.counterfactual.scheduler import BranchStatus, CounterfactualScheduler
from owl.counterfactual.state_hash import hash_state
from tests.counterfactual_phase25_helpers import source_run


def _execute(scheduler, source, cfg, run):
    decision_id = source.decisions.materialize_ids(run.ds.backend)[0]
    action = int(source.decisions.selected_action[0])
    seed = branch_seed(int(cfg.world.seed), source.state.source_state_id, 0)
    return scheduler._execute_branch(  # noqa: SLF001 - adapter parity seam
        source,
        0,
        decision_id,
        action,
        0,
        seed,
        anchor=False,
    )


def test_deferred_transfer_adapter_matches_reference_branch(tmp_path) -> None:
    cfg, run, source = source_run(tmp_path)
    try:
        reference = _execute(CounterfactualScheduler(run, cfg), source, cfg, run)
        deferred = _execute(
            DeferredTransferCounterfactualScheduler(run, cfg), source, cfg, run
        )
        assert reference.status == deferred.status == BranchStatus.COMPLETED
        assert reference.pre_force_hash.root == deferred.pre_force_hash.root
        assert reference.post_force_hash.root == deferred.post_force_hash.root
        assert {
            horizon: value.root for horizon, value in reference.horizon_hashes.items()
        } == {horizon: value.root for horizon, value in deferred.horizon_hashes.items()}
        assert reference.force_changed_leaves == deferred.force_changed_leaves
        assert len(reference.evidence) == len(deferred.evidence)
        for expected, actual in zip(reference.evidence, deferred.evidence, strict=True):
            assert expected.tick == actual.tick
            assert expected.event_codes == actual.event_codes
            assert expected.contribution_codes == actual.contribution_codes
            for name in expected.event_arrays:
                np.testing.assert_array_equal(
                    expected.event_arrays[name], actual.event_arrays[name]
                )
            for name in expected.contribution_arrays:
                np.testing.assert_array_equal(
                    expected.contribution_arrays[name],
                    actual.contribution_arrays[name],
                )
        for horizon in reference.outcomes:
            for name in reference.outcomes[horizon].values:
                np.testing.assert_array_equal(
                    reference.outcomes[horizon].values[name],
                    deferred.outcomes[horizon].values[name],
                )
    finally:
        run.close(checkpoint=False)


def test_deferred_transfer_adapter_fails_closed_on_pending_bound(tmp_path) -> None:
    cfg, run, source = source_run(tmp_path)
    cfg.counterfactual.max_packet_bytes = 1024
    cfg.counterfactual.max_pending_bytes = 1024
    try:
        result = _execute(
            DeferredTransferCounterfactualScheduler(run, cfg), source, cfg, run
        )
        assert result.status == BranchStatus.FAILED
        assert result.failure is not None
        assert "max_pending_bytes" in result.failure
    finally:
        run.close(checkpoint=False)


def test_incremental_force_hash_matches_full_hash_and_rejects_other_changes(
    tmp_path,
) -> None:
    cfg, run, source = source_run(tmp_path)
    branch = source.state.branch_clone()
    try:
        baseline = hash_state(branch)
        branch.readout[0, 0] = (int(branch.readout[0, 0]) + 1) % 22
        incremental = _rehash_forced_state(
            branch,
            source.state,
            baseline,
            allowed_changed_leaves=frozenset(
                {"arrays.readout", "arrays.raqic_readout"}
            ),
        )
        assert incremental.root == hash_state(branch).root
        branch.health[0, 0] += 1.0
        with np.testing.assert_raises_regex(RuntimeError, "outside the registered seam"):
            _rehash_forced_state(
                branch,
                source.state,
                baseline,
                allowed_changed_leaves=frozenset(
                    {"arrays.readout", "arrays.raqic_readout"}
                ),
            )
    finally:
        run.close(checkpoint=False)
