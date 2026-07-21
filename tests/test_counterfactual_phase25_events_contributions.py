from __future__ import annotations

from owl.core.actions import Action
from tests.counterfactual_phase25_helpers import execute_action


def test_branch_evidence_is_host_staged_bounded_and_typed(tmp_path) -> None:
    _, run, _, result = execute_action(tmp_path, Action.FLEE)
    try:
        assert result.evidence
        packet = result.evidence[0]
        assert packet.nbytes > 0
        assert packet.event_codes
        assert packet.contribution_codes
        assert packet.contribution_fields
        assert packet.event_arrays["event_active"].__class__.__module__.startswith("numpy")
    finally:
        run.close(checkpoint=False)
