from __future__ import annotations

from owl.record.cadc_schema import CADC_ACTION_TRANSITION_SCHEMA_DIGEST
from tests.counterfactual_phase25_helpers import source_run


def test_source_preserves_factual_v2_candidate_and_direction_grains(tmp_path) -> None:
    _, run, source = source_run(tmp_path)
    try:
        assert source.decisions.factual_schema_digest == CADC_ACTION_TRANSITION_SCHEMA_DIGEST
        assert source.decisions.policy_legal.shape == (1, 22)
        assert source.decisions.prechoice_executable.shape == (1, 22)
        assert source.decisions.direction_fields["action_direction_y"].shape == (1, 2, 8)
        assert source.state.manifest.total_array_bytes > 0
    finally:
        run.close(checkpoint=False)
