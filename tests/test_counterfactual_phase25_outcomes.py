from __future__ import annotations

from owl.core.actions import Action
from tests.counterfactual_phase25_helpers import execute_action


def test_outcome_vector_contains_survival_lineage_world_and_delta_fields(tmp_path) -> None:
    _, run, _, result = execute_action(tmp_path, Action.COMMUNICATE)
    try:
        values = result.outcomes[1].values
        required = {
            "alive",
            "death_evidence",
            "lineage_id",
            "population",
            "world_food",
            "health_delta",
            "integration_delta",
            "target_distance_delta",
        }
        assert required <= set(values)
    finally:
        run.close(checkpoint=False)
