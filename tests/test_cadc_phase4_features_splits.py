from __future__ import annotations

import numpy as np
import pytest

from owl.cadc.features import (
    FeatureDefinition,
    FeatureRegistry,
    build_agent_features,
)
from owl.cadc.schema import FeaturePerspective, FeatureStage, SplitRole
from owl.cadc.splits import build_grouped_splits, seed_role_map, validate_no_leakage


def _definition(table: str, column: str) -> FeatureDefinition:
    return FeatureDefinition(
        f"{table}.{column}",
        table,
        column,
        "float32",
        FeaturePerspective.AGENT_PRIMARY,
        FeatureStage.PRE_CHOICE,
    )


@pytest.mark.parametrize(
    ("table", "column"),
    [
        ("oracle_context", "oracle_food"),
        ("candidates", "utility"),
        ("execution", "execution_success"),
        ("agent_context", "agent_parent_intention"),
    ],
)
def test_primary_leakage_sources_are_rejected(table: str, column: str) -> None:
    with pytest.raises(ValueError):
        FeatureRegistry((_definition(table, column),))


def test_oracle_mutation_cannot_change_agent_view() -> None:
    definition = _definition("agent_context", "agent_health")
    registry = FeatureRegistry((definition,))
    first = {
        "agent_context": {"agent_health": np.asarray([0.4, 0.8])},
        "oracle_context": {"oracle_food": np.asarray([0.0, 0.0])},
    }
    second = {
        **first,
        "oracle_context": {"oracle_food": np.asarray([99.0, -99.0])},
    }
    assert np.array_equal(
        build_agent_features(first, registry)[definition.name],
        build_agent_features(second, registry)[definition.name],
    )


def test_primary_and_mechanism_views_are_disjoint() -> None:
    registry = FeatureRegistry()
    primary = set(registry.names(FeaturePerspective.AGENT_PRIMARY))
    mechanism = set(registry.names(FeaturePerspective.MECHANISM_MEDIATION))
    assert primary
    assert mechanism
    assert primary.isdisjoint(mechanism)


def test_grouped_split_is_leave_seed_out_and_seals_confirmatory_rows() -> None:
    roles = seed_role_map(
        development=[1, 2], validation=[3], calibration=[4], phase5=[5], phase6=[6]
    )
    groups = [
        {"seed": seed, "run_id": f"run-{seed}-{copy}", "condition": "c", "world_id": copy}
        for seed in (1, 2, 3, 4)
        for copy in range(2)
    ]
    registry = build_grouped_splits(
        groups, seed_roles=roles, outer_folds=3, inner_folds=2, master_seed=7
    )
    for seed in (1, 2, 3, 4):
        assert len({value.outer_fold for value in registry.assignments if value.seed == seed}) == 1
    with pytest.raises(ValueError, match="sealed confirmatory"):
        validate_no_leakage(
            np.asarray(["a"]), np.asarray([SplitRole.PHASE5_SEALED.value])
        )


def test_decision_cannot_cross_split_roles() -> None:
    with pytest.raises(ValueError, match="crosses splits"):
        validate_no_leakage(
            np.asarray(["same", "same"]), np.asarray(["train", "validation"])
        )
