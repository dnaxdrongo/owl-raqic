from __future__ import annotations

import numpy as np

from owl.cadc.outcomes import DeathCause, OutcomeRegistry, reduce_contributions, reduce_events
from owl.cadc.scalarization import (
    HomeostaticDrive,
    candidate_advantage,
    quantile_cvar,
    quantile_cvar_weights,
    stabilized_softmax,
)
from owl.cadc.sympy_contracts import verify_math_contracts
from owl.record.cadc_schema import CADCEventCode


def test_event_causes_are_horizon_specific_and_ambiguous_is_explicit() -> None:
    result = reduce_events(
        np.asarray(
            [
                int(CADCEventCode.STARVATION_EVIDENCE),
                int(CADCEventCode.DEATH),
                int(CADCEventCode.TOXIN_DAMAGE_EVIDENCE),
            ]
        ),
        np.asarray([1, 2, 3]),
        horizons=np.asarray([1, 2, 3]),
    )
    assert result["death_cause"].tolist() == [
        int(DeathCause.NONE),
        int(DeathCause.STARVATION),
        int(DeathCause.AMBIGUOUS),
    ]


def test_contribution_reduction_preserves_named_fields() -> None:
    result = reduce_contributions(
        ["health", "health", "resource"],
        np.asarray([1, 3, 2]),
        np.asarray([1.0, -0.25, 2.0]),
        horizons=np.asarray([1, 3]),
    )
    assert np.allclose(result["health"], [1.0, 0.75])
    assert np.allclose(result["resource"], [0.0, 2.0])


def test_outcome_registry_binds_externalities_to_anchor_deltas() -> None:
    registry = OutcomeRegistry()
    external = [
        value for value in registry.definitions if "delta_vs_anchor" in " ".join(value.evidence)
    ]
    assert len(external) == 5
    assert any(value.name == "lineage_persistence" for value in external)


def test_numerical_math_contracts_and_extreme_softmax() -> None:
    receipt = verify_math_contracts()
    assert receipt["passed"] is True
    probabilities = stabilized_softmax(np.asarray([[-10000.0, 0.0, 10000.0]]))
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)
    assert np.array_equal(candidate_advantage([1, 2], [3, 1]), [-2, 1])
    drive = HomeostaticDrive(("health",), (1.0,), (1.0,), (2.0,), (1.0,))
    assert drive.improvement(np.asarray([[0.5]]), np.asarray([[0.1]]))[0] > 0.0


def test_quantile_cvar_is_normalized_and_translation_equivariant() -> None:
    levels = (0.05, 0.1, 0.25, 0.5, 0.9)
    weights = quantile_cvar_weights(levels, alpha=0.1)
    assert np.allclose(weights, [0.75, 0.25, 0.0, 0.0, 0.0])
    quantiles = np.asarray([[1.0, 3.0, 4.0, 5.0, 8.0]])
    base = quantile_cvar(quantiles, levels, alpha=0.1)
    shifted = quantile_cvar(quantiles + 7.0, levels, alpha=0.1)
    assert np.allclose(base, [1.5])
    assert np.allclose(shifted, base + 7.0)
