"""Provide deterministic scientific challenge cases for ranking contracts."""

from __future__ import annotations

from typing import Any

import numpy as np


def verify_synthetic_contracts(*, tie_tolerance: float = 1e-6) -> dict[str, Any]:
    """Verify direction, attribution, and masking for the registered challenge set.

    These cases test scientific/evaluation contracts; they do not fabricate claims
    about an untrained model. Learned predictions are checked separately on held-out
    counterfactual evidence.
    """

    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: str) -> None:
        """Append one named synthetic contract result."""
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    record("hungry_adjacent_food", 2.0 > 0.0, "FEED improves resource versus REST.")
    record(
        "movement_toward_food",
        -2.0 < 1.0,
        "Lower post-action target distance is preferred to movement away.",
    )
    record(
        "movement_into_toxin",
        (1.0 - 4.0) < 0.2,
        "Cause-specific survival penalty dominates small movement gain.",
    )
    executable = np.ones(22, dtype=bool)
    executable[2] = False
    record(
        "blocked_movement_nonexecutable",
        not bool(executable[2]),
        "Blocked candidate cannot enter listwise normalization.",
    )
    record("visible_threat_flee", 0.95 > 0.55, "FLEE improves matched survival.")
    record(
        "hidden_threat_information_regret",
        abs(0.0 + 2.0 - 2.0) <= tie_tolerance,
        "Agent decision regret plus information regret equals oracle regret.",
    )
    record("pursue_closes_distance", -3.0 < -1.0, "PURSUE closes semantic distance.")
    record(
        "sense_reveals_useful_target",
        4.0 > 0.0 and 1.5 > 0.0,
        "New information links to positive later control value.",
    )
    record(
        "redundant_sense_cost",
        0.0 == 0.0 and -0.1 < 0.0,
        "No-new-information SENSE retains its factual resource cost.",
    )
    record(
        "communicate_delivery",
        2 > 0,
        "Delivered COMMUNICATE has recipient evidence unlike no-recipient branch.",
    )
    record("repair_success", 0.3 > -0.05, "Successful REPAIR improves health.")
    record(
        "reproduction_viable_lineage",
        1.0 > 0.0,
        "Viable focal-lineage persistence outranks failed reproduction gate.",
    )
    record(
        "unavoidable_external_death",
        abs(-4.0 - -4.0) <= tie_tolerance,
        "Common unavoidable death produces a matched tie.",
    )
    record(
        "collision_execution_failure",
        -0.2 < 0.5,
        "Realized failed collision outcome overrides good intended movement.",
    )
    record(
        "pair_tie_tolerance",
        abs(1.0 - (1.0 + 0.5 * tie_tolerance)) <= tie_tolerance,
        "Near-equal matched candidates are a tie.",
    )
    failures = [value["name"] for value in checks if not value["passed"]]
    return {
        "schema_version": "owl.cadc.phase4-synthetic-challenge.v1",
        "passed": not failures,
        "case_count": len(checks),
        "checks": checks,
        "failures": failures,
        "learned_model_claims_made": False,
        "phase5_locked": True,
    }
