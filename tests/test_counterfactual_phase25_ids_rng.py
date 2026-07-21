from __future__ import annotations

from owl.counterfactual.rng_registry import branch_seed, registry_manifest
from owl.counterfactual.schema import stable_id


def test_ids_and_paired_seed_are_stable_and_action_independent() -> None:
    assert stable_id("x", 1, "ab") == stable_id("x", 1, "ab")
    assert stable_id("x", 1, "ab") != stable_id("x", "1", "ab")
    assert branch_seed(7, "source", 2) == branch_seed(7, "source", 2)
    assert branch_seed(7, "source", 2) != branch_seed(7, "source", 3)
    assert len(registry_manifest()["registry_digest"]) == 64
