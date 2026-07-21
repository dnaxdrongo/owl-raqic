#!/usr/bin/env python3
"""Build the canonical CADC-MORE 2 dataset from a certified modeling corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from owl.cadc.artifacts import atomic_json  # noqa: E402
from owl.cadc.catalog import Phase4EvidenceCatalog  # noqa: E402
from owl.cadc.config import load_phase4_config  # noqa: E402
from owl.cadc.contracts import load_and_validate_certificate  # noqa: E402
from owl.cadc.dataset import CanonicalDatasetBuilder  # noqa: E402
from owl.cadc.features import FeatureRegistry  # noqa: E402
from owl.cadc.outcomes import OutcomeRegistry  # noqa: E402
from owl.cadc.schema import (  # noqa: E402
    ACTION_FAMILY_REGISTRY,
    PHASE4_SCHEMA_DIGEST,
)
from owl.cadc.splits import build_grouped_splits, seed_role_map  # noqa: E402
from owl.experiments.controller import _release_hash  # noqa: E402


def _dataset_validation(
    root: Path, *, expected_repeats: int, backend: str = "numpy"
) -> dict[str, object]:
    """Run post-transfer key, cardinality, target, and coverage gates."""

    if backend == "cupy":
        return _dataset_validation_gpu(root, expected_repeats=expected_repeats)

    try:
        import polars as pl
    except ImportError as exc:
        raise RuntimeError("dataset certification requires Polars") from exc
    decision = pl.read_parquet(root / "decision_context")
    candidate = pl.read_parquet(root / "candidate_context")
    direction = pl.read_parquet(root / "direction_context")
    target = pl.read_parquet(root / "branch_targets")
    if decision["source_decision_id"].n_unique() != decision.height:
        raise ValueError("decision_context uniqueness failed")
    candidate_counts = candidate.group_by("source_decision_id").len()["len"]
    direction_counts = direction.group_by("source_decision_id").len()["len"]
    if not (candidate_counts == 22).all():
        raise ValueError("candidate_context 22-row cardinality failed")
    if not (direction_counts == 16).all():
        raise ValueError("direction_context 16-row cardinality failed")
    if target.select(
        pl.struct("branch_id", "horizon").is_duplicated().any()
    ).item():
        raise ValueError("branch target key is not unique")
    repeat_contract = (
        target.group_by("source_decision_id", "forced_action", "horizon")
        .agg(
            pl.len().alias("rows"),
            pl.col("repeat_index").n_unique().alias("unique_repeats"),
            pl.col("repeat_index").min().alias("minimum_repeat"),
            pl.col("repeat_index").max().alias("maximum_repeat"),
        )
    )
    invalid_repeats = repeat_contract.filter(
        (pl.col("rows") != expected_repeats)
        | (pl.col("unique_repeats") != expected_repeats)
        | (pl.col("minimum_repeat") != 0)
        | (pl.col("maximum_repeat") != expected_repeats - 1)
    )
    if invalid_repeats.height:
        raise ValueError(
            "branch repeat indices must be unique and contiguous for every "
            "source/action/horizon group"
        )
    cause_columns = [f"death_cause_{value}" for value in range(5)]
    cause_error = target.select(
        (pl.sum_horizontal(*cause_columns) - 1.0).abs().max()
    ).item()
    if float(cause_error or 0.0) > 1e-6:
        raise ValueError("death-cause probabilities are not one-hot before aggregation")
    support = (
        target.group_by("forced_action", "horizon")
        .agg(
            pl.len().alias("branch_rows"),
            pl.col("source_decision_id").n_unique().alias("source_decisions"),
            pl.col("seed").n_unique().alias("independent_seeds"),
        )
        .sort("forced_action", "horizon")
        .to_dicts()
    )
    supported_actions = {int(value["forced_action"]) for value in support}
    supported_families = {
        ACTION_FAMILY_REGISTRY[action].primary_family.value
        for action in supported_actions
    }
    required_actions = {1, 20, 21}
    if not required_actions.issubset(supported_actions):
        raise ValueError("SENSE/FLEE/PURSUE branch support is incomplete")
    required_families = {
        value.primary_family.value for value in ACTION_FAMILY_REGISTRY
    }
    if supported_families != required_families:
        raise ValueError("development corpus does not cover every action family")
    context_coverage = (
        decision.group_by("condition")
        .agg(
            pl.len().alias("source_decisions"),
            pl.col("seed").n_unique().alias("independent_seeds"),
        )
        .sort("condition")
        .to_dicts()
    )
    return {
        "passed": True,
        "decision_rows": decision.height,
        "candidate_rows": candidate.height,
        "direction_rows": direction.height,
        "branch_target_rows": target.height,
        "candidate_cardinality": 22,
        "direction_cardinality": 16,
        "repeat_cardinality": expected_repeats,
        "repeat_indices_contiguous": True,
        "cause_one_hot_max_abs_error": float(cause_error or 0.0),
        "action_horizon_support": support,
        "supported_action_families": sorted(supported_families),
        "context_coverage": context_coverage,
        "sense_flee_pursue_supported": True,
        "sealed_roles_materialized": False,
    }


def _dataset_validation_gpu(
    root: Path, *, expected_repeats: int
) -> dict[str, object]:
    """Run full-table validation with cuDF and transfer only compact summaries."""

    try:
        import cudf
    except ImportError as exc:
        raise RuntimeError("GPU dataset certification requires cuDF") from exc
    decision = cudf.read_parquet(root / "decision_context")
    candidate = cudf.read_parquet(root / "candidate_context")
    direction = cudf.read_parquet(root / "direction_context")
    target = cudf.read_parquet(root / "branch_targets")
    if int(decision["source_decision_id"].nunique()) != len(decision):
        raise ValueError("decision_context uniqueness failed")
    candidate_counts = candidate.groupby("source_decision_id").size()
    direction_counts = direction.groupby("source_decision_id").size()
    if bool((candidate_counts != 22).any()):
        raise ValueError("candidate_context 22-row cardinality failed")
    if bool((direction_counts != 16).any()):
        raise ValueError("direction_context 16-row cardinality failed")
    if bool(target.duplicated(subset=["branch_id", "horizon"]).any()):
        raise ValueError("branch target key is not unique")
    if bool((target["repeat_index"] < -1).any()):
        raise ValueError("branch repeat indices contain an invalid negative sentinel")
    factual_anchor = target[target["repeat_index"] == -1]
    if not len(factual_anchor):
        raise ValueError("factual branch anchors are missing")
    anchor_keys = ["source_decision_id", "horizon"]
    if bool(factual_anchor.duplicated(subset=anchor_keys).any()):
        raise ValueError("factual branch anchors are not unique per source/horizon")
    if int(factual_anchor["source_decision_id"].nunique()) != int(
        target["source_decision_id"].nunique()
    ):
        raise ValueError("factual branch anchors do not cover every source decision")
    anchor_counts = factual_anchor.groupby("source_decision_id").size()
    if int(anchor_counts.nunique()) != 1:
        raise ValueError("factual branch anchor horizon coverage is inconsistent")
    counterfactual = target[target["repeat_index"] >= 0]
    if not len(counterfactual):
        raise ValueError("counterfactual branch repeats are missing")
    repeat_keys = ["source_decision_id", "forced_action", "horizon"]
    repeated = counterfactual.groupby(repeat_keys)
    repeat_contract = repeated.size().reset_index(name="rows")
    repeat_contract = repeat_contract.merge(
        repeated["repeat_index"].nunique().reset_index(name="unique_repeats"),
        on=repeat_keys,
    ).merge(
        repeated["repeat_index"].min().reset_index(name="minimum_repeat"),
        on=repeat_keys,
    ).merge(
        repeated["repeat_index"].max().reset_index(name="maximum_repeat"),
        on=repeat_keys,
    )
    invalid = repeat_contract[
        (repeat_contract["rows"] != expected_repeats)
        | (repeat_contract["unique_repeats"] != expected_repeats)
        | (repeat_contract["minimum_repeat"] != 0)
        | (repeat_contract["maximum_repeat"] != expected_repeats - 1)
    ]
    if len(invalid):
        raise ValueError(
            "counterfactual repeat indices must be unique and contiguous for every "
            "source/action/horizon group"
        )
    cause_columns = [f"death_cause_{value}" for value in range(5)]
    cause_error = float((target[cause_columns].sum(axis=1) - 1.0).abs().max())
    if cause_error > 1e-6:
        raise ValueError("death-cause probabilities are not one-hot before aggregation")
    support_keys = ["forced_action", "horizon"]
    supported = target.groupby(support_keys)
    support = supported.size().reset_index(name="branch_rows")
    support = support.merge(
        supported["source_decision_id"].nunique().reset_index(
            name="source_decisions"
        ),
        on=support_keys,
    ).merge(
        supported["seed"].nunique().reset_index(name="independent_seeds"),
        on=support_keys,
    ).sort_values(support_keys)
    support_records = support.to_arrow().to_pylist()
    supported_actions = {int(value["forced_action"]) for value in support_records}
    supported_families = {
        ACTION_FAMILY_REGISTRY[action].primary_family.value
        for action in supported_actions
    }
    if not {1, 20, 21}.issubset(supported_actions):
        raise ValueError("SENSE/FLEE/PURSUE branch support is incomplete")
    required_families = {
        value.primary_family.value for value in ACTION_FAMILY_REGISTRY
    }
    if supported_families != required_families:
        raise ValueError("development corpus does not cover every action family")
    contexts = decision.groupby("condition")
    context_coverage = contexts.size().reset_index(name="source_decisions").merge(
        contexts["seed"].nunique().reset_index(name="independent_seeds"),
        on="condition",
    ).sort_values("condition")
    return {
        "passed": True,
        "validation_backend": "cudf_cuda",
        "decision_rows": len(decision),
        "candidate_rows": len(candidate),
        "direction_rows": len(direction),
        "branch_target_rows": len(target),
        "candidate_cardinality": 22,
        "direction_cardinality": 16,
        "repeat_cardinality": expected_repeats,
        "repeat_indices_contiguous": True,
        "cause_one_hot_max_abs_error": cause_error,
        "action_horizon_support": support_records,
        "supported_action_families": sorted(supported_families),
        "context_coverage": context_coverage.to_arrow().to_pylist(),
        "sense_flee_pursue_supported": True,
        "sealed_roles_materialized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--corpus-certificate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_phase4_config(args.config)
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    corpus_certificate = json.loads(
        Path(args.corpus_certificate).read_text(encoding="utf-8")
    )
    if corpus_certificate.get("passed") is not True:
        raise RuntimeError("canonical dataset requires a passing corpus certificate")
    if corpus_certificate.get("plan_id") != plan.get("plan_id"):
        raise RuntimeError("corpus certificate and plan IDs do not match")
    if plan.get("config_sha256") != config.corpus_digest():
        raise RuntimeError("corpus plan scientific contract does not match configuration")
    provenance = load_and_validate_certificate(config.phase3_input.phase3_certificate)
    factual_roots = []
    counterfactual_roots = []
    groups = []
    for unit in plan["units"]:
        inventory_path = Path(unit["output_path"]) / "corpus_unit_inventory.json"
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        if inventory.get("passed") is not True:
            raise RuntimeError(f"corpus unit is not passing: {unit['unit_id']}")
        factual_roots.append(inventory["factual_root"])
        counterfactual_roots.append(inventory["counterfactual_root"])
        groups.append(
            {
                "seed": int(unit["seed"]),
                "run_id": unit["unit_id"],
                "condition": unit["context_family"],
                "world_id": unit["unit_id"],
            }
        )
    roles = seed_role_map(
        development=config.corpus.development_seeds,
        validation=config.corpus.validation_seeds,
        calibration=config.corpus.calibration_seeds,
        phase5=config.corpus.reserved_phase5_seeds,
        phase6=config.corpus.reserved_phase6_seeds,
    )
    split_registry = build_grouped_splits(
        groups,
        seed_roles=roles,
        outer_folds=config.splits.outer_folds,
        inner_folds=config.splits.inner_folds,
        master_seed=config.master_seed,
    )
    features = FeatureRegistry()
    outcomes = OutcomeRegistry()
    catalog = Phase4EvidenceCatalog.build(
        tuple(factual_roots), tuple(counterfactual_roots), provenance
    ).with_phase4_digests(
        dataset=PHASE4_SCHEMA_DIGEST,
        features=features.digest,
        outcomes=outcomes.digest,
        splits=split_registry.digest,
    )
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    catalog.write_receipt(output / "manifests" / "evidence_catalog.json")
    atomic_json(output / "manifests" / "feature_registry.json", features.manifest())
    atomic_json(output / "manifests" / "outcome_registry.json", outcomes.manifest())
    atomic_json(output / "manifests" / "split_registry.json", split_registry.manifest())
    builder = CanonicalDatasetBuilder(
        catalog,
        feature_registry=features,
        outcome_registry=outcomes,
        split_registry=split_registry,
        backend=config.runtime.backend,
        history_length=config.features.history_length,
    )
    receipts = builder.build_spines(output / "canonical_data")
    if not isinstance(config.corpus.repeat_policy, int):
        raise RuntimeError(
            "dataset construction requires a pilot-resolved integer repeat policy"
        )
    validation = _dataset_validation(
        output / "canonical_data",
        expected_repeats=config.corpus.repeat_policy,
        backend=config.runtime.backend,
    )
    atomic_json(
        output / "dataset_build_receipt.json",
        {
            "schema_version": "owl.cadc.phase4-dataset-build-receipt.v1",
            "passed": True,
            "dataset_id": builder.dataset_id,
            "parts": [receipt.__dict__ for receipt in receipts],
            "validation": validation,
            "phase3_source_sha256": provenance.phase3_source_sha256,
            "phase4_source_sha256": _release_hash(ROOT),
            "corpus_contract_sha256": config.corpus_digest(),
            "model_spec_sha256": config.model_spec_digest(),
            "phase5_locked": True,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
