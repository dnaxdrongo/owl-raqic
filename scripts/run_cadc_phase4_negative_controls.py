#!/usr/bin/env python3
"""Run required label, temporal, mechanism, and oracle-leakage controls."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from owl.cadc.artifacts import atomic_json  # noqa: E402
from owl.cadc.config import load_phase4_config  # noqa: E402
from owl.cadc.evaluation import NegativeControlRunner, evaluate_rankings  # noqa: E402
from owl.cadc.features import FeatureDefinition, FeatureRegistry  # noqa: E402
from owl.cadc.schema import (  # noqa: E402
    ACTION_FAMILY_REGISTRY,
    ActionFamily,
    FeaturePerspective,
    FeatureStage,
)


def _ranking(scores: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    valid = mask.sum(axis=1) >= 2
    if not valid.any():
        raise ValueError("negative control has no rank rows with two candidates")
    return evaluate_rankings(scores[valid], target[valid], mask[valid])


def _scalar(value: np.ndarray) -> np.ndarray:
    return (
        value[..., 0]
        + 0.7 * value[..., 1]
        + 0.3 * value[..., 2]
        + 0.3 * value[..., 3]
        + 0.2 * value[..., 4]
        - 4.0 * (1.0 - value[..., 5])
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_phase4_config(args.config)
    runner = NegativeControlRunner(config.negative_controls.random_seed)
    root = Path(args.input).resolve()
    try:
        import polars as pl
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_squared_error
        from sklearn.preprocessing import OneHotEncoder
    except ImportError as exc:
        raise RuntimeError(
            "negative controls require Polars and scikit-learn"
        ) from exc
    dataset = Path(args.dataset).resolve() / "canonical_data"
    decision_metadata = pl.read_parquet(dataset / "decision_context").select(
        "source_decision_id",
        "seed",
        "condition",
        "tick",
        "decision_sequence",
    ).unique(subset=["source_decision_id"])
    metadata_lookup = {
        str(row[0]): row[1:]
        for row in decision_metadata.iter_rows()
    }
    fold_results = []
    for path in sorted(root.glob("outer-*/calibrated_predictions.npz")):
        outer = int(path.parent.name.split("-", 1)[1])
        data = np.load(path, allow_pickle=False)
        roles = data["split_roles"].astype(str)
        folds = data["outer_folds"].astype(np.int16)
        selected = (roles == "train") & (folds == outer)
        score = data["rank_score"][selected]
        target = data["target_scalar"][selected]
        mask = data["target_mask"][selected].astype(bool)
        decisions, horizons, actions = score.shape
        selected_ids = data["decision_ids"].astype(str)[selected]
        metadata_rows = [metadata_lookup[value] for value in selected_ids]
        selected_seeds = np.asarray([value[0] for value in metadata_rows])
        selected_conditions = np.asarray([str(value[1]) for value in metadata_rows])
        temporal_order = np.asarray(
            [f"{int(value[2]):012d}:{int(value[3]):012d}" for value in metadata_rows]
        )
        temporal_strata = np.char.add(
            np.char.add(selected_seeds.astype(str), ":"), selected_conditions
        )
        baseline = _ranking(
            score.reshape(-1, actions), target.reshape(-1, actions), mask.reshape(-1, actions)
        )
        strata = np.repeat(np.arange(decisions * horizons), actions)
        action_shuffled = runner.action_shuffle(target.reshape(-1), strata).reshape(target.shape)
        action_control = _ranking(
            score.reshape(-1, actions),
            action_shuffled.reshape(-1, actions),
            mask.reshape(-1, actions),
        )
        temporal = runner.temporal_break(
            target.reshape(decisions, -1),
            temporal_order,
            temporal_strata,
        ).reshape(target.shape)
        temporal_control = _ranking(
            score.reshape(-1, actions), temporal.reshape(-1, actions), mask.reshape(-1, actions)
        )
        family_lookup = {value: index for index, value in enumerate(ActionFamily)}
        family = np.asarray(
            [family_lookup[value.primary_family] for value in ACTION_FAMILY_REGISTRY],
            dtype=np.int16,
        )
        family_grid = np.broadcast_to(family, target.shape)
        horizon_grid = np.broadcast_to(
            np.arange(horizons, dtype=np.int16)[None, :, None], target.shape
        )
        context_grid = np.broadcast_to(
            selected_conditions[:, None, None], target.shape
        )
        compatible = np.char.add(
            np.char.add(
                np.char.add(family_grid.astype(str), ":h"),
                horizon_grid.astype(str),
            ),
            np.char.add(":c", context_grid.astype(str)),
        ).reshape(-1)
        target_shuffled = runner.target_shuffle(target.reshape(-1), compatible).reshape(
            target.shape
        )
        target_control = _ranking(
            score.reshape(-1, actions),
            target_shuffled.reshape(-1, actions),
            mask.reshape(-1, actions),
        )
        rng = np.random.default_rng(config.negative_controls.random_seed + outer)
        random_control = _ranking(
            rng.normal(size=score.shape).reshape(-1, actions),
            target.reshape(-1, actions),
            mask.reshape(-1, actions),
        )
        action_agnostic = _scalar(
            data["neural_viability_baseline"][selected].reshape(
                decisions, horizons, -1
            )
        )
        action_agnostic_scores = np.broadcast_to(
            action_agnostic[..., None], score.shape
        )
        action_agnostic_control = _ranking(
            action_agnostic_scores.reshape(-1, actions),
            target.reshape(-1, actions),
            mask.reshape(-1, actions),
        )
        candidate_counts = mask.sum(axis=2)
        eligible = candidate_counts >= 2
        chance_top1 = float(
            np.mean(1.0 / candidate_counts[eligible], dtype=np.float64)
        )
        fold_results.append(
            {
                "outer_fold": outer,
                "baseline": baseline,
                "action_shuffle": action_control,
                "target_shuffle": target_control,
                "temporal_break": temporal_control,
                "random_ranker": random_control,
                "action_agnostic_rank": action_agnostic_control,
                "chance_top1": chance_top1,
                "action_shuffle_degradation": baseline["top1_accuracy"]
                - action_control["top1_accuracy"],
                "target_shuffle_degradation": baseline["top1_accuracy"]
                - target_control["top1_accuracy"],
                "temporal_break_degradation": baseline["top1_accuracy"]
                - temporal_control["top1_accuracy"],
            }
        )
    if not fold_results:
        raise FileNotFoundError("no calibrated fold predictions found")
    pairs = pl.read_parquet(dataset / "pair_labels").sort(
        "source_decision_id", "action_a", "action_b", "horizon", "repeat_index"
    )
    matched = pairs["advantage_risk_averse"].to_numpy()
    mismatched = pairs.with_columns(
        pl.col("value_b_risk_averse")
        .shift(1)
        .over("source_decision_id", "action_a", "action_b", "horizon")
        .alias("mismatched_b")
    ).filter(pl.col("mismatched_b").is_not_null()).with_columns(
        (pl.col("value_a_risk_averse") - pl.col("mismatched_b")).alias(
            "mismatched_advantage"
        )
    )
    matched_variance = float(np.var(matched, dtype=np.float64))
    mismatched_variance = float(
        np.var(mismatched["mismatched_advantage"].to_numpy(), dtype=np.float64)
    )
    repeat_mismatch_passed = mismatched_variance + 1e-12 >= matched_variance
    candidate = pl.read_parquet(dataset / "candidate_context").select(
        "source_decision_id", "action_index", "utility"
    )
    branch = pl.read_parquet(dataset / "branch_targets").group_by(
        "source_decision_id", "forced_action"
    ).agg(pl.col("agent_risk_averse").mean().alias("target"))
    decision = pl.read_parquet(dataset / "decision_context").select(
        "source_decision_id", "split_role", "outer_fold"
    )
    mechanism = candidate.join(
        branch,
        left_on=["source_decision_id", "action_index"],
        right_on=["source_decision_id", "forced_action"],
        how="inner",
    ).join(decision, on="source_decision_id", how="inner")
    mechanism_metrics = []
    for outer in mechanism["outer_fold"].unique().sort().to_list():
        train = mechanism.filter(
            (pl.col("split_role") == "train") & (pl.col("outer_fold") != outer)
        )
        test = mechanism.filter(
            (pl.col("split_role") == "train") & (pl.col("outer_fold") == outer)
        )
        if train.height and test.height:
            model = Ridge(alpha=1.0).fit(
                train.select("utility").to_numpy(), train["target"].to_numpy()
            )
            prediction = model.predict(test.select("utility").to_numpy())
            mechanism_metrics.append(
                {
                    "outer_fold": int(outer),
                    "rmse": float(
                        np.sqrt(mean_squared_error(test["target"].to_numpy(), prediction))
                    ),
                    "accepted_as_primary": False,
                }
            )
    mechanism_control_passed = bool(mechanism_metrics)
    condition = pl.read_parquet(dataset / "decision_context").select(
        "source_decision_id", "condition", "split_role", "outer_fold"
    ).unique(subset=["source_decision_id"])
    condition_target = pl.read_parquet(dataset / "branch_targets").group_by(
        "source_decision_id"
    ).agg(pl.col("agent_risk_averse").mean().alias("target"))
    condition = condition.join(condition_target, on="source_decision_id", how="inner")
    condition_metrics = []
    for outer in condition["outer_fold"].unique().sort().to_list():
        train = condition.filter(
            (pl.col("split_role") == "train") & (pl.col("outer_fold") != outer)
        )
        test = condition.filter(
            (pl.col("split_role") == "train") & (pl.col("outer_fold") == outer)
        )
        if train.height and test.height:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            train_x = encoder.fit_transform(train.select("condition").to_numpy())
            test_x = encoder.transform(test.select("condition").to_numpy())
            model = Ridge(alpha=1.0).fit(train_x, train["target"].to_numpy())
            prediction = model.predict(test_x)
            condition_metrics.append(
                {
                    "outer_fold": int(outer),
                    "rmse": float(
                        np.sqrt(mean_squared_error(test["target"].to_numpy(), prediction))
                    ),
                    "accepted_as_primary": False,
                }
            )
    condition_control_passed = bool(condition_metrics)
    oracle_guard_passed = False
    try:
        FeatureRegistry(
            (
                FeatureDefinition(
                    "oracle_context.oracle_food",
                    "oracle_context",
                    "oracle_food",
                    "float32",
                    FeaturePerspective.AGENT_PRIMARY,
                    FeatureStage.PRE_CHOICE,
                ),
            )
        )
    except ValueError:
        oracle_guard_passed = True
    collapse = all(
        value["action_shuffle"]["top1_accuracy"] <= value["chance_top1"] + 0.15
        and value["target_shuffle"]["top1_accuracy"] <= value["chance_top1"] + 0.15
        and value["temporal_break"]["top1_accuracy"] <= value["chance_top1"] + 0.20
        and value["random_ranker"]["top1_accuracy"] <= value["chance_top1"] + 0.15
        for value in fold_results
    )
    passed = bool(
        collapse
        and oracle_guard_passed
        and repeat_mismatch_passed
        and mechanism_control_passed
        and condition_control_passed
    )
    atomic_json(
        args.output,
        {
            "schema_version": "owl.cadc.phase4-negative-controls.v1",
            "passed": passed,
            "classification": (
                "NEGATIVE_CONTROLS_COLLAPSED"
                if passed
                else "FAILED_CLOSED"
            ),
            "folds": fold_results,
            "oracle_leakage_guard_passed": oracle_guard_passed,
            "mechanism_only": mechanism_metrics,
            "mechanism_only_primary_acceptance": False,
            "condition_only": condition_metrics,
            "condition_only_primary_acceptance": False,
            "repeat_mismatch": {
                "passed": repeat_mismatch_passed,
                "matched_variance": matched_variance,
                "mismatched_variance": mismatched_variance,
            },
            "phase5_locked": True,
        },
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
