#!/usr/bin/env python3
"""Evaluate held-out outer folds, calibration, support, and subgroups."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from owl.cadc.artifacts import atomic_json  # noqa: E402
from owl.cadc.config import load_phase4_config  # noqa: E402
from owl.cadc.evaluation import (  # noqa: E402
    evaluate_forecast,
    evaluate_rankings,
    evaluate_survival,
)
from owl.cadc.scalarization import quantile_cvar  # noqa: E402
from owl.cadc.schema import ACTION_FAMILY_REGISTRY  # noqa: E402
from owl.cadc.support import SupportCalibrator  # noqa: E402


def _scalar(value: np.ndarray) -> np.ndarray:
    return (
        value[..., 0]
        + 0.7 * value[..., 1]
        + 0.3 * value[..., 2]
        + 0.3 * value[..., 3]
        + 0.2 * value[..., 4]
        - 4.0 * (1.0 - value[..., 5])
    )


def _regression_diagnostics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    from scipy.stats import spearmanr

    forecast = evaluate_forecast(prediction, target)
    truth = target.reshape(-1).astype(np.float64)
    predicted = prediction.reshape(-1).astype(np.float64)
    residual = truth - predicted
    total = np.sum((truth - truth.mean()) ** 2, dtype=np.float64)
    forecast["r2"] = float(1.0 - np.sum(residual**2, dtype=np.float64) / max(total, 1e-12))
    spearman = float(spearmanr(predicted, truth).statistic)
    forecast["spearman_defined"] = bool(np.isfinite(spearman))
    forecast["spearman"] = spearman if np.isfinite(spearman) else 0.0
    design = np.stack((np.ones(predicted.size), predicted), axis=1)
    intercept, slope = np.linalg.lstsq(design, truth, rcond=None)[0]
    forecast["calibration_intercept"] = float(intercept)
    forecast["calibration_slope"] = float(slope)
    return forecast


def _ranking_with_supported_rows(
    scores: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> dict[str, float]:
    valid = mask.sum(axis=1) >= 2
    if not valid.any():
        raise ValueError("ranking evaluation has no rows with two executable candidates")
    result = evaluate_rankings(scores[valid], target[valid], mask[valid])
    result["excluded_low_candidate_rows"] = float((~valid).sum())
    return result


def _subgroup_rank(
    scores: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> dict[str, float | str]:
    valid = mask.sum(axis=1) >= 2
    if not valid.any():
        return {"status": "insufficient_subgroup_support", "rows": 0.0}
    selected_scores = scores[valid]
    selected_target = target[valid]
    selected_mask = mask[valid]
    if selected_scores.shape[1] != 22:
        width = selected_scores.shape[1]
        if width > 22:
            raise ValueError("subgroup action axis exceeds immutable action count")
        padded_scores = np.zeros((selected_scores.shape[0], 22), dtype=np.float64)
        padded_target = np.zeros_like(padded_scores)
        padded_mask = np.zeros_like(padded_scores, dtype=bool)
        padded_scores[:, :width] = selected_scores
        padded_target[:, :width] = selected_target
        padded_mask[:, :width] = selected_mask
        selected_scores, selected_target, selected_mask = (
            padded_scores,
            padded_target,
            padded_mask,
        )
    return _ranking_with_supported_rows(
        selected_scores, selected_target, selected_mask
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_phase4_config(args.config)
    root = Path(args.input).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        import polars as pl
    except ImportError as exc:
        raise RuntimeError("Phase 4 subgroup evaluation requires Polars") from exc
    decision_metadata = pl.read_parquet(
        Path(args.dataset).resolve() / "canonical_data" / "decision_context"
    ).select("source_decision_id", "condition").unique(
        subset=["source_decision_id"]
    )
    condition_lookup = dict(
        zip(
            decision_metadata["source_decision_id"].to_list(),
            decision_metadata["condition"].to_list(),
            strict=True,
        )
    )
    folds = []
    for prediction_path in sorted(root.glob("outer-*/calibrated_predictions.npz")):
        outer = int(prediction_path.parent.name.split("-", 1)[1])
        data = np.load(prediction_path, allow_pickle=False)
        roles = data["split_roles"].astype(str)
        outer_folds = data["outer_folds"].astype(np.int16)
        test_decisions = (roles == "train") & (outer_folds == outer)
        if not test_decisions.any():
            raise ValueError(f"outer fold {outer} has no test decisions")
        scores = data["rank_score"][test_decisions]
        target = data["target_scalar"][test_decisions]
        raw_target = data["target_outcomes"][test_decisions]
        mask = data["target_mask"][test_decisions].astype(bool)
        decisions, horizons, actions = scores.shape
        ranking = _ranking_with_supported_rows(
            scores.reshape(decisions * horizons, actions),
            target.reshape(decisions * horizons, actions),
            mask.reshape(decisions * horizons, actions),
        )
        forecast = evaluate_forecast(scores, target, mask)
        structural_forecast = evaluate_forecast(
            data["outcome_mean"][test_decisions], raw_target, mask[..., None]
        )
        quantile_forecast = evaluate_forecast(
            data["return_quantiles"][test_decisions],
            data["target_scalar_quantiles"][test_decisions],
            mask[..., None],
        )
        predicted_cvar = quantile_cvar(
            data["return_quantiles"][test_decisions],
            config.scalarization.quantiles,
            alpha=config.scalarization.cvar_alpha,
        )
        cvar_forecast = evaluate_forecast(
            predicted_cvar,
            data["target_scalar_cvar"][test_decisions],
            mask,
        )
        survival = evaluate_survival(
            data["survival_probability"][test_decisions],
            data["target_alive"][test_decisions],
            mask,
        )
        survival_count = mask.sum(axis=2).clip(min=1)
        action_agnostic_alive = (
            data["target_alive"][test_decisions] * mask
        ).sum(axis=2) / survival_count
        xgboost_survival_baseline = evaluate_survival(
            data["xgboost_survival_baseline"][test_decisions],
            action_agnostic_alive,
            np.ones_like(action_agnostic_alive, dtype=bool),
        )
        cause_probability = data["cause_probability"][test_decisions]
        cause_target = data["target_death_cause_probability"][test_decisions]
        cause_brier = float(
            np.mean(np.sum((cause_probability - cause_target) ** 2, axis=-1)[mask])
        )
        agent_tree_ranking = _ranking_with_supported_rows(
            data["xgboost_agent_rank"][test_decisions].reshape(
                decisions * horizons, actions
            ),
            target.reshape(decisions * horizons, actions),
            mask.reshape(decisions * horizons, actions),
        )
        oracle_scores = data["xgboost_oracle_rank"][test_decisions]
        oracle_ranking = _ranking_with_supported_rows(
            oracle_scores.reshape(decisions * horizons, actions),
            target.reshape(decisions * horizons, actions),
            mask.reshape(decisions * horizons, actions),
        )
        baseline_target = _scalar(data["action_agnostic_target"][test_decisions])
        neural_baseline = _scalar(
            data["neural_viability_baseline"][test_decisions].reshape(
                decisions, horizons, -1
            )
        )
        tree_baseline = _scalar(
            data["xgboost_viability_baseline"][test_decisions].reshape(
                decisions, horizons, -1
            )
        )
        baseline_metrics = {
            "neural": _regression_diagnostics(neural_baseline, baseline_target),
            "xgboost": _regression_diagnostics(tree_baseline, baseline_target),
        }
        action_agnostic_scores = np.broadcast_to(
            neural_baseline[..., None], scores.shape
        )
        action_agnostic_ranking = _ranking_with_supported_rows(
            action_agnostic_scores.reshape(decisions * horizons, actions),
            target.reshape(decisions * horizons, actions),
            mask.reshape(decisions * horizons, actions),
        )
        negative = np.finfo(np.float64).min
        agent_best = np.argmax(np.where(mask, scores, negative), axis=2)
        oracle_best = np.argmax(np.where(mask, oracle_scores, negative), axis=2)
        selected_action = np.broadcast_to(
            data["selected_actions"][test_decisions, None], (decisions, horizons)
        )
        row = np.arange(decisions)[:, None]
        horizon_row = np.arange(horizons)[None, :]
        oracle_best_value = oracle_scores[row, horizon_row, oracle_best]
        agent_best_oracle_value = oracle_scores[row, horizon_row, agent_best]
        selected_oracle_value = oracle_scores[row, horizon_row, selected_action]
        information_regret = oracle_best_value - agent_best_oracle_value
        decision_regret = agent_best_oracle_value - selected_oracle_value
        oracle_total_regret = oracle_best_value - selected_oracle_value
        decomposition_error = np.max(
            np.abs(information_regret + decision_regret - oracle_total_regret)
        )
        selected_supported = mask[row, horizon_row, selected_action] & (
            mask.sum(axis=2) >= 2
        )
        if not selected_supported.any():
            raise ValueError("oracle regret has no supported factual selections")
        lower = data["lower"][test_decisions]
        upper = data["upper"][test_decisions]
        coverage = float(np.mean(((target >= lower) & (target <= upper))[mask]))
        width = upper - lower
        manifest = json.loads(
            (prediction_path.parent / "calibration_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        support_index = np.load(manifest["support_index"], allow_pickle=False)
        support = SupportCalibrator(
            k=config.support.knn_k,
            minimum_seeds=config.support.minimum_seeds,
            minimum_decisions=config.support.minimum_decisions,
            minimum_repeats=config.support.minimum_repeats,
            maximum_disagreement=config.support.maximum_ensemble_disagreement,
            maximum_interval_width=config.support.maximum_conformal_width,
        ).fit(support_index["embeddings"], support_index["seeds"])
        train_decisions = (roles == "train") & (outer_folds != outer)
        train_support = data["target_mask"][train_decisions].sum(axis=0)
        action_support = np.broadcast_to(
            train_support[None, :, :], mask.shape
        )[mask]
        decisions_support = support.decide_batch(
            data["embedding"][test_decisions][mask],
            action_support=action_support,
            repeat_support=data["repeat_count"][test_decisions][mask],
            disagreement=data["rank_disagreement"][test_decisions][mask],
            interval_width=width[mask],
        )
        supported = decisions_support["support_status"] == "supported"
        supported_metrics = (
            evaluate_forecast(scores[mask][supported], target[mask][supported])
            if supported.any()
            else {"rows": 0.0, "status": "no_supported_rows"}
        )
        test_ids = data["decision_ids"].astype(str)[test_decisions]
        conditions = np.asarray([condition_lookup[value] for value in test_ids])
        subgroup: dict[str, dict[str, object]] = {
            "horizon": {},
            "action_family": {},
            "context_family": {},
            "seed": {},
        }
        horizon_values = data["horizons"].astype(np.int32)
        if horizon_values.size != horizons:
            raise ValueError("persisted horizon registry does not match prediction axis")
        for horizon in range(horizons):
            subgroup["horizon"][str(int(horizon_values[horizon]))] = {
                "forecast": evaluate_forecast(
                    scores[:, horizon], target[:, horizon], mask[:, horizon]
                ),
                "ranking": _subgroup_rank(
                    scores[:, horizon], target[:, horizon], mask[:, horizon]
                ),
            }
        family_names = np.asarray(
            [value.primary_family.value for value in ACTION_FAMILY_REGISTRY]
        )
        for family in sorted(set(family_names.tolist())):
            action_selected = family_names == family
            family_score = scores[:, :, action_selected].reshape(-1, action_selected.sum())
            family_target = target[:, :, action_selected].reshape(
                -1, action_selected.sum()
            )
            family_mask = mask[:, :, action_selected].reshape(-1, action_selected.sum())
            subgroup["action_family"][family] = {
                "forecast": evaluate_forecast(
                    family_score, family_target, family_mask
                ),
                "ranking": _subgroup_rank(
                    family_score, family_target, family_mask
                ),
            }
        for condition in sorted(set(conditions.tolist())):
            keep = conditions == condition
            subgroup["context_family"][condition] = {
                "decisions": int(keep.sum()),
                "forecast": evaluate_forecast(scores[keep], target[keep], mask[keep]),
                "ranking": _subgroup_rank(
                    scores[keep].reshape(-1, actions),
                    target[keep].reshape(-1, actions),
                    mask[keep].reshape(-1, actions),
                ),
            }
        test_seed_values = data["seeds"][test_decisions]
        for seed in sorted(set(test_seed_values.tolist())):
            keep = test_seed_values == seed
            subgroup["seed"][str(int(seed))] = {
                "decisions": int(keep.sum()),
                "forecast": evaluate_forecast(scores[keep], target[keep], mask[keep]),
                "ranking": _subgroup_rank(
                    scores[keep].reshape(-1, actions),
                    target[keep].reshape(-1, actions),
                    mask[keep].reshape(-1, actions),
                ),
            }
        information_action = np.asarray([1, 11], dtype=np.int64)
        information_mask = mask[:, :, information_action]
        information_target = (
            raw_target[:, :, information_action, 8]
            + raw_target[:, :, information_action, 9]
        )
        information_components = data["information_components"][test_decisions][
            :, :, information_action
        ]
        epistemic = {
            "new_information": evaluate_forecast(
                information_components[..., 0], information_target, information_mask
            ),
            "later_action_change": {
                "status": "unsupported_evidence",
                "reason": (
                    "counterfactual branch evidence lacks later selected-action records"
                ),
            },
            "later_value_improvement": evaluate_forecast(
                information_components[..., 2],
                target[:, :, information_action],
                information_mask,
            ),
            "cost_adjusted_control_value": evaluate_forecast(
                information_components[..., 3],
                target[:, :, information_action],
                information_mask,
            ),
        }
        externality = evaluate_forecast(
            data["externality_prediction"][test_decisions],
            raw_target[..., 10:15],
            mask[..., None],
        )
        folds.append(
            {
                "outer_fold": outer,
                "test_decisions": int(test_decisions.sum()),
                "forecast": forecast,
                "structural_outcome_forecast": structural_forecast,
                "quantile_forecast": quantile_forecast,
                "cvar_forecast": cvar_forecast,
                "ranking": ranking,
                "survival": survival,
                "xgboost_action_agnostic_survival": xgboost_survival_baseline,
                "cause_brier": cause_brier,
                "action_agnostic_baselines": baseline_metrics,
                "action_agnostic_ranking": action_agnostic_ranking,
                "xgboost_agent_ranking": agent_tree_ranking,
                "oracle_diagnostic_ranking": oracle_ranking,
                "oracle_regret": {
                    "mean_information_regret": float(
                        np.mean(information_regret[selected_supported])
                    ),
                    "mean_decision_regret_in_oracle_units": float(
                        np.mean(decision_regret[selected_supported])
                    ),
                    "mean_oracle_total_regret": float(
                        np.mean(oracle_total_regret[selected_supported])
                    ),
                    "decomposition_max_abs_error": float(decomposition_error),
                    "oracle_model_used_by_primary_grader": False,
                },
                "conformal_coverage": coverage,
                "mean_interval_width": float(np.mean(width[mask], dtype=np.float64)),
                "support_status_counts": {
                    value: int(np.sum(decisions_support["support_status"] == value))
                    for value in np.unique(decisions_support["support_status"])
                },
                "supported_forecast": supported_metrics,
                "epistemic_value": epistemic,
                "externality_forecast": externality,
                "subgroups": subgroup,
            }
        )
    if not folds:
        raise FileNotFoundError("no calibrated fold predictions found")
    aggregate = {
        "forecast_mae_mean": float(
            np.mean([value["forecast"]["mae"] for value in folds], dtype=np.float64)
        ),
        "top1_accuracy_mean": float(
            np.mean(
                [value["ranking"]["top1_accuracy"] for value in folds],
                dtype=np.float64,
            )
        ),
        "mean_regret_mean": float(
            np.mean(
                [value["ranking"]["mean_regret"] for value in folds],
                dtype=np.float64,
            )
        ),
        "survival_brier_mean": float(
            np.mean([value["survival"]["brier"] for value in folds], dtype=np.float64)
        ),
        "conformal_coverage_mean": float(
            np.mean([value["conformal_coverage"] for value in folds], dtype=np.float64)
        ),
        "action_conditioned_minus_action_agnostic_top1": float(
            np.mean(
                [
                    value["ranking"]["top1_accuracy"]
                    - value["action_agnostic_ranking"]["top1_accuracy"]
                    for value in folds
                ],
                dtype=np.float64,
            )
        ),
        "action_conditioned_minus_action_agnostic_regret": float(
            np.mean(
                [
                    value["action_agnostic_ranking"]["mean_regret"]
                    - value["ranking"]["mean_regret"]
                    for value in folds
                ],
                dtype=np.float64,
            )
        ),
    }
    seed_top1 = [
        float(seed_value["ranking"]["top1_accuracy"])
        for fold in folds
        for seed_value in fold["subgroups"]["seed"].values()
        if seed_value.get("ranking", {}).get("status")
        != "insufficient_subgroup_support"
    ]
    seed_regret = [
        float(seed_value["ranking"]["mean_regret"])
        for fold in folds
        for seed_value in fold["subgroups"]["seed"].values()
        if seed_value.get("ranking", {}).get("status")
        != "insufficient_subgroup_support"
    ]
    if not seed_top1 or not seed_regret:
        raise ValueError("seed-level primary metric aggregation has no supported groups")
    aggregate["seed_level_intervals"] = {
        "method": "empirical_seed_percentile_95",
        "aggregation_unit": "independent_world_seed",
        "top1_accuracy": {
            "mean": float(np.mean(seed_top1, dtype=np.float64)),
            "lower": float(np.quantile(seed_top1, 0.025)),
            "upper": float(np.quantile(seed_top1, 0.975)),
            "seeds": len(seed_top1),
        },
        "mean_regret": {
            "mean": float(np.mean(seed_regret, dtype=np.float64)),
            "lower": float(np.quantile(seed_regret, 0.025)),
            "upper": float(np.quantile(seed_regret, 0.975)),
            "seeds": len(seed_regret),
        },
    }
    atomic_json(
        output / "evaluation.json",
        {
            "schema_version": "owl.cadc.phase4-evaluation.v1",
            "passed": True,
            "folds": folds,
            "aggregate": aggregate,
            "all_rows_reported": True,
            "supported_rows_reported": True,
            "phase5_locked": True,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
