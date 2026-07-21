#!/usr/bin/env python3
"""Build a deterministic blinded casebook from held-out outer-fold predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from owl.cadc.artifacts import atomic_json, sha256_file  # noqa: E402
from owl.cadc.config import load_phase4_config  # noqa: E402
from owl.cadc.features import FeatureRegistry  # noqa: E402
from owl.cadc.schema import ACTION_FAMILY_REGISTRY, FeaturePerspective  # noqa: E402
from owl.cadc.support import SupportCalibrator  # noqa: E402


def _case_id(decision_id: str, horizon: int, category: str) -> str:
    body = f"owl.cadc.phase4.case.v1\x1f{decision_id}\x1f{horizon}\x1f{category}"
    return hashlib.sha256(body.encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cases-per-category", type=int, default=12)
    args = parser.parse_args()
    config = load_phase4_config(args.config)
    source = Path(args.input).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, object]] = []
    for path in sorted(source.glob("outer-*/calibrated_predictions.npz")):
        outer = int(path.parent.name.split("-", 1)[1])
        data = np.load(path, allow_pickle=False)
        horizon_values = data["horizons"].astype(np.int32)
        roles = data["split_roles"].astype(str)
        folds = data["outer_folds"].astype(np.int16)
        selected = (roles == "train") & (folds == outer)
        ids = data["decision_ids"].astype(str)[selected]
        score = data["rank_score"][selected]
        truth = data["target_scalar"][selected]
        mask = data["target_mask"][selected].astype(bool)
        disagreement = data["rank_disagreement"][selected]
        width = data["upper"][selected] - data["lower"][selected]
        survival = data["survival_probability"][selected]
        information = data["information_value"][selected]
        oracle_score = data["xgboost_oracle_rank"][selected]
        if not ids.size:
            continue
        negative = np.finfo(np.float64).min
        predicted_best = np.argmax(np.where(mask, score, negative), axis=2)
        true_best = np.argmax(np.where(mask, truth, negative), axis=2)
        rows = np.arange(ids.size)[:, None]
        horizons = np.arange(score.shape[1])[None, :]
        regret = truth[rows, horizons, true_best] - truth[rows, horizons, predicted_best]
        predicted_value = np.take_along_axis(
            score, predicted_best[..., None], axis=2
        )[..., 0]
        ordered = np.sort(np.where(mask, score, negative), axis=2)
        top_gap = ordered[..., -1] - ordered[..., -2]
        oracle_best = np.argmax(np.where(mask, oracle_score, negative), axis=2)
        hidden_oracle = (
            oracle_best != predicted_best
        ).astype(np.float64) * (
            np.take_along_axis(oracle_score, oracle_best[..., None], axis=2)[..., 0]
            - np.take_along_axis(
                oracle_score, predicted_best[..., None], axis=2
            )[..., 0]
        )
        predicted_survival = np.take_along_axis(
            survival, predicted_best[..., None], axis=2
        )[..., 0]
        diagnostics = {
            "high_regret": regret,
            "high_disagreement": np.max(np.where(mask, disagreement, negative), axis=2),
            "wide_interval": np.max(np.where(mask, width, negative), axis=2),
            "hidden_oracle_threat": hidden_oracle,
            "near_tie": 1.0 / np.maximum(top_gap, 1e-9),
            "severe_risk": 1.0 - predicted_survival,
            "high_information_value": np.max(information[..., [1, 11]], axis=2),
        }
        for family in sorted(
            {value.primary_family.value for value in ACTION_FAMILY_REGISTRY}
        ):
            family_mask = np.asarray(
                [
                    value.primary_family.value == family
                    for value in ACTION_FAMILY_REGISTRY
                ],
                dtype=bool,
            )
            diagnostics[f"action_family_{family}"] = np.max(
                np.where(mask[..., family_mask], score[..., family_mask], negative),
                axis=2,
            )
        del predicted_value
        calibration_manifest = json.loads(
            (path.parent / "calibration_manifest.json").read_text(encoding="utf-8")
        )
        support_index = np.load(calibration_manifest["support_index"], allow_pickle=False)
        support = SupportCalibrator(
            k=config.support.knn_k,
            minimum_seeds=config.support.minimum_seeds,
            minimum_decisions=config.support.minimum_decisions,
            minimum_repeats=config.support.minimum_repeats,
            maximum_disagreement=config.support.maximum_ensemble_disagreement,
            maximum_interval_width=config.support.maximum_conformal_width,
        ).fit(support_index["embeddings"], support_index["seeds"])
        training = (roles == "train") & (folds != outer)
        action_support = data["target_mask"][training].sum(axis=0)
        for category, diagnostic in diagnostics.items():
            flat = np.argsort(-diagnostic.reshape(-1), kind="stable")
            for item in flat[: args.cases_per_category]:
                row, horizon = np.unravel_index(item, diagnostic.shape)
                candidate_order = np.argsort(
                    -np.where(mask[row, horizon], score[row, horizon], negative),
                    kind="stable",
                )
                best_action = int(candidate_order[0])
                support_decision = support.decide(
                    data["embedding"][selected][row, horizon, best_action],
                    action_support=int(action_support[horizon, best_action]),
                    repeat_support=int(
                        data["repeat_count"][selected][row, horizon, best_action]
                    ),
                    disagreement=float(disagreement[row, horizon, best_action]),
                    interval_width=float(width[row, horizon, best_action]),
                )
                cases.append(
                    {
                        "case_id": _case_id(
                            ids[row], int(horizon_values[horizon]), category
                        ),
                        "outer_fold": outer,
                        "category": category,
                        "_source_decision_id": ids[row],
                        "source_decision_id_blind": hashlib.sha256(
                            ids[row].encode()
                        ).hexdigest(),
                        "horizon_slot": int(horizon),
                        "horizon": int(horizon_values[horizon]),
                        "predicted_action_order": [
                            int(value)
                            for value in candidate_order
                            if mask[row, horizon, value]
                        ],
                        "diagnostic_value": float(diagnostic[row, horizon]),
                        "predicted_best_action": best_action,
                        "predicted_best_action_family": ACTION_FAMILY_REGISTRY[
                            best_action
                        ].primary_family.value,
                        "support_status": support_decision.status.value,
                        "abstention_reason": support_decision.abstention_reason.value,
                        "outcome_labels_sealed": True,
                        "review_status": "unreviewed",
                    }
                )
    if not cases:
        raise FileNotFoundError("no held-out predictions available for casebook")
    try:
        import polars as pl
    except ImportError as exc:
        raise RuntimeError("casebook context projection requires Polars") from exc
    registry = FeatureRegistry()
    context_path = Path(args.dataset).resolve() / "canonical_data" / "decision_context"
    decision = pl.read_parquet(context_path)
    agent_names = [
        value.source_column
        for value in registry.for_perspective(FeaturePerspective.AGENT_PRIMARY)
        if value.source_table == "agent_context" and value.source_column in decision.columns
    ]
    oracle_names = [
        value.source_column
        for value in registry.for_perspective(FeaturePerspective.ORACLE_DIAGNOSTIC)
        if value.source_column in decision.columns
    ]
    execution_names = [
        value
        for value in ("selected_action", "attempted_action", "realized_action", "execution_success")
        if value in decision.columns
    ]
    wanted = sorted({str(value["_source_decision_id"]) for value in cases})
    rows = decision.filter(pl.col("source_decision_id").is_in(wanted)).select(
        "source_decision_id", *agent_names, *oracle_names, *execution_names
    ).to_dicts()
    lookup = {str(value["source_decision_id"]): value for value in rows}
    for case in cases:
        decision_id = str(case.pop("_source_decision_id"))
        row = lookup.get(decision_id)
        if row is None:
            raise RuntimeError("casebook source decision has no canonical context")
        case["agent_context"] = {name: row[name] for name in agent_names}
        case["oracle_context_diagnostic_only"] = {
            name: row[name] for name in oracle_names
        }
        case["execution_summary_postchoice"] = {
            name: row[name] for name in execution_names
        }
        case["condition_label_included"] = False
        case["mechanism_score_included"] = False
    cases.sort(key=lambda value: str(value["case_id"]))
    case_path = output / "blinded_casebook.jsonl"
    temporary = output / ".blinded_casebook.jsonl.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, sort_keys=True) + "\n")
    temporary.replace(case_path)
    challenge_entries: list[dict[str, object]] = []
    family_names = sorted(
        {value.primary_family.value for value in ACTION_FAMILY_REGISTRY}
    )
    for family in family_names:
        for case in cases:
            matching = [
                int(action)
                for action in case["predicted_action_order"]  # type: ignore[union-attr]
                if ACTION_FAMILY_REGISTRY[int(action)].primary_family.value == family
            ]
            if matching:
                challenge_entries.append(
                    {
                        "challenge_type": "action_family",
                        "action_family": family,
                        "challenge_action": matching[0],
                        "case_id": case["case_id"],
                    }
                )
                break
    for category in sorted({str(value["category"]) for value in cases}):
        selected_case = next(value for value in cases if value["category"] == category)
        challenge_entries.append(
            {
                "challenge_type": category,
                "case_id": selected_case["case_id"],
            }
        )
    execution_failure = next(
        (
            value
            for value in cases
            if value["execution_summary_postchoice"].get(  # type: ignore[union-attr]
                "execution_success"
            )
            is False
        ),
        None,
    )
    if execution_failure is not None:
        challenge_entries.append(
            {
                "challenge_type": "execution_failure",
                "case_id": execution_failure["case_id"],
            }
        )
    unsupported = next(
        (value for value in cases if value["support_status"] != "supported"), None
    )
    if unsupported is not None:
        challenge_entries.append(
            {
                "challenge_type": "insufficient_support",
                "case_id": unsupported["case_id"],
            }
        )
    challenge_path = output / "reviewer_challenge_set.jsonl"
    with challenge_path.open("w", encoding="utf-8") as handle:
        for item in challenge_entries:
            handle.write(json.dumps(item, sort_keys=True) + "\n")
    manifest = {
        "schema_version": "owl.cadc.phase4-casebook.v1",
        "passed": True,
        "config_sha256": config.canonical_digest(),
        "model_spec_sha256": config.model_spec_digest(),
        "case_count": len(cases),
        "cases_per_category": args.cases_per_category,
        "casebook_sha256": sha256_file(case_path),
        "reviewer_challenge_sha256": sha256_file(challenge_path),
        "reviewer_challenge_count": len(challenge_entries),
        "reviewer_challenge_family_coverage": sorted(
            {
                str(value["action_family"])
                for value in challenge_entries
                if "action_family" in value
            }
        ),
        "blinded": True,
        "agent_oracle_views_separate": True,
        "condition_and_mechanism_excluded": True,
        "phase5_locked": True,
    }
    atomic_json(output / "casebook_manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
