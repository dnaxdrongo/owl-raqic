#!/usr/bin/env python3
"""Materialize compact, calibrated outer-fold candidate and decision scores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from owl.cadc.artifacts import atomic_json, sha256_file  # noqa: E402
from owl.cadc.config import load_phase4_config  # noqa: E402
from owl.cadc.inference import CADCScorer  # noqa: E402
from owl.cadc.scalarization import quantile_cvar  # noqa: E402
from owl.cadc.schema import ACTION_FAMILY_REGISTRY, PHASE4_SCORE_SCHEMA_VERSION  # noqa: E402
from owl.cadc.support import SupportCalibrator  # noqa: E402


def _fixed_list(values: np.ndarray) -> Any:
    import pyarrow as pa

    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError("fixed-list score column must be a matrix")
    return pa.FixedSizeListArray.from_arrays(
        pa.array(array.reshape(-1), from_pandas=False), array.shape[1]
    )


def _column(frame: Any, name: str, default: Any) -> np.ndarray:
    if name not in frame.columns:
        return np.full(frame.height, default)
    return frame[name].fill_null(default).to_numpy()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--calibration-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_phase4_config(args.config)
    source = Path(args.predictions).resolve()
    manifest_path = Path(args.calibration_manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data = np.load(source, allow_pickle=False)
    support_index = np.load(manifest["support_index"], allow_pickle=False)
    support = SupportCalibrator(
        k=config.support.knn_k,
        minimum_seeds=config.support.minimum_seeds,
        minimum_decisions=config.support.minimum_decisions,
        minimum_repeats=config.support.minimum_repeats,
        maximum_disagreement=config.support.maximum_ensemble_disagreement,
        maximum_interval_width=config.support.maximum_conformal_width,
    ).fit(support_index["embeddings"], support_index["seeds"])
    outer = int(manifest["outer_fold"])
    roles = data["split_roles"].astype(str)
    folds = data["outer_folds"].astype(np.int16)
    selected_decisions = (roles == "train") & (folds == outer)
    if not selected_decisions.any():
        raise ValueError("scored fold has no held-out outer-test decisions")
    decision_ids = data["decision_ids"].astype(str)[selected_decisions]
    rank = data["rank_score"].astype(np.float64)[selected_decisions]
    mask = data["target_mask"].astype(bool)[selected_decisions]
    embeddings = data["embedding"][selected_decisions]
    decisions, horizons, actions = rank.shape
    if actions != 22 or embeddings.shape[:3] != rank.shape:
        raise ValueError("scored predictions violate fixed decision/horizon/action axes")
    training = (roles == "train") & (folds != outer)
    action_support = np.broadcast_to(
        data["target_mask"][training].sum(axis=0)[None, :, :], rank.shape
    )
    scorer = CADCScorer(
        model=None,
        support=support,
        model_version=config.model_spec_digest(),
    )
    flat = scorer.score_candidates(
        {
            "scalar_value": rank.reshape(decisions * horizons, actions),
            "outcome_mean": data["outcome_mean"][selected_decisions].reshape(
                decisions * horizons, actions, -1
            ),
            "embedding": embeddings.reshape(decisions * horizons, actions, -1),
            "epistemic_disagreement": data["rank_disagreement"][
                selected_decisions
            ].reshape(decisions * horizons, actions),
        },
        executable_mask=mask.reshape(decisions * horizons, actions),
        lower=data["lower"][selected_decisions].reshape(
            decisions * horizons, actions
        ),
        upper=data["upper"][selected_decisions].reshape(
            decisions * horizons, actions
        ),
        action_support=action_support.reshape(decisions * horizons, actions),
        repeat_support=data["repeat_count"][selected_decisions].reshape(
            decisions * horizons, actions
        ),
    )
    try:
        import polars as pl
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("compact Phase 4 scoring requires Polars and PyArrow") from exc
    dataset_root = Path(args.dataset).resolve()
    requested = pl.DataFrame(
        {"source_decision_id": decision_ids, "score_order": np.arange(decisions)}
    )
    metadata = requested.join(
        pl.read_parquet(dataset_root / "canonical_data" / "decision_context").unique(
            subset=["source_decision_id"]
        ),
        on="source_decision_id",
        how="left",
        validate="1:1",
    ).sort("score_order")
    if metadata["run_id"].null_count():
        raise ValueError("score metadata join is incomplete")
    candidate_metadata = (
        pl.read_parquet(dataset_root / "canonical_data" / "candidate_context")
        .select(
            "source_decision_id",
            "action_index",
            "policy_legal",
            "prechoice_executable",
        )
        .join(requested, on="source_decision_id", how="inner", validate="m:1")
        .sort("score_order", "action_index")
    )
    if candidate_metadata.height != decisions * 22:
        raise ValueError("score candidate metadata does not contain exactly 22 actions")
    policy_legal = candidate_metadata["policy_legal"].to_numpy()
    prechoice_executable = candidate_metadata["prechoice_executable"].to_numpy()
    executable_by_decision = (policy_legal & prechoice_executable).reshape(decisions, 22)
    if not np.array_equal(mask.any(axis=1), executable_by_decision):
        raise ValueError("scored executable mask differs from factual pre-choice evidence")

    d_index = np.repeat(np.arange(decisions), horizons * actions)
    h_index = np.tile(np.repeat(np.arange(horizons), actions), decisions)
    a_index = np.tile(np.arange(actions), decisions * horizons)
    row_count = d_index.size
    run_id = metadata["run_id"].cast(pl.String).to_numpy()[d_index]
    seed = metadata["seed"].to_numpy()[d_index]
    tick = metadata["tick"].to_numpy()[d_index]
    sequence = metadata["decision_sequence"].to_numpy()[d_index]
    ow_id = metadata["ow_id"].to_numpy()[d_index]
    horizon_values = data["horizons"].astype(np.int32)
    if len(horizon_values) != horizons:
        raise ValueError("prediction horizon axis differs from configured base horizons")
    horizon_tick = horizon_values[h_index]
    rank_flat = rank.reshape(-1)
    selected_action_decision = data["selected_actions"][selected_decisions].astype(np.int16)
    selected_score = rank[
        np.arange(decisions)[:, None],
        np.arange(horizons)[None, :],
        np.broadcast_to(selected_action_decision[:, None], (decisions, horizons)),
    ]
    pair_probability = 1.0 / (
        1.0 + np.exp(np.clip(-(rank - selected_score[..., None]), -60.0, 60.0))
    )
    valid_count = mask.sum(axis=2)
    percentile = np.where(
        mask,
        1.0
        - (flat.predicted_rank.reshape(decisions, horizons, actions) - 1)
        / np.maximum(valid_count[..., None] - 1, 1),
        np.nan,
    )
    outcome = data["outcome_mean"][selected_decisions]
    quantiles = data["return_quantiles"][selected_decisions]
    cvar = quantile_cvar(
        quantiles,
        config.scalarization.quantiles,
        alpha=config.scalarization.cvar_alpha,
    )
    cause = data["cause_probability"][selected_decisions]
    externality = data["externality_prediction"][selected_decisions]
    aleatoric = np.exp(
        2.0 * np.clip(data["outcome_log_scale"][selected_decisions], -12.0, 8.0)
    ).mean(axis=-1)
    candidate_table = pa.table(
        {
            "run_id": pa.array(run_id),
            "seed": pa.array(seed),
            "tick": pa.array(tick),
            "decision_sequence": pa.array(sequence),
            "ow_id": pa.array(ow_id),
            "source_decision_id": pa.array(decision_ids[d_index]),
            "horizon": pa.array(horizon_tick.astype(np.int32)),
            "action_index": pa.array(a_index.astype(np.int16)),
            "action_family": pa.array(
                [
                    ACTION_FAMILY_REGISTRY[value].primary_family.value
                    for value in a_index
                ]
            ),
            "executable": pa.array(mask.reshape(-1)),
            "support_status": pa.array(flat.support_status.reshape(-1).astype(str)),
            "abstention_reason": pa.array(
                flat.abstention_reason.reshape(-1).astype(str)
            ),
            "agent_outcome_vector": _fixed_list(
                outcome.reshape(row_count, outcome.shape[-1]).astype(np.float32)
            ),
            "oracle_outcome_vector": pa.nulls(
                row_count, type=pa.list_(pa.float32(), outcome.shape[-1])
            ),
            "oracle_outcome_status": pa.array(
                np.full(row_count, "unsupported_evidence")
            ),
            "agent_q_risk_averse": pa.array(rank_flat.astype(np.float32)),
            "oracle_q_diagnostic": pa.array(
                data["xgboost_oracle_rank"][selected_decisions]
                .reshape(-1)
                .astype(np.float32)
            ),
            "survival_probability": pa.array(
                data["survival_probability"][selected_decisions]
                .reshape(-1)
                .astype(np.float32)
            ),
            "cause_risks": _fixed_list(
                cause.reshape(row_count, cause.shape[-1]).astype(np.float32)
            ),
            "return_quantiles": _fixed_list(
                quantiles.reshape(row_count, quantiles.shape[-1]).astype(np.float32)
            ),
            "lower_tail_cvar": pa.array(cvar.reshape(-1).astype(np.float32)),
            "information_value": pa.array(
                data["information_value"][selected_decisions]
                .reshape(-1)
                .astype(np.float32)
            ),
            "externality_vector": _fixed_list(
                externality.reshape(row_count, externality.shape[-1]).astype(np.float32)
            ),
            "pairwise_win_probability_vs_selected": pa.array(
                pair_probability.reshape(-1).astype(np.float32)
            ),
            "predicted_rank": pa.array(flat.predicted_rank.reshape(-1)),
            "predicted_percentile": pa.array(percentile.reshape(-1).astype(np.float32)),
            "prediction_lower": pa.array(flat.lower.reshape(-1).astype(np.float32)),
            "prediction_upper": pa.array(flat.upper.reshape(-1).astype(np.float32)),
            "ensemble_disagreement": pa.array(
                data["rank_disagreement"][selected_decisions]
                .reshape(-1)
                .astype(np.float32)
            ),
            "aleatoric_variance_mean": pa.array(
                aleatoric.reshape(-1).astype(np.float32)
            ),
            "model_release_id": pa.array(
                np.full(row_count, config.model_spec_digest())
            ),
        }
    )
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    candidate_path = output / "candidate_scores_compact.parquet"
    pq.write_table(candidate_table, candidate_path, compression="zstd", row_group_size=65536)

    score_matrix = rank
    negative = np.finfo(np.float64).min
    agent_best = np.argmax(np.where(mask, score_matrix, negative), axis=2)
    oracle_score = data["xgboost_oracle_rank"][selected_decisions]
    oracle_best = np.argmax(np.where(mask, oracle_score, negative), axis=2)
    selected_grid = np.broadcast_to(selected_action_decision[:, None], (decisions, horizons))
    row = np.arange(decisions)[:, None]
    horizon_row = np.arange(horizons)[None, :]
    selected_predicted_rank = flat.predicted_rank.reshape(
        decisions, horizons, actions
    )[row, horizon_row, selected_grid]
    agent_regret = (
        score_matrix[row, horizon_row, agent_best]
        - score_matrix[row, horizon_row, selected_grid]
    )
    information_regret = (
        oracle_score[row, horizon_row, oracle_best]
        - oracle_score[row, horizon_row, agent_best]
    )
    oracle_total_regret = (
        oracle_score[row, horizon_row, oracle_best]
        - oracle_score[row, horizon_row, selected_grid]
    )
    selected_support = flat.support_status.reshape(decisions, horizons, actions)[
        row, horizon_row, selected_grid
    ]
    selected_reason = flat.abstention_reason.reshape(decisions, horizons, actions)[
        row, horizon_row, selected_grid
    ]
    selected_survival = data["survival_probability"][selected_decisions][
        row, horizon_row, selected_grid
    ]
    sensitivity = (
        np.argmax(np.where(mask, data["lower"][selected_decisions], negative), axis=2)
        == np.argmax(
            np.where(mask, data["upper"][selected_decisions], negative), axis=2
        )
    )
    decision_rows = decisions * horizons
    d2 = np.repeat(np.arange(decisions), horizons)
    attempted = _column(metadata, "attempted_action", -1)
    realized = _column(metadata, "realized_action", -1)
    success = _column(metadata, "execution_success", False).astype(bool)
    execution_fidelity = (
        (attempted == selected_action_decision)
        & (realized == selected_action_decision)
        & success
    )
    decision_table = pa.table(
        {
            "run_id": pa.array(metadata["run_id"].cast(pl.String).to_numpy()[d2]),
            "seed": pa.array(metadata["seed"].to_numpy()[d2]),
            "tick": pa.array(metadata["tick"].to_numpy()[d2]),
            "decision_sequence": pa.array(metadata["decision_sequence"].to_numpy()[d2]),
            "ow_id": pa.array(metadata["ow_id"].to_numpy()[d2]),
            "source_decision_id": pa.array(decision_ids[d2]),
            "horizon": pa.array(np.tile(horizon_values, decisions).astype(np.int32)),
            "selected_action": pa.array(selected_grid.reshape(-1).astype(np.int16)),
            "agent_best_action": pa.array(agent_best.reshape(-1).astype(np.int16)),
            "oracle_best_action": pa.array(oracle_best.reshape(-1).astype(np.int16)),
            "selected_rank": pa.array(selected_predicted_rank.reshape(-1)),
            "selected_percentile": pa.array(
                percentile[row, horizon_row, selected_grid].reshape(-1).astype(np.float32)
            ),
            "agent_decision_regret": pa.array(agent_regret.reshape(-1).astype(np.float32)),
            "information_regret": pa.array(
                information_regret.reshape(-1).astype(np.float32)
            ),
            "oracle_total_regret": pa.array(
                oracle_total_regret.reshape(-1).astype(np.float32)
            ),
            "execution_fidelity": pa.array(execution_fidelity[d2]),
            "survival_probability_selected": pa.array(
                selected_survival.reshape(-1).astype(np.float32)
            ),
            "risk_status": pa.array(
                np.where(selected_survival.reshape(-1) < 0.5, "elevated", "nominal")
            ),
            "support_status": pa.array(selected_support.reshape(-1).astype(str)),
            "abstention_reason": pa.array(selected_reason.reshape(-1).astype(str)),
            "component_vector": _fixed_list(
                outcome[row, horizon_row, selected_grid]
                .reshape(decision_rows, outcome.shape[-1])
                .astype(np.float32)
            ),
            "sensitivity_stable": pa.array(sensitivity.reshape(-1)),
            "model_release_id": pa.array(
                np.full(decision_rows, config.model_spec_digest())
            ),
        }
    )
    decision_path = output / "decision_scores_compact.parquet"
    pq.write_table(decision_table, decision_path, compression="zstd", row_group_size=65536)
    scalar_figure = pl.from_arrow(
        candidate_table.select(
            [
                "seed",
                "horizon",
                "action_index",
                "action_family",
                "executable",
                "support_status",
                "agent_q_risk_averse",
                "lower_tail_cvar",
                "survival_probability",
                "ensemble_disagreement",
            ]
        )
    )
    scalar_figure.write_csv(output / "candidate_score_figure_data.csv")
    receipt = {
        "schema_version": PHASE4_SCORE_SCHEMA_VERSION,
        "passed": True,
        "outer_fold": outer,
        "heldout_decisions": decisions,
        "candidate_rows": candidate_table.num_rows,
        "decision_rows": decision_table.num_rows,
        "candidate_sha256": sha256_file(candidate_path),
        "decision_sha256": sha256_file(decision_path),
        "prediction_sha256": sha256_file(source),
        "calibration_manifest_sha256": sha256_file(manifest_path),
        "model_spec_sha256": config.model_spec_digest(),
        "support_backend": support.neighbor_backend,
        "oracle_outcome_status": "unsupported_evidence",
        "phase5_locked": True,
    }
    atomic_json(output / "score_receipt.json", receipt)
    print(candidate_path)
    print(decision_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
