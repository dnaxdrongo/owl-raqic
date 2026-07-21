#!/usr/bin/env python3
"""Fit untouched calibration-fold probability, conformal, and support layers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from owl.cadc.artifacts import atomic_json, sha256_file  # noqa: E402
from owl.cadc.calibration import (  # noqa: E402
    ConformalQuantileCalibrator,
    IsotonicValueCalibrator,
    TemperatureCalibrator,
)
from owl.cadc.config import load_phase4_config  # noqa: E402
from owl.cadc.support import SupportCalibrator  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_phase4_config(args.config)
    source = Path(args.input).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    receipts = []
    for prediction_path in sorted(source.glob("outer-*/heldout_predictions.npz")):
        outer_fold = int(prediction_path.parent.name.split("-", 1)[1])
        data = np.load(prediction_path, allow_pickle=False)
        roles = data["split_roles"].astype(str)
        folds = data["outer_folds"].astype(np.int16)
        target_mask = data["target_mask"].astype(bool)
        calibration_decisions = roles == "calibration"
        if not calibration_decisions.any():
            raise ValueError(f"outer fold {outer_fold} has no calibration decisions")
        ensemble_rank = data["rank_score"].mean(axis=0, dtype=np.float64)
        disagreement = data["rank_score"].var(axis=0, dtype=np.float64)
        ensemble_survival = data["survival_probability"].mean(
            axis=0, dtype=np.float64
        )
        ensemble_cause = data["cause_probability"].mean(axis=0, dtype=np.float64)
        ensemble_information = data["information_value"].mean(
            axis=0, dtype=np.float64
        )
        ensemble_information_components = data["information_components"].mean(
            axis=0, dtype=np.float64
        )
        ensemble_outcome = data["outcome_mean"].mean(axis=0, dtype=np.float64)
        outcome_disagreement = data["outcome_mean"].var(axis=0, dtype=np.float64)
        ensemble_log_scale = data["outcome_log_scale"].mean(
            axis=0, dtype=np.float64
        )
        ensemble_externality = data["externality_prediction"].mean(
            axis=0, dtype=np.float64
        )
        ensemble_quantiles = data["return_quantiles"].mean(
            axis=0, dtype=np.float64
        )
        target_scalar = data["target_scalar"].astype(np.float64)
        target_alive = data["target_alive"].astype(np.float64)
        target_death_cause_probability = data[
            "target_death_cause_probability"
        ].astype(np.float64)
        target_death_cause = np.argmax(target_death_cause_probability, axis=-1)
        horizon_count = ensemble_rank.shape[1]
        action_count = ensemble_rank.shape[2]
        horizon_values = data["horizons"].astype(np.int32)
        if horizon_values.size != horizon_count:
            raise ValueError("calibration horizon registry does not match prediction axis")
        group = np.broadcast_to(
            np.asarray(
                [
                    f"h{int(horizon)}:a{action}"
                    for horizon in horizon_values
                    for action in range(action_count)
                ]
            ).reshape(1, horizon_count, action_count),
            ensemble_rank.shape,
        )
        calibration_mask = calibration_decisions[:, None, None] & target_mask
        isotonic = IsotonicValueCalibrator(
            minimum_rows=config.calibration.isotonic_minimum
        ).fit(
            ensemble_rank[calibration_mask],
            target_scalar[calibration_mask],
        )
        ensemble_rank = isotonic.transform(ensemble_rank)
        conformal = ConformalQuantileCalibrator(
            coverage=config.calibration.interval_level,
            minimum_group=config.calibration.mondrian_minimum,
        ).fit(
            ensemble_rank[calibration_mask],
            target_scalar[calibration_mask],
            group[calibration_mask],
        )
        lower, upper = conformal.interval(ensemble_rank, group)
        del ensemble_survival
        probability = np.clip(ensemble_cause[calibration_mask], 1e-6, 1.0)
        probability /= probability.sum(axis=-1, keepdims=True)
        logits = np.log(probability)
        temperature = TemperatureCalibrator().fit(
            logits, target_death_cause[calibration_mask]
        )
        all_probability = np.clip(ensemble_cause, 1e-6, 1.0)
        all_probability /= all_probability.sum(axis=-1, keepdims=True)
        all_logits = np.log(all_probability)
        calibrated_logits = temperature.transform(all_logits.reshape(-1, 5)).reshape(
            *all_probability.shape
        )
        shifted = calibrated_logits - calibrated_logits.max(axis=-1, keepdims=True)
        calibrated_cause = np.exp(shifted) / np.exp(shifted).sum(
            axis=-1, keepdims=True
        )
        calibrated_survival = calibrated_cause[..., 0]
        embedding = data["embedding"].astype(np.float32)
        seeds = data["seeds"].astype(np.int64)
        training_decisions = (roles == "train") & (folds != outer_fold)
        train_mask = training_decisions[:, None, None] & target_mask
        training_embeddings = embedding[train_mask]
        training_seeds = np.broadcast_to(
            seeds[:, None, None], target_mask.shape
        )[train_mask]
        maximum_index_rows = 50_000
        if training_embeddings.shape[0] > maximum_index_rows:
            rng = np.random.default_rng(config.master_seed + outer_fold)
            selected = np.sort(
                rng.choice(training_embeddings.shape[0], maximum_index_rows, replace=False)
            )
            training_embeddings = training_embeddings[selected]
            training_seeds = training_seeds[selected]
        support = SupportCalibrator(
            k=config.support.knn_k,
            minimum_seeds=config.support.minimum_seeds,
            minimum_decisions=config.support.minimum_decisions,
            minimum_repeats=config.support.minimum_repeats,
            maximum_disagreement=config.support.maximum_ensemble_disagreement,
            maximum_interval_width=config.support.maximum_conformal_width,
        ).fit(training_embeddings, training_seeds)
        index_path = output / f"outer-{outer_fold}-support-index.npz"
        np.savez_compressed(
            index_path, embeddings=training_embeddings, seeds=training_seeds
        )
        fold_root = output / f"outer-{outer_fold}"
        fold_root.mkdir(parents=True, exist_ok=True)
        calibrated_path = fold_root / "calibrated_predictions.npz"
        np.savez_compressed(
            calibrated_path,
            decision_ids=data["decision_ids"],
            horizons=data["horizons"],
            split_roles=roles,
            outer_folds=folds,
            target_mask=target_mask,
            repeat_count=data["repeat_count"],
            target_scalar=target_scalar,
            target_outcomes=data["target_outcomes"],
            target_outcome_variance=data["target_outcome_variance"],
            target_scalar_quantiles=data["target_scalar_quantiles"],
            target_scalar_cvar=data["target_scalar_cvar"],
            target_alive=target_alive,
            target_death_cause_probability=target_death_cause_probability,
            rank_score=ensemble_rank,
            rank_disagreement=disagreement,
            lower=lower,
            upper=upper,
            survival_probability=calibrated_survival,
            cause_probability=calibrated_cause,
            embedding=embedding,
            seeds=seeds,
            selected_actions=data["selected_actions"],
            neural_viability_baseline=data["neural_viability_baseline"],
            xgboost_viability_baseline=data["xgboost_viability_baseline"],
            xgboost_survival_baseline=data["xgboost_survival_baseline"],
            action_agnostic_target=data["action_agnostic_target"],
            xgboost_agent_rank=data["xgboost_agent_rank"],
            xgboost_oracle_rank=data["xgboost_oracle_rank"],
            information_value=ensemble_information,
            information_components=ensemble_information_components,
            outcome_mean=ensemble_outcome,
            outcome_disagreement=outcome_disagreement,
            outcome_log_scale=ensemble_log_scale,
            externality_prediction=ensemble_externality,
            return_quantiles=ensemble_quantiles,
        )
        manifest = {
            "schema_version": "owl.cadc.phase4-calibration.v1",
            "outer_fold": outer_fold,
            "temperature": temperature.temperature,
            "temperature_classes": [
                "no_event",
                "starvation",
                "toxin",
                "other_observed",
                "ambiguous_or_absent",
            ],
            "isotonic_value": isotonic.manifest(),
            "coverage": config.calibration.interval_level,
            "conformal_global_radius": conformal.global_radius,
            "conformal_group_radius": conformal.group_radius,
            "support": support.manifest(),
            "support_index": str(index_path),
            "support_index_sha256": sha256_file(index_path),
            "prediction_sha256": sha256_file(calibrated_path),
            "calibration_decisions": int(calibration_decisions.sum()),
            "phase5_locked": True,
        }
        atomic_json(fold_root / "calibration_manifest.json", manifest)
        receipts.append(manifest)
    if not receipts:
        raise FileNotFoundError("no held-out training predictions were found")
    atomic_json(
        output / "calibration_receipt.json",
        {
            "schema_version": "owl.cadc.phase4-calibration-receipt.v1",
            "passed": True,
            "folds": receipts,
            "phase5_locked": True,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
