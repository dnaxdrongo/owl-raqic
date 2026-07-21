#!/usr/bin/env python3
"""Export a bounded, checksum-complete CADC-MORE 2 review package."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from owl.cadc.artifacts import atomic_json, sha256_file  # noqa: E402
from owl.cadc.config import load_phase4_config  # noqa: E402

_REVIEW_FILES = (
    "phase4_certificate.json",
    "repeat_pilot.json",
    "performance.json",
    "environment.json",
    "math_verification.json",
    "synthetic_scenarios.json",
    "hotpath_audit.json",
    "negative_controls.json",
    "evaluation/evaluation.json",
    "models/training_receipt.json",
    "calibration/calibration_receipt.json",
    "scored_artifacts/scored_artifacts_receipt.json",
    "dataset/dataset_build_receipt.json",
    "dataset/manifests/feature_registry.json",
    "dataset/manifests/outcome_registry.json",
    "dataset/manifests/split_registry.json",
    "dataset/manifests/evidence_catalog.json",
    "corpus/corpus_plan.json",
    "corpus/corpus_certificate.json",
    "casebook/casebook_manifest.json",
    "casebook/blinded_casebook.jsonl",
    "casebook/reviewer_challenge_set.jsonl",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_phase4_config(args.config)
    source = Path(args.run).resolve()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()) and not config.artifacts.overwrite:
        raise FileExistsError(f"bounded export already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    receipts = []
    total = 0
    for relative in _REVIEW_FILES:
        origin = source / relative
        if not origin.is_file():
            continue
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, destination)
        size = destination.stat().st_size
        total += size
        if total > config.artifacts.local_export_max_bytes:
            raise RuntimeError("local review export exceeds configured byte bound")
        receipts.append(
            {"path": relative, "bytes": size, "sha256": sha256_file(destination)}
        )
    scored = source / "scored_artifacts"
    compact = (
        "candidate_scores_compact.parquet",
        "decision_scores_compact.parquet",
    )
    for name in compact:
        origin = scored / name
        if not origin.is_file():
            raise FileNotFoundError(f"required compact scored artifact is missing: {origin}")
        destination = output / name
        shutil.copyfile(origin, destination)
        size = destination.stat().st_size
        total += size
        receipts.append({"path": name, "bytes": size, "sha256": sha256_file(destination)})
    try:
        import polars as pl
    except ImportError as exc:
        raise RuntimeError("local Phase 4 export requires Polars") from exc
    figure_root = output / "figure_data"
    metric_root = output / "metric_tables"
    figure_root.mkdir(parents=True, exist_ok=True)
    metric_root.mkdir(parents=True, exist_ok=True)
    candidate = pl.read_parquet(output / "candidate_scores_compact.parquet")
    decision = pl.read_parquet(output / "decision_scores_compact.parquet")
    candidate_figure = candidate.select(
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
    )
    candidate_figure.write_csv(figure_root / "candidate_scores.csv")
    decision.select(
        "seed",
        "horizon",
        "selected_action",
        "agent_best_action",
        "selected_rank",
        "agent_decision_regret",
        "information_regret",
        "oracle_total_regret",
        "execution_fidelity",
        "risk_status",
        "support_status",
    ).write_csv(figure_root / "decision_scores.csv")
    support_summary = candidate.group_by(
        "action_family", "horizon", "support_status"
    ).agg(
        pl.len().alias("candidate_rows"),
        pl.col("seed").n_unique().alias("independent_seeds"),
        pl.col("ensemble_disagreement").mean().alias("mean_disagreement"),
    ).sort("action_family", "horizon", "support_status")
    support_summary.write_parquet(output / "support_summary.parquet", compression="zstd")
    support_summary.write_csv(metric_root / "support_summary.csv")
    evaluation_payload = json.loads(
        (source / "evaluation" / "evaluation.json").read_text(encoding="utf-8")
    )
    pl.DataFrame(
        [
            {
                "outer_fold": value["outer_fold"],
                "forecast_mae": value["forecast"]["mae"],
                "top1_accuracy": value["ranking"]["top1_accuracy"],
                "mean_regret": value["ranking"]["mean_regret"],
                "survival_brier": value["survival"]["brier"],
                "conformal_coverage": value["conformal_coverage"],
            }
            for value in evaluation_payload["folds"]
        ]
    ).write_csv(metric_root / "outer_fold_metrics.csv")
    calibration_rows = []
    for manifest_path in sorted((source / "calibration").glob("outer-*/calibration_manifest.json")):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        calibration_rows.append(
            {
                "outer_fold": payload["outer_fold"],
                "temperature": payload["temperature"],
                "coverage": payload["coverage"],
                "global_radius": payload["conformal_global_radius"],
                "isotonic_status": payload["isotonic_value"]["status"],
            }
        )
    pl.DataFrame(calibration_rows).write_parquet(
        output / "calibration_curves.parquet", compression="zstd"
    )
    shutil.copyfile(
        source / "dataset" / "manifests" / "feature_registry.json",
        output / "feature_dictionary.json",
    )
    shutil.copyfile(
        source / "dataset" / "manifests" / "outcome_registry.json",
        output / "outcome_dictionary.json",
    )
    model_cards = sorted((source / "models").glob("outer-*/model_card.md"))
    if not model_cards:
        raise FileNotFoundError("Phase 4 model card is missing")
    shutil.copyfile(model_cards[0], output / "model_card.md")
    shutil.copyfile(
        source / "casebook" / "blinded_casebook.jsonl", output / "casebook.jsonl"
    )
    generated = [
        *sorted(figure_root.glob("*.csv")),
        *sorted(metric_root.glob("*.csv")),
        output / "support_summary.parquet",
        output / "calibration_curves.parquet",
        output / "feature_dictionary.json",
        output / "outcome_dictionary.json",
        output / "model_card.md",
        output / "casebook.jsonl",
    ]
    for destination in generated:
        size = destination.stat().st_size
        total += size
        receipts.append(
            {
                "path": str(destination.relative_to(output)),
                "bytes": size,
                "sha256": sha256_file(destination),
            }
        )
    if total > config.artifacts.local_export_max_bytes:
        raise RuntimeError("local review export exceeds configured byte bound")
    required = {"phase4_certificate.json", "evaluation/evaluation.json"}
    present = {value["path"] for value in receipts}
    if not required.issubset(present):
        raise FileNotFoundError("required Phase 4 review artifacts are missing")
    atomic_json(
        output / "LOCAL_EXPORT_MANIFEST.json",
        {
            "schema_version": "owl.cadc.phase4-local-export.v1",
            "passed": True,
            "source_run": str(source),
            "config_sha256": config.canonical_digest(),
            "model_spec_sha256": config.model_spec_digest(),
            "total_bytes": total,
            "files": receipts,
            "excludes_raw_corpus_and_model_weights": True,
            "contains_restyleable_csv_and_parquet": True,
            "phase5_locked": True,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
