#!/usr/bin/env python3
"""Measure stochastic repeat stability without changing the frozen policy."""

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
from owl.cadc.config import load_phase4_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_phase4_config(args.config)
    if not isinstance(config.corpus.repeat_policy, int):
        raise RuntimeError("repeat pilot analysis requires a resolved integer policy")
    try:
        import polars as pl
    except ImportError as exc:
        raise RuntimeError("repeat pilot analysis requires Polars") from exc
    dataset_root = Path(args.dataset).resolve()
    target_path = dataset_root / "canonical_data" / "branch_targets"
    target = pl.read_parquet(target_path).select(
        "source_decision_id",
        "forced_action",
        "horizon",
        "repeat_index",
        "agent_risk_averse",
        "death_by_horizon",
    )
    selected = int(config.corpus.repeat_policy)
    levels = tuple(value for value in config.corpus.repeat_pilot if value <= selected)
    if selected not in levels:
        levels = (*levels, selected)
    levels = tuple(sorted(set(levels)))
    key = ["source_decision_id", "forced_action", "horizon"]
    reference = target.group_by(*key).agg(
        pl.col("agent_risk_averse").mean().alias("reference_value"),
        pl.col("death_by_horizon").cast(pl.Float64).mean().alias("reference_death"),
    )
    reference_best = reference.group_by("source_decision_id", "horizon").agg(
        pl.col("forced_action")
        .sort_by("reference_value", descending=True)
        .first()
        .alias("reference_best")
    )
    metrics = []
    for repeats in levels:
        prefix = (
            target.filter(pl.col("repeat_index") < repeats)
            .group_by(*key)
            .agg(
                pl.len().alias("repeat_count"),
                pl.col("agent_risk_averse").mean().alias("value"),
                pl.col("agent_risk_averse").std(ddof=1).fill_null(0.0).alias("std"),
                pl.col("death_by_horizon").cast(pl.Float64).mean().alias("death"),
            )
        )
        if prefix.filter(pl.col("repeat_count") != repeats).height:
            raise ValueError("repeat prefix is incomplete for at least one branch group")
        joined = prefix.join(reference, on=key, how="inner", validate="1:1").with_columns(
            (pl.col("value") - pl.col("reference_value")).abs().alias("value_error"),
            (pl.col("death") - pl.col("reference_death")).abs().alias("death_error"),
        )
        best = joined.group_by("source_decision_id", "horizon").agg(
            pl.col("forced_action")
            .sort_by("value", descending=True)
            .first()
            .alias("best")
        ).join(
            reference_best,
            on=["source_decision_id", "horizon"],
            how="inner",
            validate="1:1",
        )
        standard_error = joined["std"].to_numpy() / np.sqrt(float(repeats))
        metrics.append(
            {
                "repeats": repeats,
                "branch_groups": joined.height,
                "mean_absolute_value_drift_vs_selected": float(
                    joined["value_error"].mean()
                ),
                "p95_absolute_value_drift_vs_selected": float(
                    joined["value_error"].quantile(0.95, interpolation="linear")
                ),
                "mean_death_probability_drift_vs_selected": float(
                    joined["death_error"].mean()
                ),
                "mean_standard_error": float(np.mean(standard_error, dtype=np.float64)),
                "best_action_stability_vs_selected": float(
                    (best["best"] == best["reference_best"]).mean()
                ),
            }
        )
    output = Path(args.output).resolve()
    atomic_json(
        output,
        {
            "schema_version": "owl.cadc.phase4-repeat-pilot.v1",
            "passed": True,
            "classification": "REPEAT_STABILITY_MEASURED_DESCRIPTIVE",
            "selected_repeat_policy": selected,
            "evaluated_prefixes": list(levels),
            "metrics": metrics,
            "selection_rule": (
                "repeat policy is pre-registered in the development config; pilot "
                "metrics are descriptive until Phase 5 freeze gates are approved"
            ),
            "dataset_receipt_sha256": sha256_file(
                dataset_root / "dataset_build_receipt.json"
            ),
            "corpus_contract_sha256": config.corpus_digest(),
            "model_spec_sha256": config.model_spec_digest(),
            "phase5_locked": True,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
