"""Experiment, condition, and analysis tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import yaml

from owl.analysis.compare_runs import (
    compare_conditions,
    load_metric_tables,
    parameter_sweep_heatmap,
)
from owl.analysis.plots import (
    make_animation_from_zarr,
    plot_global_integration,
    plot_population_by_trait,
    plot_signal_channel_totals,
)
from owl.analysis.zarr_reader import load_field, open_run_zarr
from owl.core.config import SimulationConfig, load_config
from owl.experiments.conditions import (
    make_baseline_condition,
    make_carnivore_condition,
    make_fragmented_condition,
    make_integrated_condition,
    make_overcoupled_condition,
    make_rivalry_condition,
)
from owl.experiments.presets import get_condition, list_conditions
from owl.experiments.run_single import run_single
from owl.experiments.run_sweep import run_parameter_sweep


def make_experiment_cfg(tmp_path: Path, *, max_steps: int = 3) -> Path:
    """Write a small deterministic experiment config and return its path."""
    data = load_config("configs/mvp.yaml").model_dump()
    data["world"]["height"] = 20
    data["world"]["width"] = 20
    data["world"]["patch_size"] = 5
    data["world"]["max_steps"] = max_steps
    data["initialization"]["population_density"] = 0.45
    data["initialization"]["food_patch_count"] = 1
    data["recording"]["enabled"] = True
    data["recording"]["record_every"] = 1
    data["recording"]["metrics_path"] = str(tmp_path / "metrics.csv")
    data["recording"]["zarr_path"] = str(tmp_path / "run.zarr")
    data["visualization"]["enabled"] = False
    data["debug"]["assert_invariants"] = True
    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return cfg_path


def test_conditions_are_deep_copied_valid_and_distinct() -> None:
    cfg = load_config("configs/mvp.yaml")
    makers = [
        make_baseline_condition,
        make_integrated_condition,
        make_rivalry_condition,
        make_fragmented_condition,
        make_overcoupled_condition,
        make_carnivore_condition,
    ]

    outputs = [maker(cfg) for maker in makers]
    assert all(isinstance(item, SimulationConfig) for item in outputs)
    assert cfg.phase.same_scale_coupling != outputs[0].phase.same_scale_coupling
    assert outputs[0].integration.weight_synchrony == 0.0
    assert outputs[1].integration.weight_synchrony >= cfg.integration.weight_synchrony
    assert outputs[2].actions.stochastic is True
    assert outputs[3].phase.phase_noise_sigma >= cfg.phase.phase_noise_sigma
    assert outputs[4].topdown.lambda_action_bias >= cfg.topdown.lambda_action_bias
    assert outputs[5].predation.enabled is True
    # Original config must not be mutated.
    assert cfg.integration.weight_synchrony != 0.0


def test_condition_lookup_lists_and_rejects_unknown() -> None:
    names = list_conditions()
    assert names == sorted(names)
    assert {"baseline", "integrated", "rivalry", "fragmented", "overcoupled", "carnivore"} <= set(
        names
    )

    cfg = load_config("configs/mvp.yaml")
    assert get_condition("baseline", cfg).integration.weight_synchrony == 0.0

    try:
        get_condition("not-real", cfg)
    except ValueError as exc:
        assert "unknown condition" in str(exc)
    else:
        raise AssertionError("unknown condition should fail")


def test_run_single_writes_metrics_summary_snapshot_and_recording(tmp_path: Path) -> None:
    cfg_path = make_experiment_cfg(tmp_path, max_steps=3)

    metrics_path = run_single(cfg_path, condition="baseline")

    assert metrics_path.exists()
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert rows[0]["condition"] == "baseline"

    summary = metrics_path.with_name(metrics_path.stem + "_summary.json")
    snapshot = metrics_path.with_name(metrics_path.stem + "_final_snapshot.npz")
    zarr_dir = tmp_path / "run.zarr"
    assert summary.exists()
    assert snapshot.exists()
    assert zarr_dir.exists()
    assert json.loads(summary.read_text())["condition"] == "baseline"

    run = open_run_zarr(zarr_dir)
    assert getattr(run, "backend", "zarr") in {"numpy-directory-fallback", "zarr"}
    health = load_field(zarr_dir, "state/health")
    assert health.ndim == 3
    assert health.shape[1:] == (20, 20)


def test_plots_and_animation_from_recording_outputs(tmp_path: Path) -> None:
    cfg_path = make_experiment_cfg(tmp_path, max_steps=3)
    metrics_path = run_single(cfg_path)
    zarr_path = tmp_path / "run.zarr"

    out1 = tmp_path / "global.png"
    out2 = tmp_path / "population.png"
    out3 = tmp_path / "signals.png"
    anim = tmp_path / "integration_frames.npz"

    plot_global_integration(metrics_path, out1)
    plot_population_by_trait(metrics_path, out2)
    plot_signal_channel_totals(metrics_path, out3)
    make_animation_from_zarr(zarr_path, "state/integration", anim)

    for path in (out1, out2, out3, anim):
        assert path.exists()
        assert path.stat().st_size > 0

    with np.load(anim) as archive:
        frames = archive["frames"]
        assert frames.ndim == 4
        assert frames.shape[-1] == 3


def test_compare_conditions_and_heatmap(tmp_path: Path) -> None:
    cfg_path = make_experiment_cfg(tmp_path / "a", max_steps=2)
    metrics_a = run_single(cfg_path, condition="baseline")
    cfg_path_b = make_experiment_cfg(tmp_path / "b", max_steps=2)
    metrics_b = run_single(cfg_path_b, condition="integrated")

    rows = load_metric_tables([metrics_a, metrics_b])
    assert rows
    assert {row["condition"] for row in rows} == {"baseline", "integrated"}

    comparison_csv = tmp_path / "comparison.csv"
    comparison_png = tmp_path / "comparison.png"
    compare_conditions([metrics_a, metrics_b], comparison_csv)
    compare_conditions([metrics_a, metrics_b], comparison_png)
    assert comparison_csv.exists()
    assert comparison_png.exists()

    sweep_rows = [
        {
            "sweep_param_1": "phase.same_scale_coupling",
            "sweep_value_1": 0.01,
            "final_global_integration": 0.2,
        },
        {
            "sweep_param_1": "phase.same_scale_coupling",
            "sweep_value_1": 0.05,
            "final_global_integration": 0.4,
        },
    ]
    sweep_csv = tmp_path / "sweep.csv"
    with sweep_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(sweep_rows[0]))
        writer.writeheader()
        writer.writerows(sweep_rows)

    heatmap = tmp_path / "heatmap.png"
    parameter_sweep_heatmap(sweep_csv, "final_global_integration", heatmap)
    assert heatmap.exists()
    assert heatmap.stat().st_size > 0


def test_run_parameter_sweep_writes_results_and_optional_heatmap(tmp_path: Path) -> None:
    cfg_path = make_experiment_cfg(tmp_path / "base", max_steps=2)
    sweep_path = tmp_path / "sweep.yaml"
    sweep_path.write_text(
        yaml.safe_dump(
            {
                "parameters": {"phase.same_scale_coupling": [0.0, 0.02]},
                "conditions": ["baseline"],
                "max_steps": 2,
                "output_dir": str(tmp_path / "sweep_out"),
                "make_heatmap": True,
                "metric": "final_global_integration",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    results = run_parameter_sweep(cfg_path, sweep_path)

    assert results.exists()
    with results.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["sweep_param_1"] == "phase.same_scale_coupling"
    assert (results.parent / "final_global_integration_heatmap.png").exists()


def test_analysis_io_errors_are_clear(tmp_path: Path) -> None:
    try:
        open_run_zarr(tmp_path / "missing.zarr")
    except FileNotFoundError as exc:
        assert "recorded run not found" in str(exc)
    else:
        raise AssertionError("missing zarr run should fail")

    empty_metrics = tmp_path / "empty.json"
    empty_metrics.write_text("[]", encoding="utf-8")
    try:
        plot_global_integration(empty_metrics, tmp_path / "bad.png")
    except ValueError as exc:
        assert "empty metrics" in str(exc)
    else:
        raise AssertionError("empty metrics should fail")
