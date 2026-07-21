"""CLI interface for running parameter sweeps."""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path
from typing import Any

import yaml

from owl.analysis.compare_runs import parameter_sweep_heatmap
from owl.analysis.plots import _load_rows
from owl.core.config import SimulationConfig, load_config
from owl.experiments.presets import get_condition, list_conditions
from owl.experiments.run_single import run_single


def _load_sweep(path: str | Path) -> dict[str, Any]:
    """Load a YAML/JSON sweep specification."""
    sweep_path = Path(path)
    if not sweep_path.exists():
        raise FileNotFoundError(f"sweep specification not found: {sweep_path}")
    with sweep_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        raise ValueError(f"sweep specification is empty: {sweep_path}")
    if not isinstance(data, dict):
        raise TypeError(f"sweep root must be a mapping, got {type(data).__name__}")
    return data


def _set_nested(data: dict[str, Any], dotted_path: str, value: Any) -> None:
    """Set ``data[a][b]`` from a dotted path ``a.b``."""
    parts = str(dotted_path).split(".")
    if len(parts) < 2:
        raise ValueError(f"sweep parameter path must be dotted, got {dotted_path!r}")
    cursor = data
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            raise ValueError(f"unknown nested config section {part!r} in {dotted_path!r}")
        cursor = cursor[part]
    if parts[-1] not in cursor:
        raise ValueError(f"unknown config field {dotted_path!r}")
    cursor[parts[-1]] = value


def _parameter_items(spec: dict[str, Any]) -> list[tuple[str, list[Any]]]:
    """Return normalized sweep parameter items."""
    raw = spec.get("parameters", spec.get("sweep"))
    if raw is None:
        # Interpret a simple mapping as parameters unless all keys are reserved.
        reserved = {"conditions", "max_steps", "output_dir", "metric", "make_heatmap"}
        raw = {k: v for k, v in spec.items() if k not in reserved}
    if not isinstance(raw, dict) or not raw:
        raise ValueError("sweep specification must contain a nonempty 'parameters' mapping")

    items: list[tuple[str, list[Any]]] = []
    for key, values in raw.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"sweep parameter {key!r} must map to a nonempty list")
        items.append((str(key), list(values)))
    return items


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write sweep rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run_parameter_sweep(config_path: str | Path, sweep_path: str | Path) -> Path:
    """Run a configured parameter sweep.

    Sweep spec format
    -----------------
    ``parameters`` is a mapping from dotted config path to a list of values.
    Optional keys: ``conditions``, ``max_steps``, ``output_dir``, ``metric`` and
    ``make_heatmap``.

    Returns
    -------
    pathlib.Path
        Path to the sweep summary CSV.
    """
    base_cfg = load_config(config_path)
    spec = _load_sweep(sweep_path)
    items = _parameter_items(spec)
    conditions = spec.get("conditions", [None])
    if conditions is None:
        conditions = [None]
    if not isinstance(conditions, list):
        raise ValueError("sweep 'conditions' must be a list or null")

    output_dir = Path(spec.get("output_dir", "results/sweep"))
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    value_lists = [values for _, values in items]

    run_index = 0
    for combo in itertools.product(*value_lists):
        for condition in conditions:
            data = base_cfg.model_dump()
            for (param, _values), value in zip(items, combo, strict=True):
                _set_nested(data, param, value)
            if "max_steps" in spec:
                data["world"]["max_steps"] = int(spec["max_steps"])

            condition_name = "default" if condition is None else str(condition)
            cfg = SimulationConfig.model_validate(data)
            if condition is not None:
                # Validate condition early, then run_single applies it again to
                # preserve a single execution path. We save the condition-specific
                # config to avoid double-transforming.
                cfg = get_condition(condition_name, cfg)

            run_dir = output_dir / f"run_{run_index:04d}_{condition_name}"
            cfg.recording.metrics_path = str(run_dir / "metrics.csv")
            cfg.recording.zarr_path = str(run_dir / "run.zarr")
            cfg.visualization.enabled = False

            config_file = run_dir / "config.yaml"
            config_file.parent.mkdir(parents=True, exist_ok=True)
            config_file.write_text(
                yaml.safe_dump(cfg.model_dump(), sort_keys=False), encoding="utf-8"
            )

            metrics_path = run_single(config_file, condition=None)
            metrics = _load_rows(metrics_path)
            final = metrics[-1] if metrics else {}
            row: dict[str, Any] = {
                "run_index": run_index,
                "condition": condition_name,
                "metrics_path": str(metrics_path),
                "records": len(metrics),
                "final_tick": final.get("tick", 0),
                "final_alive_count": final.get("alive_count", 0),
                "final_global_integration": final.get("global_integration", 0.0),
                "mean_signal_total": float(
                    sum(float(m.get("signal_total", 0.0)) for m in metrics) / max(len(metrics), 1)
                ),
            }
            for idx, ((param, _values), value) in enumerate(
                zip(items, combo, strict=True), start=1
            ):
                row[f"sweep_param_{idx}"] = param
                row[f"sweep_value_{idx}"] = value
            rows.append(row)
            run_index += 1

    results_path = output_dir / "sweep_results.csv"
    _write_csv(rows, results_path)

    if spec.get("make_heatmap", False):
        metric = str(spec.get("metric", "final_global_integration"))
        parameter_sweep_heatmap(results_path, metric, output_dir / f"{metric}_heatmap.png")

    return results_path


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Run an Observer-Window Life parameter sweep.")
    parser.add_argument("config_path", help="Base config path.")
    parser.add_argument("sweep_path", help="YAML/JSON sweep spec path.")
    parser.add_argument(
        "--conditions", nargs="*", choices=list_conditions(), help="Override sweep conditions."
    )
    args = parser.parse_args()

    if args.conditions is not None:
        spec_path = Path(args.sweep_path)
        spec = _load_sweep(spec_path)
        spec["conditions"] = args.conditions
        tmp_path = spec_path.with_name(spec_path.stem + "_conditions_override.yaml")
        tmp_path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
        path = run_parameter_sweep(args.config_path, tmp_path)
    else:
        path = run_parameter_sweep(args.config_path, args.sweep_path)
    print(path)


if __name__ == "__main__":
    main()
