"""Comparative analysis interfaces for experiment batches."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from owl.analysis.plots import _load_rows


def _infer_condition(path: Path, rows: list[dict[str, Any]]) -> str:
    """Infer a condition name from rows or parent directory."""
    if rows and rows[0].get("condition") is not None:
        return str(rows[0]["condition"])
    return path.parent.name or path.stem


def load_metric_tables(paths: list[str | Path]) -> Any:
    """Load multiple metrics tables.

    Parameters
    ----------
    paths:
        List of metrics paths in JSON, JSONL, CSV, Parquet, or fallback-Parquet
        JSON-lines format.

    Returns
    -------
    list[dict]
        Combined scalar metric rows with added ``source_path`` and ``condition``
        fields when absent.
    """
    if not paths:
        raise ValueError("paths cannot be empty")

    rows_out: list[dict[str, Any]] = []
    for item in paths:
        path = Path(item)
        rows = _load_rows(path)
        condition = _infer_condition(path, rows)
        for row in rows:
            clean = dict(row)
            clean.setdefault("condition", condition)
            clean["source_path"] = str(path)
            rows_out.append(clean)
    return rows_out


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write rows to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def compare_conditions(metric_paths: list[str | Path], out_path: str | Path) -> None:
    """Compare scalar diagnostics across named conditions.

    Writes a CSV summary and, when ``out_path`` has an image extension, a bar
    plot of final global integration and final alive count. For ``.csv`` paths,
    only the summary table is written.
    """
    rows = load_metric_tables(metric_paths)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("condition", "unknown"))].append(row)

    summary: list[dict[str, Any]] = []
    for condition, cond_rows in sorted(grouped.items()):
        # Sort by tick when available so "final" is deterministic.
        cond_rows = sorted(cond_rows, key=lambda r: float(r.get("tick", len(cond_rows))))
        final = cond_rows[-1]
        summary.append(
            {
                "condition": condition,
                "records": len(cond_rows),
                "final_tick": float(final.get("tick", len(cond_rows) - 1)),
                "final_alive_count": float(final.get("alive_count", 0.0)),
                "final_global_integration": float(final.get("global_integration", 0.0)),
                "mean_global_integration": float(
                    np.mean([float(r.get("global_integration", 0.0)) for r in cond_rows])
                ),
                "max_alive_count": float(max(float(r.get("alive_count", 0.0)) for r in cond_rows)),
                "mean_signal_total": float(
                    np.mean([float(r.get("signal_total", 0.0)) for r in cond_rows])
                ),
            }
        )

    out = Path(out_path)
    if out.exists() and out.is_dir():
        raise ValueError(f"comparison output path points to a directory: {out}")
    if out.suffix.lower() == ".json":
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return
    if out.suffix.lower() == ".csv":
        _write_csv(summary, out)
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    conditions = [row["condition"] for row in summary]
    x = np.arange(len(summary))
    fig, ax1 = plt.subplots(figsize=(max(6, 1.4 * len(summary)), 4))
    ax1.bar(
        x - 0.2,
        [row["final_global_integration"] for row in summary],
        width=0.4,
        label="final integration",
    )
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("integration")
    ax2 = ax1.twinx()
    ax2.bar(
        x + 0.2,
        [row["final_alive_count"] for row in summary],
        width=0.4,
        label="final alive",
        alpha=0.5,
    )
    ax2.set_ylabel("alive count")
    ax1.set_xticks(x, conditions, rotation=30, ha="right")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def parameter_sweep_heatmap(results_path: str | Path, metric: str, out_path: str | Path) -> None:
    """Plot a parameter-sweep heatmap from sweep result rows.

    The function expects a CSV/JSON/etc. table with at least ``sweep_param_1``,
    ``sweep_value_1`` and the requested metric. If ``sweep_param_2`` and
    ``sweep_value_2`` are present, a 2D heatmap is produced; otherwise a
    one-row heatmap/line-like image is produced.
    """
    rows = _load_rows(results_path)
    if not rows:
        raise ValueError("cannot plot heatmap from empty sweep results")
    if metric not in rows[0]:
        raise ValueError(f"metric {metric!r} not present in sweep results")

    p1 = rows[0].get("sweep_param_1", "parameter")
    p2 = rows[0].get("sweep_param_2", None)
    x_values = sorted({str(row.get("sweep_value_1", "")) for row in rows}, key=str)
    y_values = (
        sorted({str(row.get("sweep_value_2", "value")) for row in rows}, key=str)
        if p2
        else ["value"]
    )

    grid = np.full((len(y_values), len(x_values)), np.nan, dtype=np.float64)
    counts = np.zeros_like(grid)
    x_index = {value: i for i, value in enumerate(x_values)}
    y_index = {value: i for i, value in enumerate(y_values)}

    for row in rows:
        x = x_index[str(row.get("sweep_value_1", ""))]
        y = y_index[str(row.get("sweep_value_2", "value"))] if p2 else 0
        grid[y, x] = np.nan_to_num(grid[y, x], nan=0.0) + float(row.get(metric, 0.0))
        counts[y, x] += 1.0

    grid = np.divide(grid, counts, out=np.full_like(grid, np.nan), where=counts > 0)

    out = Path(out_path)
    if out.exists() and out.is_dir():
        raise ValueError(f"heatmap output path points to a directory: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(5, len(x_values)), max(3, len(y_values) * 0.8 + 2)))
    im = ax.imshow(grid, aspect="auto")
    ax.set_xticks(np.arange(len(x_values)), x_values, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(y_values)), y_values)
    ax.set_xlabel(str(p1))
    ax.set_ylabel(str(p2) if p2 else "")
    ax.set_title(metric)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
