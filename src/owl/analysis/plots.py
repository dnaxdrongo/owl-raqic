"""Matplotlib plotting interfaces for run analysis."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib import animation

from owl.analysis.zarr_reader import load_field


def _load_rows(metrics_path: str | Path) -> list[dict[str, Any]]:
    """Load scalar metric rows from JSON/JSONL/CSV/Parquet-or-fallback files."""
    path = Path(metrics_path)
    if not path.exists():
        raise FileNotFoundError(f"metrics file not found: {path}")
    if path.is_dir():
        raise ValueError(f"metrics path points to a directory: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"metrics JSON root must be a list, got {type(data).__name__}")
        return [dict(row) for row in data]

    if suffix in {".jsonl", ".ndjson"}:
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(dict(json.loads(line)))
        return rows

    if suffix == ".csv":
        with path.open("r", newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]

    if suffix == ".parquet":
        try:
            import polars as pl

            return pl.read_parquet(str(path)).to_dicts()
        except Exception:
            pass
        try:
            import pandas as pd

            return cast(list[dict[str, Any]], pd.read_parquet(path).to_dict("records"))
        except Exception:
            rows = []
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    item = json.loads(line)
                    if item.get("format") == "jsonl-fallback":
                        continue
                    rows.append(dict(item))
            return rows

    raise ValueError(f"unsupported metrics file extension {suffix!r}")


def _column(rows: list[dict[str, Any]], name: str, default: float = 0.0) -> np.ndarray:
    """Return a numeric column from metric rows."""
    values = []
    for row in rows:
        try:
            values.append(float(row.get(name, default)))
        except (TypeError, ValueError):
            values.append(default)
    return np.asarray(values, dtype=np.float64)


def _ticks(rows: list[dict[str, Any]]) -> np.ndarray:
    """Return tick column or a record index fallback."""
    if rows and "tick" in rows[0]:
        return _column(rows, "tick")
    return np.arange(len(rows), dtype=np.float64)


def _save_current(out_path: str | Path) -> None:
    """Save current Matplotlib figure and close it."""
    path = Path(out_path)
    if path.exists() and path.is_dir():
        raise ValueError(f"plot output path points to a directory: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_global_integration(metrics_path: str | Path, out_path: str | Path) -> None:
    """Plot global integration and fragmentation over time."""
    rows = _load_rows(metrics_path)
    if not rows:
        raise ValueError("cannot plot empty metrics table")
    tick = _ticks(rows)

    plt.figure(figsize=(10, 4))
    plt.plot(tick, _column(rows, "global_integration"), label="global integration")
    plt.plot(
        tick,
        _column(
            rows, "global_fragmentation", _column(rows, "fragmentation").mean() if rows else 0.0
        ),
        label="fragmentation",
    )
    plt.plot(tick, _column(rows, "mean_integration"), label="mean cell integration", alpha=0.8)
    plt.xlabel("tick")
    plt.ylabel("value")
    plt.ylim(0, 1)
    plt.legend()
    _save_current(out_path)


def plot_population_by_trait(metrics_path: str | Path, out_path: str | Path) -> None:
    """Plot population and trait/type fractions over time."""
    rows = _load_rows(metrics_path)
    if not rows:
        raise ValueError("cannot plot empty metrics table")
    tick = _ticks(rows)

    plt.figure(figsize=(10, 4))
    plt.plot(tick, _column(rows, "alive_count"), label="alive count")
    if "grazer_fraction" in rows[0]:
        plt.plot(tick, _column(rows, "grazer_fraction"), label="grazer fraction")
    if "carnivore_fraction" in rows[0]:
        plt.plot(tick, _column(rows, "carnivore_fraction"), label="carnivore fraction")
    if "reproduce_fraction" in rows[0]:
        plt.plot(tick, _column(rows, "reproduce_fraction"), label="reproduce action fraction")
    plt.xlabel("tick")
    plt.ylabel("count / fraction")
    plt.legend()
    _save_current(out_path)


def plot_signal_channel_totals(metrics_path: str | Path, out_path: str | Path) -> None:
    """Plot communication channel totals over time."""
    rows = _load_rows(metrics_path)
    if not rows:
        raise ValueError("cannot plot empty metrics table")
    tick = _ticks(rows)

    plt.figure(figsize=(10, 4))
    plotted = False
    for name in (
        "signal_total",
        "signal_food_total",
        "signal_danger_total",
        "signal_coordination_total",
        "signal_integration_total",
    ):
        if name in rows[0]:
            plt.plot(tick, _column(rows, name), label=name)
            plotted = True
    if not plotted:
        raise ValueError("metrics table does not contain signal total columns")
    plt.xlabel("tick")
    plt.ylabel("signal mass")
    plt.legend()
    _save_current(out_path)


def _normalize_frames(arr: np.ndarray) -> np.ndarray:
    """Convert time-first scalar or RGB data into uint8 frames."""
    data = np.asarray(arr)
    if data.ndim < 3:
        raise ValueError(
            f"recorded field must be time-first with at least 3 dimensions, got {data.shape}"
        )
    if data.ndim == 4 and data.shape[-1] == 3:
        frames = np.clip(data, 0, 255).astype(np.uint8)
    elif data.ndim == 4:
        scalar = np.mean(data, axis=-1)
        min_v = float(np.nanmin(scalar))
        max_v = float(np.nanmax(scalar))
        denom = max(max_v - min_v, 1e-8)
        gray = np.clip((scalar - min_v) / denom * 255.0, 0, 255).astype(np.uint8)
        frames = np.repeat(gray[..., None], 3, axis=-1)
    else:
        min_v = float(np.nanmin(data))
        max_v = float(np.nanmax(data))
        denom = max(max_v - min_v, 1e-8)
        gray = np.clip((data - min_v) / denom * 255.0, 0, 255).astype(np.uint8)
        frames = np.repeat(gray[..., None], 3, axis=-1)
    return cast(np.ndarray, frames)


def make_animation_from_zarr(zarr_path: str | Path, field: str, out_path: str | Path) -> None:
    """Create an animation from a recorded field.

    If a video/GIF writer is unavailable, a compressed ``.npz`` animation bundle
    is written to the requested path using a binary file handle, preserving the
    caller's file name.
    """
    frames = _normalize_frames(load_field(zarr_path, field))
    path = Path(out_path)
    if path.exists() and path.is_dir():
        raise ValueError(f"animation output path points to a directory: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".npz":
        np.savez_compressed(path, frames=frames, field=str(field))
        return

    fig, ax = plt.subplots(figsize=(5, 5))
    image = ax.imshow(frames[0])
    ax.set_title(str(field))
    ax.axis("off")

    def update(i: int) -> Any:
        image.set_data(frames[i])
        ax.set_xlabel(f"frame {i}")
        return (image,)

    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=100, blit=True)
    try:
        if path.suffix.lower() == ".gif":
            anim.save(path, writer="pillow")
        else:
            anim.save(path)
    except Exception:
        with path.open("wb") as handle:
            np.savez_compressed(handle, frames=frames, field=str(field), fallback=np.asarray(True))
    finally:
        plt.close(fig)
