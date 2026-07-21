"""Zarr/fallback reading helpers for recorded histories."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


def _try_import_zarr() -> Any | None:
    """Import zarr lazily so analysis remains available in minimal runtimes."""
    try:
        import zarr

        return zarr
    except Exception:
        return None


def _field_to_fallback_filename(field: str) -> str:
    """Map a logical recording field to a fallback ``.npy`` filename."""
    clean = str(field).strip().strip("/")
    if not clean:
        raise ValueError("field name cannot be empty")
    return clean.replace("/", "__") + ".npy"


def open_run_zarr(path: str | Path) -> Any:
    """Open a recorded run group or fallback directory.

    Parameters
    ----------
    path:
        Path to a Zarr store or the NumPy-directory fallback written by
        ``ZarrRecorder``.

    Returns
    -------
    Any
        A real Zarr group when zarr is installed and the path is a Zarr store;
        otherwise a ``SimpleNamespace`` with ``backend='numpy-directory-fallback'``,
        ``path`` and ``metadata`` fields.
    """
    run_path = Path(path)
    if not run_path.exists():
        raise FileNotFoundError(f"recorded run not found: {run_path}")
    if run_path.is_file():
        raise ValueError(f"recorded run path must be a directory/store, got file: {run_path}")

    zarr = _try_import_zarr()
    metadata_path = run_path / "metadata.json"

    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return SimpleNamespace(backend="numpy-directory-fallback", path=run_path, metadata=metadata)

    if zarr is None:
        raise ImportError(
            f"zarr is not installed and no fallback metadata was found at {metadata_path}"
        )

    return zarr.open_group(str(run_path), mode="r")


def load_field(path: str | Path, field: str) -> Any:
    """Load a recorded field from a run.

    Parameters
    ----------
    path:
        Zarr store or fallback directory.
    field:
        Logical field path such as ``'state/health'`` or ``'global/integration'``.

    Returns
    -------
    np.ndarray
        Recorded array. The first axis is time for recorded dense fields.
    """
    run = open_run_zarr(path)

    if getattr(run, "backend", None) == "numpy-directory-fallback":
        filename = _field_to_fallback_filename(field)
        field_path = Path(run.path) / filename
        if not field_path.exists():
            available = sorted(p.name for p in Path(run.path).glob("*.npy"))
            raise KeyError(
                f"field {field!r} not found in fallback recording; available files: {available}"
            )
        return np.load(field_path)

    try:
        return np.asarray(run[field])
    except Exception as exc:
        keys = list(run.array_keys()) if hasattr(run, "array_keys") else []
        raise KeyError(
            f"field {field!r} not found in zarr run; available arrays include: {keys}"
        ) from exc
