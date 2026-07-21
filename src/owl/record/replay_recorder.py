"""Authoritative, append-safe replay recording for offline analysis and Pygame replay."""

from __future__ import annotations

import contextlib
import csv
import json
import os
import platform
import sys
import time
from collections.abc import Iterable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from owl.record.action_math_batch import NumPyReplayBatchBuilder, build_living_index
from owl.record.parquet_sink import PartitionedParquetSink
from owl.record.replay_schema import CompiledReplaySchema, compile_replay_schema
from owl.replay.manifest import ReplayManifest, sha256_file
from owl.viz.event_bus import VisualEvent
from owl.viz.visual_snapshot import VisualSnapshot

REPLAY_TIERS = {
    "replay_standard",
    "analysis_full",
    "analysis_sampled",
    "metrics_only",
    "debug_full",
}

REPLAY_FIELD_METADATA: dict[str, dict[str, str]] = {
    "health": {"units": "normalized", "stage": "postdecision", "role": "replay_required"},
    "resource": {"units": "normalized", "stage": "postdecision", "role": "replay_required"},
    "occupancy": {"units": "stable_ow_id", "stage": "postdecision", "role": "replay_required"},
    "readout": {"units": "action_index", "stage": "decision", "role": "replay_required"},
    "raqic_readout": {"units": "action_index", "stage": "decision", "role": "analysis"},
    "raqic_probabilities": {"units": "probability", "stage": "decision", "role": "action_math"},
    "possibility": {"units": "probability", "stage": "predecision", "role": "action_math"},
    "last_utilities": {"units": "utility", "stage": "predecision", "role": "action_math"},
    "pre_utilities": {"units": "utility", "stage": "predecision", "role": "action_math"},
    "last_logits": {"units": "logit", "stage": "decision", "role": "action_math"},
    "last_action_probabilities": {
        "units": "probability",
        "stage": "decision",
        "role": "action_math",
    },
    "raqic_score": {"units": "logit", "stage": "decision", "role": "action_math"},
    "raqic_phase": {"units": "radians", "stage": "decision", "role": "action_math"},
    "raqic_parent_intention": {
        "units": "probability",
        "stage": "predecision",
        "role": "action_math",
    },
    "raqic_pre_mixer_probabilities": {
        "units": "probability",
        "stage": "decision",
        "role": "action_math",
    },
    "raqic_utility_innovation": {
        "units": "bounded_score",
        "stage": "decision",
        "role": "action_math",
    },
    "raqic_phase_alignment": {"units": "unitless", "stage": "decision", "role": "action_math"},
    "raqic_resonant_parent_intention": {
        "units": "bounded_score",
        "stage": "decision",
        "role": "action_math",
    },
    "raqic_shadow_probabilities": {"units": "probability", "stage": "audit", "role": "action_math"},
    "authority": {"units": "legal_mask", "stage": "predecision", "role": "action_math"},
    "_authority_bool": {"units": "legal_mask", "stage": "predecision", "role": "action_math"},
}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _array_value_for_position(value: Any, y: int, x: int, world_shape: tuple[int, int]) -> Any:
    """Return the OW-local value for spatial arrays, or a jsonable global value.

    Replay diagnostics may mix several shapes:
    - scalar values
    - 2D world arrays shaped (height, width)
    - 3D world/action arrays shaped (height, width, action_count)
    - flat spatial arrays shaped (height * width,)
    - global/action vectors shaped (action_count,)

    The recorder must not assume every diagnostic field is indexable as [y, x].
    """
    array = np.asarray(value)

    if array.ndim == 0:
        return _jsonable(array)

    height, width = int(world_shape[0]), int(world_shape[1])

    if array.ndim >= 2 and array.shape[0] == height and array.shape[1] == width:
        return _jsonable(array[y, x])

    if array.ndim == 1 and array.shape[0] == height * width:
        return _jsonable(array[(y * width) + x])

    return _jsonable(array)


def _write_embedded_viewer_launchers(root: Path) -> None:
    viewer = root / "viewer"
    viewer.mkdir(parents=True, exist_ok=True)
    (viewer / "viewer_compatibility.json").write_text(
        json.dumps(
            {
                "schema_version": "owl.replay.viewer-compatibility.v1",
                "replay_schema_major": 1,
                "recommended_command": "owl-replay <bundle-root>",
                "standalone_application": "OWLReplayViewer",
                "bundle_is_read_only": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (viewer / "README.md").write_text(
        "# OWL Replay Viewer\n\n"
        "Open this experiment bundle with the standalone `OWLReplayViewer` application, "
        "or from an installed OWL environment run `owl-replay <bundle-root>`.\n\n"
        "The viewer reads the bundle without modifying it. Screenshots, bookmarks, and "
        "CSV exports are written to a separate local output directory.\n",
        encoding="utf-8",
    )
    (viewer / "launch_viewer_windows.bat").write_text(
        "@echo off\r\n"
        "set BUNDLE=%~dp0..\r\n"
        'where OWLReplayViewer.exe >nul 2>nul && (OWLReplayViewer.exe "%BUNDLE%" & exit /b)\r\n'
        'where owl-replay >nul 2>nul && (owl-replay "%BUNDLE%" & exit /b)\r\n'
        "echo Install or place OWLReplayViewer.exe on PATH, then reopen this file.\r\n"
        "pause\r\n",
        encoding="utf-8",
    )
    shell = viewer / "launch_viewer.sh"
    shell.write_text(
        "#!/usr/bin/env sh\n"
        "set -eu\n"
        'BUNDLE=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)\n'
        'exec owl-replay "$BUNDLE"\n',
        encoding="utf-8",
    )
    with contextlib.suppress(OSError):
        shell.chmod(0o755)


def estimate_replay_bytes(
    *,
    world_shape: tuple[int, int],
    ticks: int,
    field_specs: dict[str, tuple[tuple[int, ...], np.dtype[Any]]],
    compression_ratio: float = 0.35,
) -> dict[str, Any]:
    raw = 0
    for shape, dtype in field_specs.values():
        raw += int(np.prod((ticks, *shape), dtype=np.int64)) * int(np.dtype(dtype).itemsize)
    return {
        "world_shape": list(world_shape),
        "ticks": int(ticks),
        "field_count": len(field_specs),
        "raw_bytes": raw,
        "estimated_compressed_bytes": int(raw * float(compression_ratio)),
    }


def _zarr_local_store(zarr_module: Any, path: Path) -> Any:
    """Return a local filesystem Zarr store for Zarr v2 or v3."""
    store_path = str(path)
    storage_module = getattr(zarr_module, "storage", None)
    local_store = getattr(storage_module, "LocalStore", None)
    if local_store is not None:
        return local_store(store_path)

    directory_store = getattr(zarr_module, "DirectoryStore", None)
    if directory_store is not None:
        return directory_store(store_path)

    return store_path


def _zarr_open_group_compat(zarr_module: Any, store: Any, *, mode: str) -> Any:
    """Open/create a Zarr group using APIs compatible with Zarr v2 and v3.

    Force zarr_format=2 where supported because the replay bundle currently uses
    numcodecs.Blosc and appendable array semantics that were authored for v2.
    """
    open_group = getattr(zarr_module, "open_group", None)
    if open_group is not None:
        try:
            return open_group(store=store, mode=mode, zarr_format=2)
        except TypeError:
            return open_group(store=store, mode=mode)

    if mode == "w":
        return zarr_module.group(store=store, overwrite=True)
    return zarr_module.open(store=store, mode=mode)


def _zarr_create_dataset_compat(
    group: Any,
    name: str,
    *,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    dtype: Any,
    compressor: Any | None = None,
) -> Any:
    """Create an appendable replay array across Zarr v2 and Zarr v3 APIs."""
    create_dataset = getattr(group, "create_dataset", None)
    if create_dataset is not None:
        try:
            return create_dataset(
                name,
                shape=shape,
                chunks=chunks,
                dtype=dtype,
                compressor=compressor,
            )
        except TypeError:
            # Fall through to the v3 create_array API.
            pass

    create_array = getattr(group, "create_array", None)
    if create_array is None:
        raise AttributeError("Zarr group has neither create_dataset nor create_array")

    kwargs = {
        "shape": shape,
        "chunks": chunks,
        "dtype": dtype,
    }

    # For zarr_format=2, zarr v3's create_array still accepts the singular
    # compressor argument. If a future version rejects it, retry without it.
    if compressor is not None:
        kwargs["compressor"] = compressor

    try:
        return create_array(name, **kwargs)
    except TypeError:
        kwargs.pop("compressor", None)
        try:
            return create_array(name, **kwargs)
        except TypeError:
            # Final fallback for true v3 stores.
            kwargs["compressors"] = None
            return create_array(name, **kwargs)


def _event_arrow_schema(schema_digest: str) -> Any:
    import pyarrow as pa

    metadata = {
        b"owl.replay.columnar.schema_version": b"owl.replay.columnar.v1",
        b"owl.replay.schema_digest": schema_digest.encode(),
    }
    return pa.schema(
        [
            pa.field("tick", pa.int64(), nullable=False),
            pa.field("event_type", pa.string(), nullable=False),
            pa.field("event_type_code", pa.int16(), nullable=False),
            pa.field("event_sequence", pa.int32(), nullable=False),
            pa.field("event_id", pa.string(), nullable=False),
            pa.field("y", pa.int32(), nullable=False),
            pa.field("x", pa.int32(), nullable=False),
            pa.field("target_y", pa.int32(), nullable=False),
            pa.field("target_x", pa.int32(), nullable=False),
            pa.field("action", pa.int16(), nullable=False),
            pa.field("intensity", pa.float64(), nullable=False),
            pa.field("ttl", pa.int32(), nullable=False),
            pa.field("source_id", pa.int64(), nullable=False),
            pa.field("channel", pa.int32(), nullable=False),
            pa.field("payload0", pa.float64(), nullable=False),
            pa.field("payload1", pa.float64(), nullable=False),
            pa.field("priority", pa.int32(), nullable=False),
        ],
        metadata=metadata,
    )


def _events_record_batch(rows: list[dict[str, Any]], schema: Any) -> Any:
    import pyarrow as pa

    arrays = []
    for field in schema:
        values = [row[field.name] for row in rows]
        arrays.append(pa.array(values, type=field.type, from_pandas=False, safe=True))
    batch = pa.RecordBatch.from_arrays(arrays, schema=schema)
    batch.validate(full=False)
    return batch


class ReplayRecorder:
    """Write one scientific snapshot per committed tick into a versioned bundle."""

    def __init__(
        self,
        bundle_root: str | Path,
        *,
        run_id: str,
        condition: str,
        seed: int,
        requested_ticks: int,
        recording_tier: str = "replay_standard",
        source_sha256: str = "unknown",
        config_sha256: str = "unknown",
        action_names: Iterable[str] = (),
        hardware: dict[str, Any] | None = None,
        qiskit_execution: dict[str, Any] | None = None,
        max_output_bytes: int | None = None,
        table_flush_ticks: int = 1,
        columnar_backend: str | None = None,
        max_batch_rows: int = 131_072,
        max_batch_bytes: int = 128 * 1024 * 1024,
        parquet_row_group_rows: int = 131_072,
        materialization_mode: str = "inline",
        pinned_pool_bytes: int = 256 * 1024 * 1024,
        queue_depth: int = 1,
        compression: str = "zstd",
        telemetry_enabled: bool = True,
        adaptive_batching: bool = False,
        strict_acceleration: bool = False,
        cadc_config: Any | None = None,
    ) -> None:
        if recording_tier not in REPLAY_TIERS:
            raise ValueError(f"unknown replay recording tier: {recording_tier}")
        self.root = Path(bundle_root)
        if self.root.exists() and any(self.root.iterdir()):
            raise FileExistsError(f"replay bundle root is not empty: {self.root}")
        self.root.mkdir(parents=True, exist_ok=True)
        for relative in (
            "replay",
            "analysis",
            "logs",
            "checkpoints",
            "checksums",
            "schema",
            "viewer",
        ):
            (self.root / relative).mkdir(parents=True, exist_ok=True)
        self.run_id = str(run_id)
        self.condition = str(condition)
        self.seed = int(seed)
        self.requested_ticks = int(requested_ticks)
        self.recording_tier = recording_tier
        self.source_sha256 = source_sha256
        self.config_sha256 = config_sha256
        self.action_names = tuple(str(item) for item in action_names)
        self.hardware = dict(hardware or {})
        self.qiskit_execution = dict(qiskit_execution or {})
        self.max_output_bytes = None if max_output_bytes is None else int(max_output_bytes)
        self.table_flush_ticks = max(
            1, int(os.environ.get("OWL_REPLAY_TABLE_FLUSH_TICKS", str(table_flush_ticks)))
        )
        self.columnar_backend = str(
            os.environ.get("OWL_REPLAY_COLUMNAR_BACKEND", columnar_backend or "auto")
        ).lower()
        if self.columnar_backend not in {"auto", "numpy_host", "cupy_staged"}:
            raise ValueError(f"unknown replay columnar backend: {self.columnar_backend}")
        self.max_batch_rows = max(
            1, int(os.environ.get("OWL_REPLAY_MAX_BATCH_ROWS", str(max_batch_rows)))
        )
        self.max_batch_bytes = max(
            1, int(os.environ.get("OWL_REPLAY_MAX_BATCH_BYTES", str(max_batch_bytes)))
        )
        self.parquet_row_group_rows = max(
            1,
            int(os.environ.get("OWL_REPLAY_PARQUET_ROW_GROUP_ROWS", str(parquet_row_group_rows))),
        )
        self.materialization_mode = str(
            os.environ.get("OWL_REPLAY_MATERIALIZATION_MODE", materialization_mode)
        ).lower()
        if self.materialization_mode not in {"inline", "deferred"}:
            raise ValueError(f"unknown replay materialization mode: {self.materialization_mode}")
        self.pinned_pool_bytes = max(
            1, int(os.environ.get("OWL_REPLAY_PINNED_POOL_BYTES", str(pinned_pool_bytes)))
        )
        self.queue_depth = max(1, int(os.environ.get("OWL_REPLAY_QUEUE_DEPTH", str(queue_depth))))
        if self.queue_depth != 1:
            raise ValueError(
                "v0.9.9 queue_depth must remain 1 until overlap is target-GPU certified"
            )
        self.compression = str(
            os.environ.get("OWL_REPLAY_PARQUET_COMPRESSION", compression)
        ).lower()
        if self.compression not in {"zstd", "snappy", "gzip", "none"}:
            raise ValueError(f"unsupported replay Parquet compression: {self.compression}")
        self.telemetry_enabled = (
            bool(telemetry_enabled) and os.environ.get("OWL_REPLAY_TELEMETRY", "1") != "0"
        )
        self.adaptive_batching = (
            bool(adaptive_batching) or os.environ.get("OWL_REPLAY_ADAPTIVE_BATCHING", "0") == "1"
        )
        self.strict_acceleration = (
            bool(strict_acceleration)
            or os.environ.get("OWL_REPLAY_STRICT_ACCELERATION", "0") == "1"
        )
        if cadc_config is None:
            self.cadc_config: dict[str, Any] = {"enabled": False}
        elif hasattr(cadc_config, "model_dump"):
            self.cadc_config = dict(cadc_config.model_dump(mode="json"))
        else:
            self.cadc_config = dict(cadc_config)
        self._cadc_recorder: Any | None = None
        self._cadc_resume = False
        self._zarr: Any | None = None
        self._group: Any | None = None
        self._arrays: dict[str, Any] = {}
        self._field_specs: dict[str, tuple[tuple[int, ...], np.dtype[Any]]] = {}
        self._world_shape: tuple[int, int] | None = None
        self._boundary_mode = "toroidal"
        self._ticks: list[int] = []
        self._event_rows: list[dict[str, Any]] = []
        self._metric_rows: list[dict[str, Any]] = []
        self._compiled_schema: CompiledReplaySchema | None = None
        self._batch_builder: NumPyReplayBatchBuilder | None = None
        self._sinks: dict[str, PartitionedParquetSink] = {}
        self._resume_sinks = False
        self._max_committed_tick: int | None = None
        self._recording_telemetry: dict[str, Any] = {}
        self._adaptive_policy: Any | None = None
        self._closed = False
        self._created_at = datetime.now(UTC).isoformat()
        _atomic_json(
            self.root / "run_status.json",
            {
                "schema_version": "owl.replay.status.v1",
                "state": "RUNNING",
                "run_id": self.run_id,
                "requested_ticks": self.requested_ticks,
                "completed_ticks": 0,
                "last_committed_tick": None,
                "created_at": self._created_at,
                "recording_backend": self.columnar_backend,
                "materialization_mode": self.materialization_mode,
                "materialization_state": (
                    "pending" if self.materialization_mode == "deferred" else "complete"
                ),
            },
        )
        _write_embedded_viewer_launchers(self.root)
        _atomic_json(
            self.root / "source_environment.json",
            {
                "python": sys.version,
                "platform": platform.platform(),
                "hardware": self.hardware,
                "qiskit_execution": self.qiskit_execution,
                "columnar_replay": {
                    "version": "0.9.9",
                    "backend": self.columnar_backend,
                    "max_batch_rows": self.max_batch_rows,
                    "max_batch_bytes": self.max_batch_bytes,
                    "parquet_row_group_rows": self.parquet_row_group_rows,
                    "materialization_mode": self.materialization_mode,
                    "pinned_pool_bytes": self.pinned_pool_bytes,
                    "queue_depth": self.queue_depth,
                    "compression": self.compression,
                    "telemetry_enabled": self.telemetry_enabled,
                    "adaptive_batching": self.adaptive_batching,
                    "strict_acceleration": self.strict_acceleration,
                },
                "cadc_factual": {
                    "schema_version": "owl.cadc.factual.v1",
                    "enabled": bool(self.cadc_config.get("enabled", False)),
                    "config": self.cadc_config,
                },
            },
        )

    @classmethod
    def resume(cls, bundle_root: str | Path) -> ReplayRecorder:
        """Reopen a cleanly closed interrupted bundle for exact append-only continuation."""

        import zarr

        root = Path(bundle_root)
        manifest = ReplayManifest.load(root)
        status = json.loads((root / "run_status.json").read_text(encoding="utf-8"))
        if status.get("state") != "INTERRUPTED_RESUMABLE":
            raise RuntimeError(
                "only INTERRUPTED_RESUMABLE replay bundles may be reopened for append"
            )
        recorder = cls.__new__(cls)
        recorder.root = root
        recorder.run_id = manifest.run_id
        recorder.condition = manifest.condition
        recorder.seed = manifest.seed
        recorder.requested_ticks = manifest.requested_ticks
        recorder.recording_tier = manifest.recording_tier
        recorder.source_sha256 = manifest.source_sha256
        recorder.config_sha256 = manifest.config_sha256
        recorder.action_names = manifest.action_names
        recorder.hardware = dict(manifest.hardware)
        recorder.qiskit_execution = dict(manifest.qiskit_execution)
        recorder.max_output_bytes = None
        recorder.table_flush_ticks = 1
        recorder.columnar_backend = os.environ.get("OWL_REPLAY_COLUMNAR_BACKEND", "auto").lower()
        recorder.max_batch_rows = max(1, int(os.environ.get("OWL_REPLAY_MAX_BATCH_ROWS", "131072")))
        recorder.max_batch_bytes = max(
            1, int(os.environ.get("OWL_REPLAY_MAX_BATCH_BYTES", str(128 * 1024 * 1024)))
        )
        recorder.parquet_row_group_rows = max(
            1, int(os.environ.get("OWL_REPLAY_PARQUET_ROW_GROUP_ROWS", "131072"))
        )
        recorder.materialization_mode = str(
            status.get(
                "materialization_mode",
                os.environ.get("OWL_REPLAY_MATERIALIZATION_MODE", manifest.materialization_mode),
            )
        ).lower()
        recorder.pinned_pool_bytes = max(
            1, int(os.environ.get("OWL_REPLAY_PINNED_POOL_BYTES", str(256 * 1024 * 1024)))
        )
        recorder.queue_depth = max(1, int(os.environ.get("OWL_REPLAY_QUEUE_DEPTH", "1")))
        if recorder.queue_depth != 1:
            raise ValueError("resume queue_depth must remain 1")
        recorder.compression = os.environ.get("OWL_REPLAY_PARQUET_COMPRESSION", "zstd").lower()
        recorder.telemetry_enabled = os.environ.get("OWL_REPLAY_TELEMETRY", "1") != "0"
        recorder.adaptive_batching = os.environ.get("OWL_REPLAY_ADAPTIVE_BATCHING", "0") == "1"
        recorder.strict_acceleration = os.environ.get("OWL_REPLAY_STRICT_ACCELERATION", "0") == "1"
        source_environment = json.loads(
            (root / "source_environment.json").read_text(encoding="utf-8")
        )
        cadc_metadata = source_environment.get("cadc_factual", {})
        recorder.cadc_config = dict(cadc_metadata.get("config", {"enabled": False}))
        recorder._cadc_recorder = None
        recorder._cadc_resume = bool(cadc_metadata.get("enabled", False))
        recorder._zarr = zarr
        resume_store = _zarr_local_store(zarr, root / "replay" / "replay.zarr")
        recorder._group = _zarr_open_group_compat(zarr, resume_store, mode="a")
        recorder._arrays = {"tick": recorder._group["tick"]}
        state_group = recorder._group.get("state")
        if state_group is not None:
            for name in state_group.array_keys():
                recorder._arrays[str(name)] = state_group[str(name)]
        commits = sorted((root / "replay" / "commits").glob("tick_*.json"))
        committed_ticks = [
            int(json.loads(path.read_text(encoding="utf-8"))["tick"]) for path in commits
        ]
        committed_count = len(committed_ticks)
        for dataset in recorder._arrays.values():
            if int(dataset.shape[0]) > committed_count:
                dataset.resize((committed_count, *dataset.shape[1:]))
            if int(dataset.shape[0]) != committed_count:
                raise RuntimeError("replay arrays do not match committed tick count during resume")
        stored_ticks = [int(item) for item in np.asarray(recorder._arrays["tick"][:]).tolist()]
        if stored_ticks != committed_ticks:
            raise RuntimeError("Zarr tick values and commit markers disagree during resume")
        recorder._field_specs = {
            str(name): (tuple(int(item) for item in dataset.shape[1:]), np.dtype(dataset.dtype))
            for name, dataset in recorder._arrays.items()
            if name != "tick"
        }
        recorder._world_shape = manifest.world_shape
        recorder._boundary_mode = manifest.boundary_mode
        recorder._ticks = committed_ticks
        recorder._event_rows = []
        recorder._metric_rows = []
        recorder._compiled_schema = None
        recorder._batch_builder = None
        recorder._sinks = {}
        recorder._resume_sinks = True
        recorder._max_committed_tick = committed_ticks[-1] if committed_ticks else None
        recorder._recording_telemetry = {}
        recorder._adaptive_policy = None
        recorder._closed = False
        recorder._created_at = manifest.created_at
        (root / "checksums" / "SHA256SUMS.txt").unlink(missing_ok=True)
        _atomic_json(
            root / "run_status.json",
            {
                **status,
                "state": "RUNNING",
                "resumed_at": datetime.now(UTC).isoformat(),
                "completed_ticks": committed_count,
                "last_committed_tick": committed_ticks[-1] if committed_ticks else None,
                "materialization_mode": recorder.materialization_mode,
                "materialization_state": (
                    "pending" if recorder.materialization_mode == "deferred" else "complete"
                ),
            },
        )
        return recorder

    @staticmethod
    def _chunk_for(shape: tuple[int, ...]) -> tuple[int, ...]:
        return (1, *(min(max(1, int(value)), 64) for value in shape))

    def _initialize(self, snapshot: VisualSnapshot) -> None:
        import zarr
        from numcodecs import Blosc

        self._zarr = zarr
        self._world_shape = snapshot.world_shape
        self._boundary_mode = snapshot.boundary_mode
        store = _zarr_local_store(zarr, self.root / "replay" / "replay.zarr")
        self._group = _zarr_open_group_compat(zarr, store, mode="w")
        self._group.attrs.update(
            {
                "schema_version": "owl.replay.arrays.v1",
                "run_id": self.run_id,
                "world_shape": list(snapshot.world_shape),
                "boundary_mode": snapshot.boundary_mode,
                "recording_tier": self.recording_tier,
            }
        )
        compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
        self._arrays["tick"] = _zarr_create_dataset_compat(
            self._group,
            "tick",
            shape=(0,),
            chunks=(256,),
            dtype="i8",
            compressor=compressor,
        )
        selected = dict(snapshot.arrays)
        if self.recording_tier == "metrics_only":
            selected = {
                name: value for name, value in selected.items() if name in {"health", "occupancy"}
            }
        elif self.recording_tier == "replay_standard":
            needed = {
                "health",
                "resource",
                "toxin",
                "food",
                "waste",
                "obstacle",
                "occupancy",
                "readout",
                "raqic_readout",
                "integration",
                "boundary",
                "age",
                "ow_type",
                "lineage_id",
                "parent_id",
                "development_stage",
                "starvation_debt",
                "genome",
                "mobility",
                "metabolism",
                "predation",
                "grazing",
                "cooperation",
                "aggression",
                "curiosity",
                "reproduction_rate",
                "toxin_resistance",
                "memory_capacity",
                "coupling_strength",
                "emit_strength",
                "signal_precision",
                "honesty_bias",
                "deception_bias",
                "signal_emission",
                "last_death_mask",
                "raqic_record_confidence",
            }
            selected = {name: value for name, value in selected.items() if name in needed}
        for name, value in sorted(selected.items()):
            array = np.asarray(value)
            if array.dtype == object:
                continue
            path = f"state/{name}"
            dataset = _zarr_create_dataset_compat(
                self._group,
                path,
                shape=(0, *array.shape),
                chunks=self._chunk_for(array.shape),
                dtype=array.dtype,
                compressor=compressor,
            )
            self._arrays[name] = dataset
            self._field_specs[name] = (tuple(int(item) for item in array.shape), array.dtype)
        schema = {
            "schema_version": "owl.replay.schema.v1",
            "fields": {
                name: {
                    "path": f"replay/replay.zarr/state/{name}",
                    "shape_per_tick": list(shape),
                    "dtype": str(dtype),
                    "tick_timing": REPLAY_FIELD_METADATA.get(name, {}).get(
                        "stage", "postdecision_scientific_boundary"
                    ),
                    "units": REPLAY_FIELD_METADATA.get(name, {}).get("units", "unspecified"),
                    "role": REPLAY_FIELD_METADATA.get(name, {}).get("role", "analysis"),
                    "missing_semantics": "field absent from source snapshot",
                    "empty_dead_semantics": "masked by health<=0 or occupancy<0",
                }
                for name, (shape, dtype) in self._field_specs.items()
            },
        }
        _atomic_json(self.root / "schema" / "replay_schema.json", schema)
        estimate = estimate_replay_bytes(
            world_shape=snapshot.world_shape,
            ticks=self.requested_ticks,
            field_specs=self._field_specs,
        )
        _atomic_json(self.root / "schema" / "storage_estimate.json", estimate)
        if (
            self.max_output_bytes is not None
            and estimate["estimated_compressed_bytes"] > self.max_output_bytes
        ):
            raise RuntimeError(
                "estimated replay output exceeds configured maximum: "
                f"{estimate['estimated_compressed_bytes']} > {self.max_output_bytes}"
            )
        self._initialize_columnar(snapshot)

    def _initialize_columnar(self, snapshot: VisualSnapshot) -> None:
        if self._compiled_schema is not None:
            return
        compiled = compile_replay_schema(
            dict(snapshot.arrays),
            world_shape=snapshot.world_shape,
            action_names=self.action_names,
            recording_tier=self.recording_tier,
        )
        if self.materialization_mode == "deferred" and compiled.action_math_schema is None:
            raise ValueError(
                "deferred materialization requires an analysis tier with action mathematics"
            )
        self._compiled_schema = compiled
        self._batch_builder = NumPyReplayBatchBuilder(
            compiled,
            condition=self.condition,
            seed=self.seed,
            action_names=self.action_names,
            max_batch_rows=self.max_batch_rows,
            max_batch_bytes=self.max_batch_bytes,
            full_validation=os.environ.get("OWL_REPLAY_FULL_ARROW_VALIDATION", "0") == "1",
        )
        if self.adaptive_batching and compiled.action_math_schema is not None:
            from owl.record.replay_telemetry import AdaptiveBatchPolicy

            self._adaptive_policy = AdaptiveBatchPolicy(
                action_count=compiled.action_count,
                initial_rows=self.max_batch_rows,
                min_rows=min(
                    self.max_batch_rows,
                    max(compiled.action_count, compiled.action_count * 64),
                ),
                max_rows=self.max_batch_rows,
                target_batch_bytes=self.max_batch_bytes,
            )
        event_schema = _event_arrow_schema(compiled.schema_digest)
        self._sinks["events"] = PartitionedParquetSink(
            self.root / "replay" / "events.parquet",
            event_schema,
            table_name="events",
            schema_digest=compiled.schema_digest,
            compression=self.compression,
            row_group_rows=self.parquet_row_group_rows,
            resume=self._resume_sinks,
            max_committed_tick=self._max_committed_tick,
        )
        if self.recording_tier != "metrics_only":
            self._sinks["ow_state"] = PartitionedParquetSink(
                self.root / "analysis" / "ow_state.parquet",
                compiled.state_schema,
                table_name="ow_state",
                schema_digest=compiled.schema_digest,
                compression=self.compression,
                row_group_rows=self.parquet_row_group_rows,
                resume=self._resume_sinks,
                max_committed_tick=self._max_committed_tick,
            )
            self._sinks["ow_decisions"] = PartitionedParquetSink(
                self.root / "analysis" / "ow_decisions.parquet",
                compiled.decision_schema,
                table_name="ow_decisions",
                schema_digest=compiled.schema_digest,
                compression=self.compression,
                row_group_rows=self.parquet_row_group_rows,
                resume=self._resume_sinks,
                max_committed_tick=self._max_committed_tick,
            )
            if compiled.action_math_schema is not None and self.materialization_mode == "inline":
                self._sinks["ow_action_math"] = PartitionedParquetSink(
                    self.root / "analysis" / "ow_action_math.parquet",
                    compiled.action_math_schema,
                    table_name="ow_action_math",
                    schema_digest=compiled.schema_digest,
                    compression=self.compression,
                    row_group_rows=self.parquet_row_group_rows,
                    resume=self._resume_sinks,
                    max_committed_tick=self._max_committed_tick,
                )
        schema_path = self.root / "schema" / "columnar_schema.json"
        _atomic_json(schema_path, compiled.metadata())
        source_environment = json.loads(
            (self.root / "source_environment.json").read_text(encoding="utf-8")
        )
        columnar_metadata = source_environment.setdefault("columnar_replay", {})
        columnar_metadata["schema_digest"] = compiled.schema_digest
        columnar_metadata["materialization_mode"] = self.materialization_mode
        columnar_metadata["materialization_state"] = (
            "pending" if self.materialization_mode == "deferred" else "complete"
        )
        _atomic_json(self.root / "source_environment.json", source_environment)

    @staticmethod
    def _append(dataset: Any, value: Any) -> None:
        index = int(dataset.shape[0])
        dataset.resize((index + 1, *dataset.shape[1:]))
        dataset[index] = value

    def record(
        self,
        snapshot: VisualSnapshot,
        *,
        diagnostics: dict[str, Any] | None = None,
        device_source: Any | None = None,
    ) -> None:
        if self._closed:
            raise RuntimeError("ReplayRecorder is closed")
        if self._group is None:
            self._initialize(snapshot)
        if self._compiled_schema is None:
            self._initialize_columnar(snapshot)
        assert self._world_shape is not None
        if snapshot.world_shape != self._world_shape:
            raise ValueError("replay world shape changed during run")
        if self._ticks and snapshot.tick <= self._ticks[-1]:
            raise ValueError("replay ticks must be strictly increasing")

        tick = int(snapshot.tick)
        prior_count = len(self._ticks)
        cadc_written = False
        marker = self.root / "replay" / "commits" / f"tick_{tick:08d}.json"
        try:
            # A commit marker, not a partially extended array or table part, is
            # the authoritative boundary. The exception path rolls back every
            # sink so an ordinary error is as recoverable as a process crash.
            self._append(self._arrays["tick"], tick)
            for name, dataset in self._arrays.items():
                if name == "tick":
                    continue
                value = snapshot.arrays.get(name)
                if value is None:
                    raise ValueError(f"required replay field disappeared: {name}")
                self._append(dataset, np.asarray(value))

            cadc_receipt = self._record_cadc(device_source, tick=tick)
            cadc_written = cadc_receipt is not None

            self._record_events(snapshot.events)
            if self.recording_tier != "metrics_only":
                self._write_columnar_analysis(snapshot, device_source=device_source)
            self._record_metrics(snapshot, diagnostics or {})
            self._flush_small_buffers(tick=tick)

            self._ticks.append(tick)
            assert self._compiled_schema is not None
            _atomic_json(
                marker,
                {
                    "tick": tick,
                    "record_index": len(self._ticks) - 1,
                    "columnar_schema_digest": self._compiled_schema.schema_digest,
                    "recording_telemetry": self._recording_telemetry,
                    "materialization_mode": self.materialization_mode,
                    "action_math_materialized": self.materialization_mode == "inline",
                    "expected_action_math_rows": int(
                        self._recording_telemetry.get("expected_action_math_rows", 0)
                    ),
                    "cadc_row_counts": (
                        None if cadc_receipt is None else cadc_receipt.row_counts
                    ),
                },
            )
            _atomic_json(
                self.root / "run_status.json",
                {
                    "schema_version": "owl.replay.status.v1",
                    "state": "RUNNING",
                    "run_id": self.run_id,
                    "requested_ticks": self.requested_ticks,
                    "completed_ticks": len(self._ticks),
                    "last_committed_tick": tick,
                    "created_at": self._created_at,
                    "updated_at": datetime.now(UTC).isoformat(),
                    "recording_backend": self._recording_telemetry.get("selected_backend"),
                    "recording_phase": "tick_committed",
                    "materialization_mode": self.materialization_mode,
                    "materialization_state": (
                        "pending" if self.materialization_mode == "deferred" else "complete"
                    ),
                },
            )
        except Exception:
            marker.unlink(missing_ok=True)
            if len(self._ticks) > prior_count:
                del self._ticks[prior_count:]
            for dataset in self._arrays.values():
                if int(dataset.shape[0]) > prior_count:
                    dataset.resize((prior_count, *dataset.shape[1:]))
            for sink in self._sinks.values():
                sink.rollback_tick(tick)
            if cadc_written and self._cadc_recorder is not None:
                self._cadc_recorder.rollback_tick(tick)
            self._event_rows.clear()
            self._metric_rows.clear()
            self._recording_telemetry = {
                "selected_backend": self.columnar_backend,
                "rolled_back_tick": tick,
            }
            _atomic_json(
                self.root / "run_status.json",
                {
                    "schema_version": "owl.replay.status.v1",
                    "state": "RUNNING",
                    "run_id": self.run_id,
                    "requested_ticks": self.requested_ticks,
                    "completed_ticks": prior_count,
                    "last_committed_tick": self._ticks[-1] if self._ticks else None,
                    "created_at": self._created_at,
                    "updated_at": datetime.now(UTC).isoformat(),
                    "recording_backend": self.columnar_backend,
                    "recording_phase": "tick_rolled_back",
                    "rolled_back_tick": tick,
                    "materialization_mode": self.materialization_mode,
                    "materialization_state": (
                        "pending" if self.materialization_mode == "deferred" else "complete"
                    ),
                },
            )
            raise

    def _record_cadc(self, device_source: Any | None, *, tick: int) -> Any | None:
        if not bool(self.cadc_config.get("enabled", False)):
            return None
        if device_source is None:
            raise RuntimeError("enabled CADC factual recording requires a device source")
        from owl.record.cadc_writer import CADCFactualRecorder
        from owl.record.gpu_replay_staging import collect_cadc_host_packet

        packet = collect_cadc_host_packet(device_source)
        if packet.tick != int(tick):
            raise RuntimeError(f"CADC packet tick {packet.tick} does not match replay tick {tick}")
        if self._cadc_recorder is None:
            self._cadc_recorder = CADCFactualRecorder(
                self.root,
                packet,
                run_id=self.run_id,
                condition=self.condition,
                seed=self.seed,
                config=self.cadc_config,
                compression=str(self.cadc_config.get("compression", self.compression)),
                row_group_rows=int(
                    self.cadc_config.get(
                        "parquet_row_group_rows", self.parquet_row_group_rows
                    )
                ),
                resume=self._cadc_resume,
                max_committed_tick=self._max_committed_tick,
                source_sha256=self.source_sha256,
                config_sha256=self.config_sha256,
            )
        receipt = self._cadc_recorder.record(packet)
        self._recording_telemetry["cadc"] = {
            "row_counts": receipt.row_counts,
            "transfer_bytes": receipt.transfer_bytes,
            "transfer_count": receipt.transfer_count,
            "source_backend": receipt.source_backend,
            "event_overflow": receipt.event_overflow,
        }
        return receipt

    def record_device(
        self,
        source: Any,
        snapshot: VisualSnapshot,
        *,
        diagnostics: dict[str, Any] | None = None,
    ) -> None:
        """Explicit device-aware compatibility entry point used by the controller."""

        self.record(snapshot, diagnostics=diagnostics, device_source=source)

    def _select_action_backend(self, device_source: Any | None) -> str:
        if self.columnar_backend == "numpy_host":
            return "numpy_host"
        if self.columnar_backend == "cupy_staged":
            if device_source is None:
                if self.strict_acceleration:
                    raise RuntimeError("strict cupy_staged recording requires a device source")
                return "numpy_host"
            return "cupy_staged"
        # The full snapshot has already crossed to host for authoritative Zarr.
        # Reusing it avoids a second D2H transfer. Users can force cupy_staged
        # for target profiling and for future deferred/direct materialization.
        return "numpy_host_reuse" if device_source is not None else "numpy_host"

    def _observe_action_batch(
        self,
        *,
        rows: int,
        arrow_bytes: int,
        elapsed_seconds: float,
        builder: Any,
    ) -> None:
        if self._adaptive_policy is None:
            return
        next_rows = self._adaptive_policy.observe(
            rows=rows,
            arrow_bytes=arrow_bytes,
            elapsed_seconds=elapsed_seconds,
        )
        builder.max_batch_rows = int(next_rows)

    def _write_columnar_analysis(
        self, snapshot: VisualSnapshot, *, device_source: Any | None
    ) -> None:
        assert self._batch_builder is not None
        arrays = snapshot.arrays
        living = build_living_index(arrays, world_shape=snapshot.world_shape)
        state_rows = 0
        decision_rows = 0
        action_rows = 0
        started = time.perf_counter()
        cadc_telemetry = self._recording_telemetry.get("cadc")
        state_started = started
        for columnar in self._batch_builder.iter_state_batches(
            arrays, tick=int(snapshot.tick), living=living
        ):
            batch = columnar.to_record_batch(full_validation=False)
            self._sinks["ow_state"].write_batch(batch, tick=int(snapshot.tick))
            state_rows += columnar.row_count
        state_seconds = time.perf_counter() - state_started
        decision_started = time.perf_counter()
        for columnar in self._batch_builder.iter_decision_batches(
            arrays, tick=int(snapshot.tick), living=living
        ):
            batch = columnar.to_record_batch(full_validation=False)
            self._sinks["ow_decisions"].write_batch(batch, tick=int(snapshot.tick))
            decision_rows += columnar.row_count
        decision_seconds = time.perf_counter() - decision_started

        sampled = self.recording_tier == "analysis_sampled"
        if sampled:
            selected_ids = np.asarray(arrays["occupancy"]).reshape(-1)[living.flat]
            expected_action_rows = int(np.count_nonzero((selected_ids % 32) == 0)) * len(
                self.action_names
            )
        else:
            expected_action_rows = int(living.count) * len(self.action_names)
        action_started = time.perf_counter()
        backend = self._select_action_backend(device_source)
        if "ow_action_math" in self._sinks:
            if backend == "cupy_staged":
                try:
                    from owl.record.gpu_replay_staging import CuPyActionMathBatchBuilder

                    assert self._compiled_schema is not None
                    gpu_builder = CuPyActionMathBatchBuilder(
                        self._compiled_schema,
                        condition=self.condition,
                        seed=self.seed,
                        action_names=self.action_names,
                        max_batch_rows=self.max_batch_rows,
                        max_batch_bytes=self.max_batch_bytes,
                        max_pinned_pool_bytes=self.pinned_pool_bytes,
                    )
                    iterator = gpu_builder.iter_batches(
                        device_source, tick=int(snapshot.tick), sampled=sampled
                    )
                    for columnar in iterator:
                        write_started = time.perf_counter()
                        batch = columnar.to_record_batch(full_validation=False)
                        arrow_bytes = int(batch.get_total_buffer_size())
                        self._sinks["ow_action_math"].write_batch(batch, tick=int(snapshot.tick))
                        elapsed = time.perf_counter() - write_started
                        action_rows += columnar.row_count
                        self._observe_action_batch(
                            rows=columnar.row_count,
                            arrow_bytes=arrow_bytes,
                            elapsed_seconds=elapsed,
                            builder=gpu_builder,
                        )
                    gpu_telemetry = gpu_builder.telemetry
                    transfer = {
                        "gpu_batches": gpu_telemetry.batches,
                        "gpu_transfer_bytes": gpu_telemetry.transfer_bytes,
                        "gpu_transfer_count": gpu_telemetry.transfer_count,
                    }
                except Exception:
                    if self.strict_acceleration:
                        raise
                    backend = "numpy_host_fallback"
                    transfer = {"gpu_fallback": True}
                    for columnar in self._batch_builder.iter_action_math_batches(
                        arrays, tick=int(snapshot.tick), living=living, sampled=sampled
                    ):
                        write_started = time.perf_counter()
                        batch = columnar.to_record_batch(full_validation=False)
                        arrow_bytes = int(batch.get_total_buffer_size())
                        self._sinks["ow_action_math"].write_batch(batch, tick=int(snapshot.tick))
                        elapsed = time.perf_counter() - write_started
                        action_rows += columnar.row_count
                        self._observe_action_batch(
                            rows=columnar.row_count,
                            arrow_bytes=arrow_bytes,
                            elapsed_seconds=elapsed,
                            builder=self._batch_builder,
                        )
            else:
                transfer = {}
                for columnar in self._batch_builder.iter_action_math_batches(
                    arrays, tick=int(snapshot.tick), living=living, sampled=sampled
                ):
                    write_started = time.perf_counter()
                    batch = columnar.to_record_batch(full_validation=False)
                    arrow_bytes = int(batch.get_total_buffer_size())
                    self._sinks["ow_action_math"].write_batch(batch, tick=int(snapshot.tick))
                    elapsed = time.perf_counter() - write_started
                    action_rows += columnar.row_count
                    self._observe_action_batch(
                        rows=columnar.row_count,
                        arrow_bytes=arrow_bytes,
                        elapsed_seconds=elapsed,
                        builder=self._batch_builder,
                    )
        else:
            transfer = {}
        action_seconds = time.perf_counter() - action_started
        total_seconds = time.perf_counter() - started
        self._recording_telemetry = {
            "selected_backend": (
                "deferred" if self.materialization_mode == "deferred" else backend
            ),
            "living_ows": living.count,
            "state_rows": state_rows,
            "decision_rows": decision_rows,
            "action_math_rows": action_rows,
            "expected_action_math_rows": expected_action_rows,
            "state_seconds": state_seconds,
            "decision_seconds": decision_seconds,
            "action_seconds": action_seconds,
            "record_total_seconds": total_seconds,
            "max_batch_rows": self.max_batch_rows,
            "max_batch_bytes": self.max_batch_bytes,
            "adaptive_batching": self.adaptive_batching,
            "adaptive_policy": (
                self._adaptive_policy.telemetry.to_dict()
                if self._adaptive_policy is not None
                else None
            ),
            **transfer,
        }
        if cadc_telemetry is not None:
            self._recording_telemetry["cadc"] = cadc_telemetry

    def _record_events(self, events: tuple[VisualEvent, ...]) -> None:
        for sequence, event in enumerate(events):
            row = asdict(event)
            row["event_type"] = event.event_type.name
            row["event_type_code"] = int(event.event_type)
            row["event_sequence"] = int(sequence)
            row["event_id"] = f"{event.tick}:{sequence}:{event.source_id}:{int(event.event_type)}"
            self._event_rows.append(row)

    def _record_metrics(self, snapshot: VisualSnapshot, diagnostics: dict[str, Any]) -> None:
        health = np.asarray(snapshot.arrays["health"], dtype=float)
        resource = np.asarray(snapshot.arrays.get("resource", np.zeros_like(health)), dtype=float)
        live = health > 0
        row: dict[str, Any] = {
            "tick": int(snapshot.tick),
            "population": int(np.count_nonzero(live)),
            "mean_health": float(np.mean(health[live])) if np.any(live) else 0.0,
            "mean_resource": float(np.mean(resource[live])) if np.any(live) else 0.0,
            "food_total": float(np.sum(np.asarray(snapshot.arrays.get("food", 0.0)))),
            "toxin_total": float(np.sum(np.asarray(snapshot.arrays.get("toxin", 0.0)))),
            "waste_total": float(np.sum(np.asarray(snapshot.arrays.get("waste", 0.0)))),
            "events": sum(1 for event in self._event_rows if int(event["tick"]) == snapshot.tick),
        }
        readout = snapshot.arrays.get("raqic_readout", snapshot.arrays.get("readout"))
        if readout is not None:
            values = np.asarray(readout)[live].astype(np.int64, copy=False)
            counts = np.bincount(values[values >= 0], minlength=max(1, len(self.action_names)))
            row["action_histogram"] = json.dumps(counts.astype(int).tolist())
        profile = diagnostics.get("profile", {})
        if isinstance(profile, dict):
            row["profile"] = json.dumps(profile, sort_keys=True, default=str)
        validation = diagnostics.get("qiskit_validation")
        row["qiskit_validation"] = (
            json.dumps(validation, sort_keys=True, default=str) if validation else ""
        )
        self._metric_rows.append(row)

    @staticmethod
    def _append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = sorted({key for row in rows for key in row})
        exists = path.exists() and path.stat().st_size > 0
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            if not exists:
                writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        key: json.dumps(value, sort_keys=True)
                        if isinstance(value, (dict, list, tuple))
                        else value
                        for key, value in row.items()
                    }
                )

    def _flush_small_buffers(self, *, tick: int | None = None) -> None:
        if self._event_rows:
            if "events" not in self._sinks:
                raise RuntimeError("event sink is not initialized")
            batch = _events_record_batch(self._event_rows, self._sinks["events"].schema)
            event_tick = int(tick if tick is not None else self._event_rows[-1]["tick"])
            self._sinks["events"].write_batch(batch, tick=event_tick)
            self._append_csv(self.root / "analysis" / "events.csv", self._event_rows)
            self._event_rows.clear()
        if self._metric_rows:
            self._append_csv(self.root / "analysis" / "tick_metrics.csv", self._metric_rows)
            self._metric_rows.clear()

    def _validate_before_close(self) -> None:
        if self._group is None or self._world_shape is None:
            raise RuntimeError("cannot close an empty replay bundle")
        expected = len(self._ticks)
        for name, dataset in self._arrays.items():
            if int(dataset.shape[0]) != expected:
                raise RuntimeError(
                    f"replay dataset length mismatch for {name}: {dataset.shape[0]} != {expected}"
                )
        commit_count = len(tuple((self.root / "replay" / "commits").glob("tick_*.json")))
        if commit_count != expected:
            raise RuntimeError(f"replay commit count mismatch: {commit_count} != {expected}")
        if self._cadc_recorder is not None and len(self._cadc_recorder._ticks) != expected:
            raise RuntimeError("CADC committed tick count differs from replay")

    def refresh_checksums(self) -> None:
        """Recompute checksums after trusted controller metadata is synchronized."""

        if not self._closed:
            raise RuntimeError("refresh_checksums requires a closed replay recorder")
        self._write_checksums()

    def close(self, *, state: str = "SUCCEEDED", failure: str | None = None) -> ReplayManifest:
        if self._closed:
            return ReplayManifest.load(self.root)
        self._closed = True
        self._flush_small_buffers()
        if self._cadc_recorder is not None:
            self._cadc_recorder.close(finalize_pending=state != "INTERRUPTED_RESUMABLE")
        for sink in self._sinks.values():
            sink.close()
        self._validate_before_close()
        if self._group is None or self._world_shape is None:
            raise RuntimeError("cannot close an empty replay bundle")
        if self.materialization_mode == "deferred":
            manifest_materialization_state = "failed" if state == "FAILED_PARTIAL" else "pending"
        else:
            manifest_materialization_state = "complete"
        manifest = ReplayManifest(
            schema_version="owl.replay.v1",
            run_id=self.run_id,
            condition=self.condition,
            seed=self.seed,
            requested_ticks=self.requested_ticks,
            completed_ticks=len(self._ticks),
            world_shape=self._world_shape,
            boundary_mode=self._boundary_mode,
            recording_tier=self.recording_tier,
            source_sha256=self.source_sha256,
            config_sha256=self.config_sha256,
            action_names=self.action_names,
            array_fields=tuple(sorted(self._field_specs)),
            created_at=self._created_at,
            hardware=self.hardware,
            qiskit_execution=self.qiskit_execution,
            materialization_mode=self.materialization_mode,
            materialization_state=manifest_materialization_state,
            columnar_schema_digest=(
                self._compiled_schema.schema_digest
                if self._compiled_schema is not None
                else "unknown"
            ),
        )
        _atomic_json(self.root / "run_manifest.json", manifest.to_dict())
        effective_state = (
            "SCIENTIFIC_ARRAYS_COMPLETE_MATERIALIZATION_PENDING"
            if state == "SUCCEEDED" and self.materialization_mode == "deferred"
            else state
        )
        _atomic_json(
            self.root / "run_status.json",
            {
                "schema_version": "owl.replay.status.v1",
                "state": effective_state,
                "run_id": self.run_id,
                "requested_ticks": self.requested_ticks,
                "completed_ticks": len(self._ticks),
                "last_committed_tick": self._ticks[-1] if self._ticks else None,
                "failure": failure,
                "created_at": self._created_at,
                "closed_at": datetime.now(UTC).isoformat(),
                "materialization_mode": self.materialization_mode,
                "materialization_state": manifest_materialization_state,
                "columnar_schema_digest": manifest.columnar_schema_digest,
            },
        )
        self._write_checksums()
        return manifest

    def _write_checksums(self) -> None:
        entries: list[str] = []
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or "checksums" in path.parts:
                continue
            entries.append(f"{sha256_file(path)}  {path.relative_to(self.root).as_posix()}")
        target = self.root / "checksums" / "SHA256SUMS.txt"
        target.write_text("\n".join(entries) + "\n", encoding="utf-8")
