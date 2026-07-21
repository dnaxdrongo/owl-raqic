"""Deferred, resumable action-math materialization for replay schema ."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from owl.record.action_math_batch import NumPyReplayBatchBuilder, build_living_index
from owl.record.parquet_sink import PartitionedParquetSink
from owl.record.replay_schema import compile_replay_schema
from owl.replay.manifest import ReplayManifest, sha256_file

PENDING_STATE = "SCIENTIFIC_ARRAYS_COMPLETE_MATERIALIZATION_PENDING"
MATERIALIZING_STATE = "MATERIALIZING"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_checksums(root: Path) -> None:
    entries: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "checksums" in path.parts:
            continue
        entries.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    target = root / "checksums" / "SHA256SUMS.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(entries) + "\n", encoding="utf-8")


def _batch_digest(batch: Any) -> bytes:
    """Return deterministic Arrow IPC bytes for semantic verification."""

    import pyarrow as pa

    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, batch.schema) as writer:
        writer.write_batch(batch)
    return bytes(sink.getvalue())


def _read_journal(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": "owl.replay.action-materialization.v1",
            "state": "pending",
            "completed_ticks": [],
            "ticks": {},
        }
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError("invalid action materialization journal")
    value: dict[str, Any] = raw
    if value.get("schema_version") != "owl.replay.action-materialization.v1":
        raise RuntimeError("unsupported action materialization journal")
    return value


def _update_outer_progress(
    bundle: Path, *, state: str, phase: str, error: str | None = None
) -> None:
    run_root = bundle.parent if bundle.name == "bundle" else None
    if run_root is None:
        return
    progress_path = run_root / "run_progress.json"
    if not progress_path.exists():
        return
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress.update(
        {
            "state": state,
            "phase": phase,
            "updated_at": datetime.now(UTC).isoformat(),
        }
    )
    if error is not None:
        progress["error"] = error
    _atomic_json(progress_path, progress)


def materialize_action_math(
    bundle_root: str | Path,
    *,
    max_batch_rows: int = 131_072,
    max_batch_bytes: int = 128 * 1024 * 1024,
    row_group_rows: int = 131_072,
    compression: str = "zstd",
) -> dict[str, Any]:
    """Derive the canonical long-form action table from committed dense Zarr arrays.

    The operation is append-safe and resumable. It is intentionally CPU/NumPy
    based so paid GPU simulation can finish before this redundant representation
    is produced on a cheaper machine. The same schema and batch builder used by
    inline recording are reused here.
    """

    import zarr

    root = Path(bundle_root)
    manifest = ReplayManifest.load(root)
    status_path = root / "run_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if manifest.materialization_mode != "deferred":
        raise RuntimeError("bundle was not created with deferred materialization")
    if status.get("state") == "SUCCEEDED" and manifest.materialization_state == "complete":
        return {
            "state": "SUCCEEDED",
            "completed_ticks": manifest.completed_ticks,
            "already_complete": True,
        }
    if status.get("state") not in {PENDING_STATE, MATERIALIZING_STATE, "FAILED_PARTIAL"}:
        raise RuntimeError(f"bundle is not materializable from state {status.get('state')!r}")

    checksums = root / "checksums" / "SHA256SUMS.txt"
    checksums.unlink(missing_ok=True)
    status.update(
        {
            "state": MATERIALIZING_STATE,
            "materialization_state": "materializing",
            "materialization_started_at": datetime.now(UTC).isoformat(),
        }
    )
    _atomic_json(status_path, status)
    _update_outer_progress(root, state=MATERIALIZING_STATE, phase="materialize_action_math")

    journal_path = root / "analysis" / "action_materialization_journal.json"
    journal = _read_journal(journal_path)
    completed_ticks = [int(value) for value in journal.get("completed_ticks", [])]
    last_completed = max(completed_ticks, default=-1)

    group = zarr.open_group(str(root / "replay" / "replay.zarr"), mode="r")
    ticks = np.asarray(group["tick"][:], dtype=np.int64)
    if ticks.size != manifest.completed_ticks:
        raise RuntimeError("Zarr tick count differs from replay manifest")
    state_group = group["state"]
    field_names = tuple(str(name) for name in state_group.array_keys())
    if not field_names:
        raise RuntimeError("deferred materializer found no dense state arrays")

    first_arrays = {name: np.asarray(state_group[name][0]) for name in field_names}
    compiled = compile_replay_schema(
        first_arrays,
        world_shape=manifest.world_shape,
        action_names=manifest.action_names,
        recording_tier=manifest.recording_tier,
    )
    if compiled.action_math_schema is None:
        raise RuntimeError("recording tier has no canonical action-math table")
    if manifest.columnar_schema_digest not in {"unknown", compiled.schema_digest}:
        raise RuntimeError("deferred source schema digest differs from replay manifest")
    prior_digest = str(journal.get("columnar_schema_digest", compiled.schema_digest))
    if prior_digest != compiled.schema_digest:
        raise RuntimeError("materialization journal schema digest mismatch")

    builder = NumPyReplayBatchBuilder(
        compiled,
        condition=manifest.condition,
        seed=manifest.seed,
        action_names=manifest.action_names,
        max_batch_rows=max_batch_rows,
        max_batch_bytes=max_batch_bytes,
        full_validation=False,
    )
    sink = PartitionedParquetSink(
        root / "analysis" / "ow_action_math.parquet",
        compiled.action_math_schema,
        table_name="ow_action_math",
        schema_digest=compiled.schema_digest,
        compression=compression,
        row_group_rows=row_group_rows,
        resume=True,
        max_committed_tick=last_completed,
    )

    try:
        for record_index, raw_tick in enumerate(ticks.tolist()):
            tick = int(raw_tick)
            if tick in completed_ticks:
                continue
            arrays = {name: np.asarray(state_group[name][record_index]) for name in field_names}
            # Fail before writing if a later tick changes any declared source shape/dtype.
            current = compile_replay_schema(
                arrays,
                world_shape=manifest.world_shape,
                action_names=manifest.action_names,
                recording_tier=manifest.recording_tier,
            )
            if current.schema_digest != compiled.schema_digest:
                raise RuntimeError(f"columnar schema changed at tick {tick}")
            living = build_living_index(arrays, world_shape=manifest.world_shape)
            sampled = manifest.recording_tier == "analysis_sampled"
            digest = hashlib.sha256()
            rows = 0
            parts_before = sink.parts_written
            for columnar in builder.iter_action_math_batches(
                arrays,
                tick=tick,
                living=living,
                sampled=sampled,
            ):
                batch = columnar.to_record_batch(full_validation=False)
                digest.update(_batch_digest(batch))
                sink.write_batch(batch, tick=tick)
                rows += int(batch.num_rows)
            if sampled:
                live_ids = np.asarray(arrays["occupancy"]).reshape(-1)[living.flat]
                expected_rows = int(np.count_nonzero((live_ids % 32) == 0)) * len(
                    manifest.action_names
                )
            else:
                expected_rows = living.count * len(manifest.action_names)
            if rows != expected_rows:
                sink.rollback_tick(tick)
                raise RuntimeError(
                    f"deferred action row count mismatch at tick {tick}: {rows} != {expected_rows}"
                )
            completed_ticks.append(tick)
            journal.setdefault("ticks", {})[str(tick)] = {
                "record_index": int(record_index),
                "living_ows": int(living.count),
                "expected_rows": int(expected_rows),
                "rows": int(rows),
                "parts_written": int(sink.parts_written - parts_before),
                "semantic_digest": digest.hexdigest(),
                "completed_at": datetime.now(UTC).isoformat(),
            }
            journal.update(
                {
                    "state": "materializing",
                    "columnar_schema_digest": compiled.schema_digest,
                    "completed_ticks": sorted(completed_ticks),
                    "last_completed_tick": tick,
                    "updated_at": datetime.now(UTC).isoformat(),
                }
            )
            _atomic_json(journal_path, journal)
            status.update(
                {
                    "state": MATERIALIZING_STATE,
                    "materialization_state": "materializing",
                    "materialized_ticks": len(completed_ticks),
                    "last_materialized_tick": tick,
                    "materialization_rows": sink.rows_written,
                    "materialization_parts": sink.parts_written,
                    "updated_at": datetime.now(UTC).isoformat(),
                }
            )
            _atomic_json(status_path, status)
        sink.close()
    except Exception as exc:
        with np.errstate(all="ignore"):
            sink.close()
        journal.update(
            {
                "state": "failed",
                "error": repr(exc),
                "updated_at": datetime.now(UTC).isoformat(),
            }
        )
        _atomic_json(journal_path, journal)
        status.update(
            {
                "state": "FAILED_PARTIAL",
                "materialization_state": "failed",
                "failure": repr(exc),
                "updated_at": datetime.now(UTC).isoformat(),
            }
        )
        _atomic_json(status_path, status)
        _update_outer_progress(
            root, state="FAILED_PARTIAL", phase="materialization_failed", error=repr(exc)
        )
        raise

    journal.update(
        {
            "state": "complete",
            "completed_ticks": sorted(completed_ticks),
            "completed_at": datetime.now(UTC).isoformat(),
            "rows_written": sink.rows_written,
            "parts_written": sink.parts_written,
        }
    )
    _atomic_json(journal_path, journal)
    updated_manifest = replace(
        manifest,
        materialization_state="complete",
        columnar_schema_digest=compiled.schema_digest,
    )
    _atomic_json(root / "run_manifest.json", updated_manifest.to_dict())
    environment_path = root / "source_environment.json"
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    columnar = environment.setdefault("columnar_replay", {})
    columnar.update(
        {
            "materialization_mode": "deferred",
            "materialization_state": "complete",
            "materialization_backend": "numpy_host",
            "schema_digest": compiled.schema_digest,
        }
    )
    _atomic_json(environment_path, environment)
    status.update(
        {
            "state": "SUCCEEDED",
            "materialization_state": "complete",
            "materialized_ticks": len(completed_ticks),
            "last_materialized_tick": completed_ticks[-1] if completed_ticks else None,
            "materialization_rows": sink.rows_written,
            "materialization_parts": sink.parts_written,
            "materialization_completed_at": datetime.now(UTC).isoformat(),
            "failure": None,
        }
    )
    _atomic_json(status_path, status)
    _write_checksums(root)
    _update_outer_progress(root, state="SUCCEEDED", phase="materialization_complete")
    return {
        "state": "SUCCEEDED",
        "completed_ticks": len(completed_ticks),
        "rows_written": sink.rows_written,
        "parts_written": sink.parts_written,
        "schema_digest": compiled.schema_digest,
        "journal": str(journal_path),
    }
