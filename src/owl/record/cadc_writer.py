"""Vectorized Arrow/Parquet writer for factual CADC evidence."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from owl.record.cadc_schema import (
    CADC_ACTION_COUNT,
    CADC_ACTION_TRANSITION_SCHEMA_VERSION,
    CADC_SCHEMA_DIGEST,
    schema_manifest,
)
from owl.record.gpu_replay_staging import CADCHostPacket
from owl.record.parquet_sink import PartitionedParquetSink


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _pa() -> Any:
    import pyarrow as pa

    return pa


def _schema(fields: list[Any]) -> Any:
    pa = _pa()
    metadata = {
        b"owl.cadc.schema_version": b"owl.cadc.factual.v1",
        b"owl.cadc.schema_digest": CADC_SCHEMA_DIGEST.encode(),
    }
    return pa.schema(fields, metadata=metadata)


def _base_fields() -> list[Any]:
    pa = _pa()
    return [
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("condition", pa.string(), nullable=False),
        pa.field("seed", pa.int64(), nullable=False),
        pa.field("tick", pa.int64(), nullable=False),
        pa.field("decision_sequence", pa.int64(), nullable=False),
        pa.field("ow_id", pa.int64(), nullable=False),
        pa.field("source_y", pa.int32(), nullable=False),
        pa.field("source_x", pa.int32(), nullable=False),
    ]


def _fixed_list(dtype: Any, size: int) -> Any:
    # Parquet restores the standard Arrow nested child name as ``element``.
    # Declare it explicitly so a freshly compiled resume schema is identical
    # to the on-disk schema, including nested field names.
    return _pa().list_(_pa().field("element", dtype), int(size))


def compile_cadc_schemas(packet: CADCHostPacket) -> dict[str, Any]:
    pa = _pa()
    f = pa.float32()
    base = _base_fields()
    agent_scalars = sorted(
        name
        for name, value in packet.arrays.items()
        if name.startswith("agent_") and value.ndim == 2
    )
    agent_vectors = sorted(
        name
        for name, value in packet.arrays.items()
        if name.startswith("agent_") and value.ndim == 3
    )
    oracle_scalars = sorted(
        name
        for name, value in packet.arrays.items()
        if name.startswith("oracle_") and value.ndim == 2
    )
    oracle_vectors = sorted(
        name
        for name, value in packet.arrays.items()
        if name.startswith("oracle_") and value.ndim == 3
    )
    action_transitions = "candidate_compiled_action" in packet.arrays
    candidate_extension_fields = (
        [
            pa.field("target_source", pa.int16(), nullable=False),
            pa.field("target_distance", f, nullable=False),
            pa.field("target_confidence", f, nullable=False),
            pa.field("compiled_action", pa.int16(), nullable=False),
        ]
        if action_transitions
        else []
    )
    execution_extension_fields = (
        [
            pa.field("compiled_execution_action", pa.int16(), nullable=False),
            pa.field("intent_target_y", pa.int32(), nullable=False),
            pa.field("intent_target_x", pa.int32(), nullable=False),
            pa.field("intent_target_ow_id", pa.int64(), nullable=False),
            pa.field("intent_target_kind", pa.int16(), nullable=False),
            pa.field("intent_target_source", pa.int16(), nullable=False),
            *[
                pa.field(name, f, nullable=False)
                for name in (
                    "intent_target_distance_before",
                    "intent_target_distance_after",
                    "intent_known_hazard_before",
                    "intent_known_hazard_after",
                    "intent_contact_opportunity_before",
                    "intent_contact_opportunity_after",
                )
            ],
        ]
        if action_transitions
        else []
    )
    information_extension_fields = (
        [
            pa.field("new_cell_count", pa.int32(), nullable=False),
            pa.field("new_target_count", pa.int32(), nullable=False),
            pa.field("active_memory_changed", pa.bool_(), nullable=False),
            pa.field("information_execution_success", pa.bool_(), nullable=False),
            pa.field("no_new_information", pa.bool_(), nullable=False),
            *[
                pa.field(name, f, nullable=False)
                for name in (
                    "sensed_food_before",
                    "sensed_food_after",
                    "sensed_toxin_before",
                    "sensed_toxin_after",
                    "sensed_alive_before",
                    "sensed_alive_after",
                )
            ],
        ]
        if action_transitions
        else []
    )
    schemas = {
        "decisions": _schema(
            base
            + [
                pa.field("ow_type", pa.int16(), nullable=False),
                pa.field("lineage_id", pa.int64(), nullable=False),
                pa.field("parent_id", pa.int64(), nullable=False),
                pa.field("age", pa.int32(), nullable=False),
                pa.field("development_stage", pa.int16(), nullable=False),
                pa.field("selected_action", pa.int16(), nullable=False),
                pa.field("selected_target_y", pa.int32(), nullable=False),
                pa.field("selected_target_x", pa.int32(), nullable=False),
                pa.field("selected_target_ow_id", pa.int64(), nullable=False),
                pa.field("selected_probability", f, nullable=False),
                pa.field("agent_context_ref", pa.int64(), nullable=False),
                pa.field("oracle_context_ref", pa.int64(), nullable=False),
                pa.field("dense_context_ref", pa.int64(), nullable=False),
            ]
        ),
        "agent_context": _schema(
            base
            + [
                pa.field(
                    name,
                    pa.int32()
                    if packet.arrays[name].dtype.kind in {"i", "u"}
                    else pa.bool_()
                    if packet.arrays[name].dtype.kind == "b"
                    else f,
                    nullable=False,
                )
                for name in agent_scalars
            ]
            + [
                pa.field(name, _fixed_list(f, int(packet.arrays[name].shape[-1])), nullable=False)
                for name in agent_vectors
            ]
        ),
        "oracle_context": _schema(
            base
            + [
                pa.field(
                    name,
                    pa.int64()
                    if packet.arrays[name].dtype.kind in {"i", "u"}
                    else pa.bool_()
                    if packet.arrays[name].dtype.kind == "b"
                    else f,
                    nullable=False,
                )
                for name in oracle_scalars
            ]
            + [
                pa.field(name, _fixed_list(f, int(packet.arrays[name].shape[-1])), nullable=False)
                for name in oracle_vectors
            ]
        ),
        "candidates": _schema(
            base
            + [
                pa.field("candidate_sequence", pa.int64(), nullable=False),
                pa.field("action_index", pa.int16(), nullable=False),
                pa.field("target_kind", pa.int8(), nullable=False),
                pa.field("proposed_y", pa.int32(), nullable=False),
                pa.field("proposed_x", pa.int32(), nullable=False),
                pa.field("resolved_y", pa.int32(), nullable=False),
                pa.field("resolved_x", pa.int32(), nullable=False),
                pa.field("target_ow_id", pa.int64(), nullable=False),
                pa.field("destination_occupancy", pa.int64(), nullable=False),
                pa.field("destination_obstacle", pa.bool_(), nullable=False),
                pa.field("destination_food", f, nullable=False),
                pa.field("destination_toxin", f, nullable=False),
                pa.field("opportunity_count", pa.int16(), nullable=False),
                pa.field("utility", f, nullable=False),
                pa.field("policy_legal", pa.bool_(), nullable=False),
                pa.field("prechoice_executable", pa.bool_(), nullable=False),
                pa.field("prechoice_reason_code", pa.int16(), nullable=False),
                *candidate_extension_fields,
            ]
        ),
        "execution": _schema(
            base
            + [
                pa.field("selected_action", pa.int16(), nullable=False),
                pa.field("attempted_action", pa.int16(), nullable=False),
                pa.field("realized_action", pa.int16(), nullable=False),
                pa.field("execution_success", pa.bool_(), nullable=False),
                pa.field("execution_reason_code", pa.int16(), nullable=False),
                pa.field("realized_target_y", pa.int32(), nullable=False),
                pa.field("realized_target_x", pa.int32(), nullable=False),
                pa.field("realized_target_ow_id", pa.int64(), nullable=False),
                *execution_extension_fields,
            ]
            + [
                pa.field(name, f, nullable=False)
                for name in (
                    "amount_consumed",
                    "amount_transferred",
                    "amount_repaired",
                    "amount_damaged",
                    "amount_emitted",
                    "amount_received",
                    "direct_cost",
                )
            ]
        ),
        "events": _schema(
            [
                pa.field("run_id", pa.string(), nullable=False),
                pa.field("condition", pa.string(), nullable=False),
                pa.field("seed", pa.int64(), nullable=False),
                pa.field("tick", pa.int64(), nullable=False),
                pa.field("event_sequence", pa.int64(), nullable=False),
                pa.field("decision_sequence", pa.int64(), nullable=False),
                pa.field("event_code", pa.int16(), nullable=False),
                pa.field("stage_code", pa.int16(), nullable=False),
                pa.field("reason_code", pa.int16(), nullable=False),
                pa.field("actor_ow_id", pa.int64(), nullable=False),
                pa.field("source_y", pa.int32(), nullable=False),
                pa.field("source_x", pa.int32(), nullable=False),
                pa.field("target_ow_id", pa.int64(), nullable=False),
                pa.field("target_y", pa.int32(), nullable=False),
                pa.field("target_x", pa.int32(), nullable=False),
                *[pa.field(f"payload{index}", f, nullable=False) for index in range(4)],
            ]
        ),
        "contributions": _schema(
            base
            + [
                pa.field("contribution_sequence", pa.int64(), nullable=False),
                pa.field("contribution_code", pa.int16(), nullable=False),
                *[
                    pa.field(f"delta_{name}", f, nullable=False)
                    for name in packet.contribution_fields
                ],
                *[
                    pa.field(f"start_{name}", f, nullable=False)
                    for name in packet.contribution_fields
                ],
                *[
                    pa.field(f"end_{name}", f, nullable=False)
                    for name in packet.contribution_fields
                ],
            ]
        ),
        "information": _schema(
            base
            + [
                pa.field("information_kind", pa.int8(), nullable=False),
                pa.field("pre_observation_ref", pa.int64(), nullable=False),
                pa.field("post_memory_ref", pa.int64(), nullable=False),
                pa.field("pre_signal_sum", f, nullable=False),
                pa.field("post_signal_memory_sum", f, nullable=False),
                pa.field("memory_delta", f, nullable=False),
                pa.field("amount_emitted", f, nullable=False),
                pa.field("amount_received", f, nullable=False),
                pa.field("followup_tick", pa.int64(), nullable=False),
                pa.field("timing_code", pa.int8(), nullable=False),
                pa.field("receiver_count", pa.int32(), nullable=False),
                pa.field("receiver_link_status", pa.int8(), nullable=False),
                *information_extension_fields,
                pa.field(
                    "observation_before",
                    _fixed_list(f, packet.channel_count),
                    nullable=False,
                ),
                pa.field(
                    "memory_before", _fixed_list(f, packet.channel_count), nullable=False
                ),
                pa.field(
                    "memory_after", _fixed_list(f, packet.channel_count), nullable=False
                ),
                pa.field(
                    "emitted_channels", _fixed_list(f, packet.channel_count), nullable=False
                ),
                pa.field(
                    "received_channels", _fixed_list(f, packet.channel_count), nullable=False
                ),
            ]
        ),
        "information_followups": _schema(
            [
                pa.field("run_id", pa.string(), nullable=False),
                pa.field("condition", pa.string(), nullable=False),
                pa.field("seed", pa.int64(), nullable=False),
                pa.field("tick", pa.int64(), nullable=False),
                pa.field("source_decision_sequence", pa.int64(), nullable=False),
                pa.field("source_ow_id", pa.int64(), nullable=False),
                pa.field("followup_decision_sequence", pa.int64(), nullable=False),
                pa.field("followup_observation_ref", pa.int64(), nullable=False),
                pa.field("followup_status", pa.int8(), nullable=False),
            ]
        ),
    }
    if action_transitions:
        schemas["action_directions"] = _schema(
            base
            + [
                pa.field("direction_sequence", pa.int64(), nullable=False),
                pa.field("action_family", pa.int8(), nullable=False),
                pa.field("direction_index", pa.int8(), nullable=False),
                pa.field("target_y", pa.int32(), nullable=False),
                pa.field("target_x", pa.int32(), nullable=False),
                pa.field("target_ow_id", pa.int64(), nullable=False),
                pa.field("target_kind", pa.int16(), nullable=False),
                pa.field("target_source", pa.int16(), nullable=False),
                pa.field("target_distance", f, nullable=False),
                pa.field("target_confidence", f, nullable=False),
                pa.field("direction_y", pa.int32(), nullable=False),
                pa.field("direction_x", pa.int32(), nullable=False),
                pa.field("direction_executable", pa.bool_(), nullable=False),
                pa.field("direction_score", f, nullable=False),
                pa.field("distance_delta", f, nullable=False),
                pa.field("known_hazard", f, nullable=False),
                pa.field("opportunity", f, nullable=False),
            ]
        )
    dense_names = sorted(name for name in packet.arrays if name.startswith("dense_oracle_"))
    if dense_names:
        dense_fields = []
        for name in dense_names:
            value = packet.arrays[name]
            dtype = (
                pa.int64()
                if value.dtype.kind in {"i", "u"}
                else pa.bool_()
                if value.dtype.kind == "b"
                else f
            )
            dense_fields.append(
                pa.field(name, _fixed_list(dtype, int(value.shape[-1])), nullable=False)
            )
        schemas["dense_context"] = _schema(
            base
            + [
                pa.field("dense_context_id", pa.int64(), nullable=False),
                pa.field("chunk_id", pa.int64(), nullable=False),
                pa.field("chunk_offset", pa.int64(), nullable=False),
                pa.field("radius", pa.int16(), nullable=False),
                *dense_fields,
            ]
        )
    metadata = {
        b"owl.cadc.schema_version": packet.schema_version.encode(),
        b"owl.cadc.schema_digest": packet.schema_digest.encode(),
    }
    return {name: schema.with_metadata(metadata) for name, schema in schemas.items()}


def _string_column(value: str, rows: int) -> np.ndarray:
    return np.full(rows, value, dtype=object)


def _array_digest(arrays: dict[str, np.ndarray], names: list[str]) -> str:
    digest = hashlib.sha256()
    for name in sorted(names):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _record_batch(schema: Any, columns: dict[str, Any]) -> Any:
    pa = _pa()
    arrays = []
    for field in schema:
        value = columns[field.name]
        if pa.types.is_fixed_size_list(field.type):
            values = np.ascontiguousarray(value).reshape(-1)
            child = pa.array(values, type=field.type.value_type, from_pandas=False)
            array = pa.FixedSizeListArray.from_arrays(child, field.type.list_size)
        else:
            array = pa.array(value, type=field.type, from_pandas=False, safe=True)
        arrays.append(array)
    batch = pa.RecordBatch.from_arrays(arrays, schema=schema)
    batch.validate(full=False)
    return batch


@dataclass(frozen=True)
class CADCTickReceipt:
    tick: int
    row_counts: dict[str, int]
    transfer_bytes: int
    transfer_count: int
    source_backend: str
    event_overflow: int


class CADCFactualRecorder:
    """Own the additive factual tables and their append/rollback lifecycle."""

    def __init__(
        self,
        root: str | Path,
        packet: CADCHostPacket,
        *,
        run_id: str,
        condition: str,
        seed: int,
        config: dict[str, Any],
        compression: str,
        row_group_rows: int,
        resume: bool = False,
        max_committed_tick: int | None = None,
        source_sha256: str = "unknown",
        config_sha256: str = "unknown",
    ) -> None:
        action_transitions = packet.schema_version == CADC_ACTION_TRANSITION_SCHEMA_VERSION
        manifest = schema_manifest(action_transitions=action_transitions)
        if packet.schema_digest != manifest["schema_digest"]:
            raise RuntimeError("CADC device/host schema digest mismatch")
        self.schema_version = packet.schema_version
        self.schema_digest = packet.schema_digest
        self.action_transitions = action_transitions
        self.root = Path(root) / "analysis" / ("cadc_v2" if action_transitions else "cadc_v1")
        self.root.mkdir(parents=True, exist_ok=True)
        self.run_id = str(run_id)
        self.condition = str(condition)
        self.seed = int(seed)
        self.config = dict(config)
        self.source_sha256 = str(source_sha256)
        self.config_sha256 = str(config_sha256)
        from owl.science.contract import current_scientific_contract

        scientific_contract = current_scientific_contract()
        self.scientific_contract = {
            "version": scientific_contract.version,
            "sha256": scientific_contract.sha256(),
            "action_schema_hash": scientific_contract.action_schema_hash,
            "random_contract_version": scientific_contract.random_contract_version,
        }
        self.max_batch_rows = max(1, int(self.config.get("max_batch_rows", 131_072)))
        self.max_batch_bytes = max(
            1, int(self.config.get("max_batch_bytes", 128 * 1024 * 1024))
        )
        maximum_pending = int(self.config.get("max_pending_bytes", 512 * 1024 * 1024))
        if packet.transfer_bytes > maximum_pending:
            raise MemoryError(
                "CADC host packet exceeds max_pending_bytes: "
                f"{packet.transfer_bytes} > {maximum_pending}"
            )
        self.world_shape = packet.world_shape
        self.schemas = compile_cadc_schemas(packet)
        self.sinks = {
            name: PartitionedParquetSink(
                self.root / f"{name}.parquet",
                schema,
                table_name=f"cadc_{name}",
                schema_digest=self.schema_digest,
                compression=compression,
                row_group_rows=row_group_rows,
                full_validation=str(self.config.get("validation", "full")) == "full",
                resume=resume,
                max_committed_tick=max_committed_tick,
            )
            for name, schema in self.schemas.items()
        }
        self._ticks: list[int] = []
        self._pending_information: dict[str, np.ndarray] | None = None
        self._pending_before_tick: dict[str, np.ndarray] | None = None
        self._closed = False
        _atomic_json(
            Path(root)
            / "schema"
            / ("cadc_factual_v2.json" if action_transitions else "cadc_factual_v1.json"),
            {
                **manifest,
                "world_shape": list(packet.world_shape),
                "channel_count": packet.channel_count,
                "tables": {name: schema.to_string() for name, schema in self.schemas.items()},
                "config": self.config,
                "source_sha256": self.source_sha256,
                "config_sha256": self.config_sha256,
                "scientific_contract": self.scientific_contract,
            },
        )
        if resume:
            self._restore_resume_state(max_committed_tick=max_committed_tick)

    def _restore_resume_state(self, *, max_committed_tick: int | None) -> None:
        """Restore append state from authoritative CADC commit markers."""
        commits = sorted((self.root / "commits").glob("tick_*.json"))
        ticks = [
            int(json.loads(path.read_text(encoding="utf-8"))["tick"])
            for path in commits
        ]
        if ticks != sorted(set(ticks)):
            raise RuntimeError("CADC commit ticks are not strictly increasing and unique")
        if max_committed_tick is not None and any(
            tick > int(max_committed_tick) for tick in ticks
        ):
            raise RuntimeError("CADC commits extend beyond the replay commit boundary")
        self._ticks = ticks
        if not ticks:
            return

        import pyarrow.parquet as pq

        information_root = self.root / "information.parquet"
        parts = sorted(information_root.glob("part-*.parquet"))
        if not parts:
            self._pending_information = {
                "decision_sequence": np.empty(0, dtype=np.int64),
                "ow_id": np.empty(0, dtype=np.int64),
            }
            return
        table = pq.read_table(
            information_root,
            columns=["tick", "decision_sequence", "ow_id"],
        )
        table_ticks = table.column("tick").to_numpy(zero_copy_only=False)
        keep = table_ticks == ticks[-1]
        self._pending_information = {
            "decision_sequence": table.column("decision_sequence")
            .to_numpy(zero_copy_only=False)[keep]
            .astype(np.int64, copy=True),
            "ow_id": table.column("ow_id")
            .to_numpy(zero_copy_only=False)[keep]
            .astype(np.int64, copy=True),
        }

    def _base(self, packet: CADCHostPacket, flat: np.ndarray) -> dict[str, Any]:
        n = int(flat.size)
        h, w = packet.world_shape
        del h
        arrays = packet.arrays
        return {
            "run_id": _string_column(self.run_id, n),
            "condition": _string_column(self.condition, n),
            "seed": np.full(n, self.seed, dtype=np.int64),
            "tick": np.full(n, packet.tick, dtype=np.int64),
            "decision_sequence": arrays["decision_sequence"].reshape(-1)[flat],
            "ow_id": arrays["pre_ow_id"].reshape(-1)[flat],
            "source_y": (flat // w).astype(np.int32, copy=False),
            "source_x": (flat % w).astype(np.int32, copy=False),
        }

    @staticmethod
    def _gather(array: np.ndarray, flat: np.ndarray) -> np.ndarray:
        trailing = int(np.prod(array.shape[2:])) if array.ndim > 2 else 1
        view = array.reshape(array.shape[0] * array.shape[1], trailing)
        selected = view[flat]
        return np.asarray(selected[:, 0] if trailing == 1 else selected)

    def _write(self, name: str, columns: dict[str, Any], *, tick: int) -> int:
        rows = len(next(iter(columns.values())))
        if rows == 0:
            return 0
        estimated_bytes = 0
        for value in columns.values():
            nbytes = int(getattr(value, "nbytes", 0))
            estimated_bytes += nbytes if nbytes else rows * 16
        estimated_width = max(1, (estimated_bytes + rows - 1) // rows)
        row_limit = min(self.max_batch_rows, max(1, self.max_batch_bytes // estimated_width))
        written = 0
        for start in range(0, rows, row_limit):
            stop = min(rows, start + row_limit)
            batch = _record_batch(
                self.schemas[name],
                {key: value[start:stop] for key, value in columns.items()},
            )
            if batch.num_rows != stop - start:
                raise RuntimeError(f"CADC {name} row count changed during Arrow conversion")
            self.sinks[name].write_batch(batch, tick=tick)
            written += int(batch.num_rows)
        if written != rows:
            raise RuntimeError(f"CADC {name} batched row count mismatch: {written} != {rows}")
        return int(rows)

    def _write_followups(
        self, current_ow: np.ndarray, current_decision: np.ndarray, *, tick: int
    ) -> int:
        pending = self._pending_information
        if pending is None or pending["ow_id"].size == 0:
            return 0
        order = np.argsort(current_ow, kind="stable")
        sorted_ow = current_ow[order]
        positions = np.searchsorted(sorted_ow, pending["ow_id"])
        in_range = positions < sorted_ow.size
        safe = np.minimum(positions, max(0, sorted_ow.size - 1))
        matched = in_range & (sorted_ow[safe] == pending["ow_id"]) if sorted_ow.size else in_range
        followup = np.full(pending["ow_id"].size, -1, dtype=np.int64)
        if sorted_ow.size:
            followup[matched] = current_decision[order[safe[matched]]]
        rows = int(pending["ow_id"].size)
        return self._write(
            "information_followups",
            {
                "run_id": _string_column(self.run_id, rows),
                "condition": _string_column(self.condition, rows),
                "seed": np.full(rows, self.seed, dtype=np.int64),
                "tick": np.full(rows, tick, dtype=np.int64),
                "source_decision_sequence": pending["decision_sequence"],
                "source_ow_id": pending["ow_id"],
                "followup_decision_sequence": followup,
                "followup_observation_ref": followup.copy(),
                "followup_status": np.where(matched, 1, 2).astype(np.int8),
            },
            tick=tick,
        )

    def record(self, packet: CADCHostPacket) -> CADCTickReceipt:
        if self._closed:
            raise RuntimeError("CADCFactualRecorder is closed")
        if packet.stage_code != 80:
            raise RuntimeError(f"CADC packet is not at tick commit: stage={packet.stage_code}")
        if packet.world_shape != self.world_shape:
            raise ValueError("CADC world shape changed")
        event_overflow = int(packet.arrays["event_overflow"][0])
        if event_overflow and bool(self.config.get("strict_overflow", True)):
            raise OverflowError(f"CADC event capacity exceeded by {event_overflow} events")
        arrays = packet.arrays
        living = np.flatnonzero(
            ((arrays["pre_alive"] > 0) & (arrays["pre_ow_id"] >= 0)).reshape(-1)
        ).astype(np.int64, copy=False)
        base = self._base(packet, living)
        row_counts: dict[str, int] = {}
        self._pending_before_tick = self._pending_information
        try:
            decisions = dict(base)
            decisions.update(
                {
                    "ow_type": self._gather(arrays["pre_ow_type"], living),
                    "lineage_id": self._gather(arrays["pre_lineage_id"], living),
                    "parent_id": self._gather(arrays["pre_parent_id"], living),
                    "age": self._gather(arrays["pre_age"], living),
                    "development_stage": self._gather(
                        arrays["pre_development_stage"], living
                    ),
                    "selected_action": self._gather(arrays["selected_action"], living),
                    "selected_target_y": self._gather(arrays["selected_target_y"], living),
                    "selected_target_x": self._gather(arrays["selected_target_x"], living),
                    "selected_target_ow_id": self._gather(
                        arrays["selected_target_ow_id"], living
                    ),
                    "selected_probability": self._gather(
                        arrays["selected_probability"], living
                    ),
                    "agent_context_ref": base["decision_sequence"].copy(),
                    "oracle_context_ref": base["decision_sequence"].copy(),
                    "dense_context_ref": (
                        base["decision_sequence"].copy()
                        if "dense_context" in self.schemas
                        else np.full(living.size, -1, dtype=np.int64)
                    ),
                }
            )
            row_counts["decisions"] = self._write("decisions", decisions, tick=packet.tick)

            agent = dict(base)
            oracle = dict(base)
            for name, value in arrays.items():
                if name.startswith("agent_"):
                    agent[name] = self._gather(value, living)
                elif name.startswith("oracle_"):
                    oracle[name] = self._gather(value, living)
            row_counts["agent_context"] = self._write(
                "agent_context", agent, tick=packet.tick
            )
            row_counts["oracle_context"] = self._write(
                "oracle_context", oracle, tick=packet.tick
            )
            if "dense_context" in self.schemas:
                local_cells = int(arrays["dense_oracle_food"].shape[-1])
                radius = (int(round(local_cells**0.5)) - 1) // 2
                dense = dict(base)
                dense.update(
                    {
                        "dense_context_id": base["decision_sequence"].copy(),
                        "chunk_id": np.full(living.size, packet.tick, dtype=np.int64),
                        "chunk_offset": living.copy(),
                        "radius": np.full(living.size, radius, dtype=np.int16),
                    }
                )
                for name in sorted(
                    item for item in arrays if item.startswith("dense_oracle_")
                ):
                    dense[name] = self._gather(arrays[name], living)
                row_counts["dense_context"] = self._write(
                    "dense_context", dense, tick=packet.tick
                )

            action = np.tile(np.arange(CADC_ACTION_COUNT, dtype=np.int16), living.size)
            candidate_base = {
                key: np.repeat(value, CADC_ACTION_COUNT) for key, value in base.items()
            }
            decision_repeat = candidate_base["decision_sequence"]
            candidate_base.update(
                {
                    "candidate_sequence": decision_repeat * CADC_ACTION_COUNT + action,
                    "action_index": action,
                }
            )
            candidate_sources = {
                "target_kind": "candidate_target_kind",
                "proposed_y": "candidate_proposed_y",
                "proposed_x": "candidate_proposed_x",
                "resolved_y": "candidate_resolved_y",
                "resolved_x": "candidate_resolved_x",
                "target_ow_id": "candidate_target_ow_id",
                "destination_occupancy": "candidate_destination_occupancy",
                "destination_obstacle": "candidate_destination_obstacle",
                "destination_food": "candidate_destination_food",
                "destination_toxin": "candidate_destination_toxin",
                "opportunity_count": "candidate_opportunity_count",
                "utility": "candidate_utility",
                "policy_legal": "policy_legal",
                "prechoice_executable": "candidate_executable",
                "prechoice_reason_code": "candidate_reason_code",
            }
            if self.action_transitions:
                candidate_sources.update(
                    {
                        "target_source": "candidate_target_source",
                        "target_distance": "candidate_target_distance",
                        "target_confidence": "candidate_target_confidence",
                        "compiled_action": "candidate_compiled_action",
                    }
                )
            for destination, source in candidate_sources.items():
                candidate_base[destination] = arrays[source].reshape(-1, CADC_ACTION_COUNT)[
                    living
                ].reshape(-1)
            row_counts["candidates"] = self._write(
                "candidates", candidate_base, tick=packet.tick
            )
            if row_counts["candidates"] != row_counts["decisions"] * CADC_ACTION_COUNT:
                raise RuntimeError("CADC candidate cardinality is not exactly 22 per decision")

            execution = dict(base)
            execution_names = [
                "selected_action",
                "attempted_action",
                "realized_action",
                "execution_success",
                "execution_reason_code",
                "realized_target_y",
                "realized_target_x",
                "realized_target_ow_id",
                "amount_consumed",
                "amount_transferred",
                "amount_repaired",
                "amount_damaged",
                "amount_emitted",
                "amount_received",
                "direct_cost",
            ]
            if self.action_transitions:
                execution_names.extend(
                    [
                        "compiled_execution_action",
                        "intent_target_y",
                        "intent_target_x",
                        "intent_target_ow_id",
                        "intent_target_kind",
                        "intent_target_source",
                        "intent_target_distance_before",
                        "intent_target_distance_after",
                        "intent_known_hazard_before",
                        "intent_known_hazard_after",
                        "intent_contact_opportunity_before",
                        "intent_contact_opportunity_after",
                    ]
                )
            for name in execution_names:
                execution[name] = self._gather(arrays[name], living)
            row_counts["execution"] = self._write("execution", execution, tick=packet.tick)

            if self.action_transitions:
                direction_count = 16
                direction_base = {
                    key: np.repeat(value, direction_count) for key, value in base.items()
                }
                local_index = np.tile(
                    np.arange(direction_count, dtype=np.int64), living.size
                )
                direction_base.update(
                    {
                        "direction_sequence": (
                            direction_base["decision_sequence"] * direction_count
                            + local_index
                        ),
                        "action_family": (local_index // 8).astype(np.int8),
                        "direction_index": (local_index % 8).astype(np.int8),
                    }
                )
                for destination, source in (
                    ("direction_y", "action_direction_y"),
                    ("direction_x", "action_direction_x"),
                    ("direction_executable", "action_direction_executable"),
                    ("direction_score", "action_direction_score"),
                    ("distance_delta", "action_direction_distance_delta"),
                    ("known_hazard", "action_direction_hazard"),
                    ("opportunity", "action_direction_opportunity"),
                ):
                    direction_base[destination] = self._gather(
                        arrays[source], living
                    ).reshape(-1)
                for destination, source in (
                    ("target_y", "action_target_y"),
                    ("target_x", "action_target_x"),
                    ("target_ow_id", "action_target_ow_id"),
                    ("target_kind", "action_target_kind"),
                    ("target_source", "action_target_source"),
                    ("target_distance", "action_target_distance"),
                    ("target_confidence", "action_target_confidence"),
                ):
                    family_values = self._gather(arrays[source], living).reshape(-1, 2)
                    direction_base[destination] = np.repeat(
                        family_values, 8, axis=1
                    ).reshape(-1)
                row_counts["action_directions"] = self._write(
                    "action_directions", direction_base, tick=packet.tick
                )
                if row_counts["action_directions"] != row_counts["decisions"] * 16:
                    raise RuntimeError(
                        "CADC action-direction cardinality is not exactly 16 per decision"
                    )

            contribution_count = len(packet.contribution_codes)
            contribution = {
                key: np.repeat(value, contribution_count) for key, value in base.items()
            }
            contribution["contribution_sequence"] = (
                contribution["decision_sequence"] * contribution_count
                + np.tile(np.arange(contribution_count, dtype=np.int64), living.size)
            )
            contribution["contribution_code"] = np.tile(
                np.asarray(packet.contribution_codes, dtype=np.int16), living.size
            )
            deltas = arrays["contribution_delta"].reshape(
                contribution_count, -1, len(packet.contribution_fields)
            )
            selected_deltas = deltas[:, living, :].transpose(1, 0, 2).reshape(
                -1, len(packet.contribution_fields)
            )
            starts = arrays["tick_start"].reshape(-1, len(packet.contribution_fields))[
                living
            ]
            ends = arrays["tick_end"].reshape(-1, len(packet.contribution_fields))[living]
            for index, name in enumerate(packet.contribution_fields):
                contribution[f"delta_{name}"] = selected_deltas[:, index]
                contribution[f"start_{name}"] = np.repeat(starts[:, index], contribution_count)
                contribution[f"end_{name}"] = np.repeat(ends[:, index], contribution_count)
            row_counts["contributions"] = self._write(
                "contributions", contribution, tick=packet.tick
            )

            event_active = arrays["event_active"].reshape(-1)
            event_flat = np.flatnonzero(event_active).astype(np.int64, copy=False)
            world_cells = int(np.prod(packet.world_shape))
            code_slot = event_flat // world_cells
            event_rows = int(event_flat.size)
            source_y = arrays["event_source_y"].reshape(-1)[event_flat]
            source_x = arrays["event_source_x"].reshape(-1)[event_flat]
            safe_source = np.clip(source_y, 0, packet.world_shape[0] - 1) * packet.world_shape[
                1
            ] + np.clip(source_x, 0, packet.world_shape[1] - 1)
            events = {
                "run_id": _string_column(self.run_id, event_rows),
                "condition": _string_column(self.condition, event_rows),
                "seed": np.full(event_rows, self.seed, dtype=np.int64),
                "tick": np.full(event_rows, packet.tick, dtype=np.int64),
                "event_sequence": packet.tick * len(packet.event_codes) * world_cells + event_flat,
                "decision_sequence": arrays["decision_sequence"].reshape(-1)[safe_source],
                "event_code": np.asarray(packet.event_codes, dtype=np.int16)[code_slot],
                "stage_code": arrays["event_stage_code"].reshape(-1)[event_flat],
                "reason_code": arrays["event_reason_code"].reshape(-1)[event_flat],
                "actor_ow_id": arrays["pre_ow_id"].reshape(-1)[safe_source],
                "source_y": source_y,
                "source_x": source_x,
                "target_ow_id": arrays["event_target_ow_id"].reshape(-1)[event_flat],
                "target_y": arrays["event_target_y"].reshape(-1)[event_flat],
                "target_x": arrays["event_target_x"].reshape(-1)[event_flat],
            }
            payload = arrays["event_payload"].reshape(-1, 4)[event_flat]
            for index in range(4):
                events[f"payload{index}"] = payload[:, index]
            row_counts["events"] = self._write("events", events, tick=packet.tick)

            current_ow = base["ow_id"]
            current_decision = base["decision_sequence"]
            row_counts["information_followups"] = self._write_followups(
                current_ow, current_decision, tick=packet.tick
            )
            information_flat = np.flatnonzero(arrays["information_active"].reshape(-1)).astype(
                np.int64, copy=False
            )
            information = self._base(packet, information_flat)
            for destination, source in (
                ("information_kind", "information_kind"),
                ("pre_observation_ref", "information_pre_observation_ref"),
                ("post_memory_ref", "information_post_memory_ref"),
                ("pre_signal_sum", "information_pre_signal_sum"),
                ("post_signal_memory_sum", "information_post_signal_memory_sum"),
                ("memory_delta", "information_memory_delta"),
                ("amount_emitted", "amount_emitted"),
                ("amount_received", "information_amount_received"),
                ("followup_tick", "information_followup_tick"),
                ("timing_code", "information_timing_code"),
                ("receiver_count", "information_receiver_count"),
                ("receiver_link_status", "information_receiver_link_status"),
                ("observation_before", "information_observation_before"),
                ("memory_before", "information_memory_before"),
                ("memory_after", "information_memory_after"),
                ("emitted_channels", "information_emitted_channels"),
                ("received_channels", "information_received_channels"),
            ):
                information[destination] = self._gather(arrays[source], information_flat)
            if self.action_transitions:
                for destination, source in (
                    ("new_cell_count", "information_new_cell_count"),
                    ("new_target_count", "information_new_target_count"),
                    ("active_memory_changed", "information_memory_changed"),
                    ("information_execution_success", "information_execution_success"),
                    ("no_new_information", "information_no_new_information"),
                    ("sensed_food_before", "information_sensed_food_before"),
                    ("sensed_food_after", "information_sensed_food_after"),
                    ("sensed_toxin_before", "information_sensed_toxin_before"),
                    ("sensed_toxin_after", "information_sensed_toxin_after"),
                    ("sensed_alive_before", "information_sensed_alive_before"),
                    ("sensed_alive_after", "information_sensed_alive_after"),
                ):
                    information[destination] = self._gather(
                        arrays[source], information_flat
                    )
            row_counts["information"] = self._write(
                "information", information, tick=packet.tick
            )
            self._pending_information = {
                "decision_sequence": information["decision_sequence"].copy(),
                "ow_id": information["ow_id"].copy(),
            }
            _atomic_json(
                self.root / "commits" / f"tick_{packet.tick:08d}.json",
                {
                    "schema_version": "owl.cadc.tick-commit.v1",
                    "tick": packet.tick,
                    "row_counts": row_counts,
                    "transfer_bytes": packet.transfer_bytes,
                    "transfer_count": packet.transfer_count,
                    "source_backend": packet.source_backend,
                    "event_overflow": event_overflow,
                    "tick_open_state_hash": _array_digest(
                        arrays, ["pre_ow_id", "tick_start"]
                    ),
                    "agent_context_hash": _array_digest(
                        arrays,
                        [name for name in arrays if name.startswith("agent_")],
                    ),
                    "oracle_context_hash": _array_digest(
                        arrays,
                        [
                            name
                            for name in arrays
                            if name.startswith("oracle_")
                            or name.startswith("dense_oracle_")
                        ],
                    ),
                    "candidate_context_hash": _array_digest(
                        arrays,
                        [
                            name
                            for name in arrays
                            if name.startswith("candidate_") or name == "policy_legal"
                        ],
                    ),
                },
            )
            self._ticks.append(packet.tick)
        except Exception:
            self.rollback_tick(packet.tick)
            raise
        return CADCTickReceipt(
            packet.tick,
            row_counts,
            packet.transfer_bytes,
            packet.transfer_count,
            packet.source_backend,
            event_overflow,
        )

    def rollback_tick(self, tick: int) -> None:
        for sink in self.sinks.values():
            sink.rollback_tick(tick)
        (self.root / "commits" / f"tick_{int(tick):08d}.json").unlink(missing_ok=True)
        self._ticks = [item for item in self._ticks if item != int(tick)]
        self._pending_information = self._pending_before_tick
        self._pending_before_tick = None

    def close(self, *, finalize_pending: bool = True) -> None:
        if self._closed:
            return
        if finalize_pending and self._pending_information is not None:
            pending = self._pending_information
            rows = int(pending["ow_id"].size)
            if rows:
                tick = self._ticks[-1] if self._ticks else 0
                self._write(
                    "information_followups",
                    {
                        "run_id": _string_column(self.run_id, rows),
                        "condition": _string_column(self.condition, rows),
                        "seed": np.full(rows, self.seed, dtype=np.int64),
                        "tick": np.full(rows, tick, dtype=np.int64),
                        "source_decision_sequence": pending["decision_sequence"],
                        "source_ow_id": pending["ow_id"],
                        "followup_decision_sequence": np.full(rows, -1, dtype=np.int64),
                        "followup_observation_ref": np.full(rows, -1, dtype=np.int64),
                        "followup_status": np.full(rows, 3, dtype=np.int8),
                    },
                    tick=tick,
                )
        for sink in self.sinks.values():
            sink.close()
        _atomic_json(
            self.root / "manifest.json",
            {
                "schema_version": "owl.cadc.factual-manifest.v1",
                "factual_schema_version": self.schema_version,
                "schema_digest": self.schema_digest,
                "run_id": self.run_id,
                "condition": self.condition,
                "seed": self.seed,
                "ticks": self._ticks,
                "row_counts": {name: sink.rows_written for name, sink in self.sinks.items()},
                "closed_at": datetime.now(UTC).isoformat(),
                "pending_finalized": bool(finalize_pending),
                "source_sha256": self.source_sha256,
                "config_sha256": self.config_sha256,
                "scientific_contract": self.scientific_contract,
            },
        )
        self._closed = True
