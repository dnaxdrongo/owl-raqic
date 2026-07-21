from __future__ import annotations

import csv
import json
from collections.abc import Collection, Sequence
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np

from owl.replay.cache import LRUCache
from owl.replay.data_source import OWReplayDetails
from owl.replay.manifest import ReplayManifest, sha256_file
from owl.viz.event_bus import VisualEvent, VisualEventType
from owl.viz.visual_snapshot import VisualSnapshot, snapshot_from_arrays


class ZarrReplayDataSource:
    """Read a completed ``owl.replay.v1`` bundle without mutating it."""

    def __init__(
        self,
        bundle_root: str | Path,
        *,
        cache_entries: int = 8,
        verify_metadata: bool = True,
    ) -> None:
        import zarr

        self.root = Path(bundle_root)
        self._manifest = ReplayManifest.load(self.root)
        self._group = zarr.open_group(str(self.root / "replay" / "replay.zarr"), mode="r")
        self._ticks = np.asarray(self._group["tick"][:], dtype=np.int64)
        if self._ticks.size != self._manifest.completed_ticks:
            raise ValueError("replay tick index and manifest completed_ticks disagree")
        if self._ticks.size and np.any(np.diff(self._ticks) <= 0):
            raise ValueError("replay tick index is not strictly increasing")
        self._tick_to_index = {int(tick): index for index, tick in enumerate(self._ticks)}
        self._cache: LRUCache[VisualSnapshot] = LRUCache(cache_entries)
        self._read_lock = RLock()
        self._events_path = self.root / "replay" / "events.parquet"
        self._event_index: tuple[tuple[int, int], ...] | None = None
        self._ow_table: Any | None = None
        self._decision_table: Any | None = None
        self._action_math_table: Any | None = None
        self._cadc_root = self.root / "analysis" / "cadc_v1"
        self._cadc_manifest: dict[str, Any] | None = None
        self._datasets: dict[str, Any] = {}
        self.verification_status = "not_requested"
        if verify_metadata:
            self.verify(metadata_only=True)

    @property
    def manifest(self) -> ReplayManifest:
        return self._manifest

    def tick_count(self) -> int:
        return int(self._ticks.size)

    def available_ticks(self) -> Sequence[int]:
        return tuple(int(item) for item in self._ticks.tolist())

    @property
    def cadc_manifest(self) -> dict[str, Any] | None:
        """Return additive CADC metadata when present, without requiring it."""
        if self._cadc_manifest is None:
            path = self._cadc_root / "manifest.json"
            if not path.exists():
                return None
            self._cadc_manifest = json.loads(path.read_text(encoding="utf-8"))
        return dict(self._cadc_manifest)

    def available_cadc_tables(self) -> Sequence[str]:
        """Discover factual evidence tables while preserving old replay support."""
        if not self._cadc_root.exists():
            return ()
        return tuple(
            sorted(path.name.removesuffix(".parquet") for path in self._cadc_root.glob("*.parquet"))
        )

    def load_cadc_table(
        self,
        name: str,
        *,
        tick: int | None = None,
        ow_id: int | None = None,
        columns: Sequence[str] | None = None,
    ) -> Any | None:
        """Read one discovered CADC table with Arrow filter pushdown."""
        available = self.available_cadc_tables()
        if name not in available:
            return None
        return self._scan_table(
            self._cadc_root / f"{name}.parquet",
            columns=columns,
            tick=tick,
            ow_id=ow_id,
        )

    def _dataset(self, path: Path) -> Any | None:
        """Return a cached Arrow Dataset for a Parquet file or partition directory."""

        if not path.exists():
            return None
        key = str(path.resolve())
        with self._read_lock:
            dataset = self._datasets.get(key)
            if dataset is None:
                import pyarrow.dataset as ds

                dataset = ds.dataset(path, format="parquet")
                self._datasets[key] = dataset
            return dataset

    @staticmethod
    def _filter_expression(
        *,
        start_tick: int | None = None,
        end_tick: int | None = None,
        tick: int | None = None,
        ow_id: int | None = None,
        source_id: int | None = None,
        parent_id: int | None = None,
        lineage_id: int | None = None,
    ) -> Any | None:
        import pyarrow.dataset as ds

        expression: Any | None = None

        def add(clause: Any) -> None:
            nonlocal expression
            expression = clause if expression is None else expression & clause

        if start_tick is not None:
            add(ds.field("tick") >= int(start_tick))
        if end_tick is not None:
            add(ds.field("tick") <= int(end_tick))
        if tick is not None:
            add(ds.field("tick") == int(tick))
        if ow_id is not None:
            add(ds.field("ow_id") == int(ow_id))
        if source_id is not None:
            add(ds.field("source_id") == int(source_id))
        if parent_id is not None:
            add(ds.field("parent_id") == int(parent_id))
        if lineage_id is not None:
            add(ds.field("lineage_id") == int(lineage_id))
        return expression

    def _scanner(
        self,
        path: Path,
        *,
        columns: Sequence[str] | None = None,
        start_tick: int | None = None,
        end_tick: int | None = None,
        tick: int | None = None,
        ow_id: int | None = None,
        source_id: int | None = None,
        parent_id: int | None = None,
        lineage_id: int | None = None,
        batch_size: int = 65_536,
    ) -> Any | None:
        """Build a projection/filter-pushdown scanner against a cached dataset."""

        dataset = self._dataset(path)
        if dataset is None:
            return None
        expression = self._filter_expression(
            start_tick=start_tick,
            end_tick=end_tick,
            tick=tick,
            ow_id=ow_id,
            source_id=source_id,
            parent_id=parent_id,
            lineage_id=lineage_id,
        )
        return dataset.scanner(
            columns=None if columns is None else list(columns),
            filter=expression,
            batch_size=max(1, int(batch_size)),
            use_threads=True,
        )

    def _scan_table(
        self,
        path: Path,
        *,
        columns: Sequence[str] | None = None,
        start_tick: int | None = None,
        end_tick: int | None = None,
        tick: int | None = None,
        ow_id: int | None = None,
        source_id: int | None = None,
        parent_id: int | None = None,
        lineage_id: int | None = None,
    ) -> Any | None:
        scanner = self._scanner(
            path,
            columns=columns,
            start_tick=start_tick,
            end_tick=end_tick,
            tick=tick,
            ow_id=ow_id,
            source_id=source_id,
            parent_id=parent_id,
            lineage_id=lineage_id,
        )
        return None if scanner is None else scanner.to_table()

    @staticmethod
    def _csv_compatible_batch(batch: Any) -> Any:
        """Cast dictionary columns to strings for Arrow CSV without row objects."""

        import pyarrow as pa

        arrays: list[Any] = []
        fields: list[Any] = []
        for field, column in zip(batch.schema, batch.columns, strict=True):
            if pa.types.is_dictionary(field.type):
                arrays.append(column.cast(pa.string()))
                fields.append(pa.field(field.name, pa.string(), nullable=field.nullable))
            else:
                arrays.append(column)
                fields.append(field)
        return pa.RecordBatch.from_arrays(arrays, schema=pa.schema(fields))

    def _write_scanner_csv(self, scanner: Any, destination: Path) -> int:
        """Stream a scanner to CSV in bounded batches; return written row count."""

        import pyarrow.csv as pacsv

        batches = iter(scanner.to_batches())
        try:
            first = self._csv_compatible_batch(next(batches))
        except StopIteration:
            import pyarrow as pa

            schema = scanner.projected_schema
            fields = [
                pa.field(field.name, pa.string(), nullable=field.nullable)
                if pa.types.is_dictionary(field.type)
                else field
                for field in schema
            ]
            with pacsv.CSVWriter(destination, pa.schema(fields)):
                return 0

        row_count = int(first.num_rows)
        with pacsv.CSVWriter(destination, first.schema) as writer:
            writer.write_batch(first)
            for batch in batches:
                compatible = self._csv_compatible_batch(batch)
                writer.write_batch(compatible)
                row_count += int(compatible.num_rows)
        return row_count

    def _read_event_rows(
        self,
        *,
        start_tick: int | None = None,
        end_tick: int | None = None,
        ow_id: int | None = None,
    ) -> list[dict[str, Any]]:
        table = self._scan_table(
            self._events_path,
            start_tick=start_tick,
            end_tick=end_tick,
            source_id=ow_id,
        )
        return [] if table is None else table.to_pylist()

    def _load_event_index(self) -> tuple[tuple[int, int], ...]:
        if self._event_index is not None:
            return self._event_index
        table = self._scan_table(self._events_path, columns=("tick", "source_id"))
        if table is None:
            self._event_index = ()
            return self._event_index
        tick_values = table.column("tick").to_numpy(zero_copy_only=False)
        source_values = table.column("source_id").to_numpy(zero_copy_only=False)
        self._event_index = tuple(
            (int(tick), int(source_id))
            for tick, source_id in zip(tick_values, source_values, strict=True)
        )
        return self._event_index

    @staticmethod
    def _event_from_row(row: dict[str, Any]) -> VisualEvent:
        raw_type = row.get("event_type_code", row.get("event_type", 1))
        if isinstance(raw_type, str) and not raw_type.isdigit():
            event_type = VisualEventType[raw_type]
        else:
            event_type = VisualEventType(int(raw_type))
        return VisualEvent(
            tick=int(row.get("tick", 0)),
            event_type=event_type,
            y=int(row.get("y", -1)),
            x=int(row.get("x", -1)),
            target_y=int(row.get("target_y", -1)),
            target_x=int(row.get("target_x", -1)),
            action=int(row.get("action", 0)),
            intensity=float(row.get("intensity", 1.0)),
            ttl=int(row.get("ttl", 3)),
            source_id=int(row.get("source_id", -1)),
            channel=int(row.get("channel", -1)),
            payload0=float(row.get("payload0", 0.0)),
            payload1=float(row.get("payload1", 0.0)),
            priority=int(row.get("priority", 0)),
        )

    def load_events(self, start_tick: int, end_tick: int) -> tuple[VisualEvent, ...]:
        return tuple(
            self._event_from_row(row)
            for row in self._read_event_rows(start_tick=start_tick, end_tick=end_tick)
        )

    def load_snapshot(
        self,
        tick: int,
        fields: Collection[str] | None = None,
    ) -> VisualSnapshot:
        requested_tick = int(tick)
        cached = self._cache.get(requested_tick) if fields is None else None
        if cached is not None:
            return cached
        try:
            index = self._tick_to_index[requested_tick]
        except KeyError as exc:
            raise KeyError(f"tick {requested_tick} is unavailable") from exc
        available = tuple(str(item) for item in self._manifest.array_fields)
        wanted = set(available if fields is None else fields)
        # Scene construction requires health and occupancy for stable IDs.
        wanted.update({"health", "occupancy"})
        arrays: dict[str, np.ndarray] = {}
        with self._read_lock:
            for name in available:
                if name in wanted:
                    arrays[name] = np.asarray(self._group[f"state/{name}"][index])
        events = self.load_events(requested_tick, requested_tick)
        snapshot = snapshot_from_arrays(
            tick=requested_tick,
            boundary_mode=self._manifest.boundary_mode,
            arrays=arrays,
            events=events,
            metadata={
                "source": "replay_bundle",
                "bundle": str(self.root),
                "run_id": self._manifest.run_id,
                "recording_tier": self._manifest.recording_tier,
            },
        )
        if fields is None:
            self._cache.put(requested_tick, snapshot)
        return snapshot

    @staticmethod
    def _load_parquet(path: Path) -> Any:
        import pyarrow.parquet as pq

        return pq.read_table(path) if path.exists() else None

    def _rows_for_ow(self, tick: int, ow_id: int) -> list[dict[str, Any]]:
        table = self._scan_table(
            self.root / "analysis" / "ow_state.parquet",
            tick=tick,
            ow_id=ow_id,
        )
        return [] if table is None else table.to_pylist()

    @staticmethod
    def _decode_json_columns(row: dict[str, Any]) -> dict[str, Any]:
        decoded: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, str) and value[:1] in {"[", "{"}:
                try:
                    decoded[key] = json.loads(value)
                    continue
                except json.JSONDecodeError:
                    pass
            decoded[key] = value
        return decoded

    def _require_action_materialization_or_legacy(self) -> None:
        if (
            self._manifest.materialization_mode == "deferred"
            and self._manifest.materialization_state != "complete"
        ):
            raise RuntimeError(
                "action-math materialization is incomplete; run owl-experiment materialize "
                "before reading or exporting canonical action rows"
            )

    def load_action_math(self, tick: int, ow_id: int) -> tuple[dict[str, Any], ...]:
        action_path = self.root / "analysis" / "ow_action_math.parquet"
        if action_path.exists():
            table = self._scan_table(action_path, tick=tick, ow_id=ow_id)
            if table is None:
                return ()
            rows = [self._decode_json_columns(row) for row in table.to_pylist()]
            return tuple(sorted(rows, key=lambda row: int(row.get("action_index", -1))))

        self._require_action_materialization_or_legacy()

        # Some replay inputs store all action vectors on one decision row;
        # expand that bounded representation when reading the alternate layout.
        decision_table = self._scan_table(
            self.root / "analysis" / "ow_decisions.parquet",
            tick=tick,
            ow_id=ow_id,
        )
        if decision_table is None:
            return ()
        rows = [self._decode_json_columns(row) for row in decision_table.to_pylist()]
        if not rows:
            return ()
        row = rows[0]
        action_count = len(self._manifest.action_names)
        output: list[dict[str, Any]] = []
        for action_index in range(action_count):
            action: dict[str, Any] = {
                "action_index": action_index,
                "action_name": self._manifest.action_names[action_index],
                "selected": int(row.get("raqic_readout", row.get("readout", -1))) == action_index,
            }
            for field in (
                "last_utilities",
                "pre_utilities",
                "last_logits",
                "last_action_probabilities",
                "raqic_probabilities",
                "possibility",
                "raqic_score",
                "raqic_phase",
                "raqic_parent_intention",
                "raqic_pre_mixer_probabilities",
                "raqic_utility_innovation",
                "raqic_phase_alignment",
                "raqic_resonant_parent_intention",
                "raqic_shadow_probabilities",
                "authority",
                "_authority_bool",
            ):
                values = row.get(field)
                if isinstance(values, list) and action_index < len(values):
                    action[field] = values[action_index]
            output.append(action)
        return tuple(output)

    def load_ow_details(self, tick: int, ow_id: int) -> OWReplayDetails | None:
        rows = self._rows_for_ow(tick, ow_id)
        if not rows:
            return None
        values = self._decode_json_columns(rows[0])
        decision_path = self.root / "analysis" / "ow_decisions.parquet"
        if decision_path.exists():
            decision_table = self._scan_table(decision_path, tick=tick, ow_id=ow_id)
            decision_rows = [] if decision_table is None else decision_table.to_pylist()
            if decision_rows:
                values.update(self._decode_json_columns(decision_rows[0]))
        position = (int(values.get("y", -1)), int(values.get("x", -1)))
        events = tuple(
            self._read_event_rows(
                start_tick=int(tick) - 5,
                end_tick=int(tick),
                ow_id=int(ow_id),
            )
        )
        return OWReplayDetails(
            tick=int(tick),
            ow_id=int(ow_id),
            position=position,
            values=values,
            action_math=self.load_action_math(tick, ow_id),
            recent_events=events,
        )

    def event_ticks(self, *, ow_id: int | None = None) -> tuple[int, ...]:
        ticks = {
            tick
            for tick, source_id in self._load_event_index()
            if tick >= 0 and (ow_id is None or source_id == int(ow_id))
        }
        return tuple(sorted(ticks))

    def nearest_event_tick(
        self,
        current_tick: int,
        *,
        direction: int,
        ow_id: int | None = None,
    ) -> int | None:
        ticks = self.event_ticks(ow_id=ow_id)
        if int(direction) < 0:
            matches = [tick for tick in ticks if tick < int(current_tick)]
            return max(matches) if matches else None
        matches = [tick for tick in ticks if tick > int(current_tick)]
        return min(matches) if matches else None

    def find_children(self, tick: int, parent_id: int) -> tuple[int, ...]:
        path = self.root / "analysis" / "ow_state.parquet"
        if not path.exists():
            return ()
        table = self._scan_table(path, columns=("ow_id",), tick=tick, parent_id=parent_id)
        if table is None:
            return ()
        return tuple(sorted({int(value) for value in table.column("ow_id").to_pylist()}))

    def find_lineage_members(self, tick: int, lineage_id: int) -> tuple[int, ...]:
        path = self.root / "analysis" / "ow_state.parquet"
        if not path.exists():
            return ()
        table = self._scan_table(path, columns=("ow_id",), tick=tick, lineage_id=lineage_id)
        if table is None:
            return ()
        return tuple(sorted({int(value) for value in table.column("ow_id").to_pylist()}))

    def load_ow_history(
        self,
        ow_id: int,
        *,
        start_tick: int,
        end_tick: int,
    ) -> tuple[dict[str, Any], ...]:
        path = self.root / "analysis" / "ow_state.parquet"
        if not path.exists():
            return ()
        table = self._scan_table(path, ow_id=ow_id, start_tick=start_tick, end_tick=end_tick)
        rows = (
            [] if table is None else [self._decode_json_columns(row) for row in table.to_pylist()]
        )
        return tuple(sorted(rows, key=lambda row: int(row.get("tick", -1))))

    def export_action_math_csv(
        self,
        destination: str,
        *,
        ow_id: int,
        start_tick: int,
        end_tick: int,
    ) -> str:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        action_path = self.root / "analysis" / "ow_action_math.parquet"
        if not action_path.exists():
            self._require_action_materialization_or_legacy()
        scanner = self._scanner(
            action_path,
            ow_id=ow_id,
            start_tick=start_tick,
            end_tick=end_tick,
        )
        if scanner is not None:
            self._write_scanner_csv(scanner, path)
            return str(path)

        # Convert the single-row vector layout only when the canonical schema
        # materialization is unavailable.
        rows: list[dict[str, Any]] = []
        for tick in self.available_ticks():
            if not int(start_tick) <= int(tick) <= int(end_tick):
                continue
            for action in self.load_action_math(int(tick), int(ow_id)):
                rows.append({"tick": int(tick), "ow_id": int(ow_id), **action})
        fields = sorted({key for row in rows for key in row})
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return str(path)

    def export_selection_csv(
        self,
        destination: str,
        *,
        ow_id: int,
        start_tick: int,
        end_tick: int,
    ) -> str:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        source_path = self.root / "analysis" / "ow_state.parquet"
        scanner = self._scanner(
            source_path,
            ow_id=ow_id,
            start_tick=start_tick,
            end_tick=end_tick,
        )
        if scanner is not None:
            self._write_scanner_csv(scanner, path)
            return str(path)

        path.write_text("", encoding="utf-8")
        return str(path)

    def verify(self, *, metadata_only: bool = False) -> dict[str, Any]:
        checksum_path = self.root / "checksums" / "SHA256SUMS.txt"
        failures: list[str] = []
        checked = 0
        if checksum_path.exists():
            for line in checksum_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                expected, relative = line.split("  ", 1)
                if metadata_only and not relative.endswith((".json", ".yaml", ".txt")):
                    continue
                path = self.root / relative
                checked += 1
                if not path.exists() or sha256_file(path) != expected:
                    failures.append(relative)
        self.verification_status = "passed" if not failures else "failed"
        if failures:
            raise ValueError(f"replay checksum failures: {failures}")
        return {"passed": True, "checked": checked, "metadata_only": metadata_only}
