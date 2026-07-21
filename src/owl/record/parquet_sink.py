"""Append-safe, fixed-schema partitioned Parquet sink."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from owl.replay.manifest import sha256_file


@dataclass(frozen=True)
class ParquetPartReceipt:
    table_name: str
    part_index: int
    path: str
    rows: int
    uncompressed_bytes: int
    file_bytes: int
    sha256: str
    schema_digest: str
    tick_min: int
    tick_max: int


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


class PartitionedParquetSink:
    """Write independent atomic Parquet parts for one canonical table."""

    def __init__(
        self,
        root: str | Path,
        schema: Any,
        *,
        table_name: str,
        schema_digest: str,
        compression: str = "zstd",
        row_group_rows: int = 131_072,
        full_validation: bool = False,
        resume: bool = False,
        max_committed_tick: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.journal_root = self.root / "_journal"
        self.journal_root.mkdir(parents=True, exist_ok=True)
        self.schema = schema
        self.table_name = str(table_name)
        self.schema_digest = str(schema_digest)
        self.compression = str(compression)
        self.row_group_rows = max(1, int(row_group_rows))
        self.full_validation = bool(full_validation)
        self._closed = False
        self._rows_written = 0
        self._parts_written = 0
        self._remove_temporary_files()
        if resume:
            self._recover(max_committed_tick=max_committed_tick)
        self._next_part = self._discover_next_part()

    def _remove_temporary_files(self) -> None:
        for path in self.root.glob("*.tmp.*"):
            path.unlink(missing_ok=True)
        for path in self.root.glob(".*.tmp.*"):
            path.unlink(missing_ok=True)
        for path in self.journal_root.glob(".*.tmp.*"):
            path.unlink(missing_ok=True)

    def _part_paths(self) -> list[Path]:
        return sorted(self.root.glob("part-*.parquet"))

    def _discover_next_part(self) -> int:
        parts = self._part_paths()
        if not parts:
            return 0
        return max(int(path.stem.split("-")[-1]) for path in parts) + 1

    def _receipt_path(self, part_index: int) -> Path:
        return self.journal_root / f"part-{part_index:06d}.json"

    def _validate_part_schema(self, path: Path) -> Any:
        import pyarrow.parquet as pq

        actual = pq.read_schema(path)
        if not actual.equals(self.schema, check_metadata=True):
            raise RuntimeError(f"Parquet schema mismatch while resuming {path}")
        return actual

    def _part_tick_range(self, path: Path) -> tuple[int, int]:
        import pyarrow.parquet as pq

        table = pq.read_table(path, columns=["tick"])
        if table.num_rows == 0:
            return (-1, -1)
        ticks = table.column("tick").to_numpy(zero_copy_only=False)
        return int(ticks.min()), int(ticks.max())

    def _recover(self, *, max_committed_tick: int | None) -> None:
        for part in self._part_paths():
            part_index = int(part.stem.split("-")[-1])
            receipt_path = self._receipt_path(part_index)
            receipt: dict[str, Any] | None = None
            if receipt_path.exists():
                try:
                    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    receipt = None
            self._validate_part_schema(part)
            tick_min, tick_max = self._part_tick_range(part)
            if max_committed_tick is not None and tick_max > max_committed_tick:
                part.unlink(missing_ok=True)
                receipt_path.unlink(missing_ok=True)
                continue
            if receipt is None or receipt.get("sha256") != sha256_file(part):
                rebuilt = ParquetPartReceipt(
                    table_name=self.table_name,
                    part_index=part_index,
                    path=part.name,
                    rows=self._parquet_rows(part),
                    uncompressed_bytes=0,
                    file_bytes=part.stat().st_size,
                    sha256=sha256_file(part),
                    schema_digest=self.schema_digest,
                    tick_min=tick_min,
                    tick_max=tick_max,
                )
                _atomic_json(receipt_path, asdict(rebuilt))
            self._rows_written += self._parquet_rows(part)
            self._parts_written += 1

    @staticmethod
    def _parquet_rows(path: Path) -> int:
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(path).metadata.num_rows)

    @property
    def rows_written(self) -> int:
        return self._rows_written

    @property
    def parts_written(self) -> int:
        return self._parts_written

    def write_batch(self, batch: Any, *, tick: int) -> ParquetPartReceipt:
        if self._closed:
            raise RuntimeError(f"{self.table_name} sink is closed")
        import pyarrow as pa
        import pyarrow.parquet as pq

        if not isinstance(batch, pa.RecordBatch):
            raise TypeError(f"{self.table_name} sink accepts only pyarrow.RecordBatch")
        if not batch.schema.equals(self.schema, check_metadata=True):
            raise ValueError(f"{self.table_name} batch schema differs from compiled schema")
        batch.validate(full=self.full_validation)
        if batch.num_rows <= 0:
            raise ValueError(f"{self.table_name} cannot write an empty batch")

        part_index = self._next_part
        final_path = self.root / f"part-{part_index:06d}.parquet"
        if final_path.exists():
            raise FileExistsError(f"refusing to overwrite existing Parquet part: {final_path}")
        temporary = self.root / f".{final_path.name}.tmp.{os.getpid()}"
        table = pa.Table.from_batches([batch], schema=self.schema)
        pq.write_table(
            table,
            temporary,
            compression=None if self.compression == "none" else self.compression,
            row_group_size=min(self.row_group_rows, max(1, batch.num_rows)),
            use_dictionary=True,
            write_statistics=True,
        )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, final_path)
        _fsync_directory(self.root)
        receipt = ParquetPartReceipt(
            table_name=self.table_name,
            part_index=part_index,
            path=final_path.name,
            rows=int(batch.num_rows),
            uncompressed_bytes=int(batch.get_total_buffer_size()),
            file_bytes=int(final_path.stat().st_size),
            sha256=sha256_file(final_path),
            schema_digest=self.schema_digest,
            tick_min=int(tick),
            tick_max=int(tick),
        )
        _atomic_json(self._receipt_path(part_index), asdict(receipt))
        self._next_part += 1
        self._parts_written += 1
        self._rows_written += int(batch.num_rows)
        return receipt

    def rollback_tick(self, tick: int) -> None:
        """Remove parts for an uncommitted tick and restore append counters.

         writes each RecordBatch for one tick, so a part never mixes
        committed and uncommitted ticks. This method is used only before the
        authoritative tick commit marker is created.
        """

        if self._closed:
            raise RuntimeError(f"{self.table_name} sink is closed")
        target = int(tick)
        for part in reversed(self._part_paths()):
            part_index = int(part.stem.split("-")[-1])
            receipt_path = self._receipt_path(part_index)
            tick_min, tick_max = self._part_tick_range(part)
            if tick_min <= target <= tick_max:
                part.unlink(missing_ok=True)
                receipt_path.unlink(missing_ok=True)
        self._rows_written = 0
        self._parts_written = 0
        for part in self._part_paths():
            self._validate_part_schema(part)
            self._rows_written += self._parquet_rows(part)
            self._parts_written += 1
        self._next_part = self._discover_next_part()
        _fsync_directory(self.root)
        _fsync_directory(self.journal_root)

    def close(self) -> None:
        self._closed = True
