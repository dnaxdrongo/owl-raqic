"""Write counterfactual packets to Parquet with atomic checksum receipts."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from owl.counterfactual.rng_registry import registry_manifest
from owl.counterfactual.schema import (
    COUNTERFACTUAL_SCHEMA_DIGEST,
    COUNTERFACTUAL_SCHEMA_VERSION,
)
from owl.counterfactual.staging import TablePacket


@dataclass(frozen=True)
class CounterfactualPartReceipt:
    table_name: str
    part_id: str
    path: str
    rows: int
    packet_bytes: int
    file_bytes: int
    sha256: str
    schema_digest: str
    source_sha256: str
    phase25_certificate_sha256: str
    factual_v2_digest: str
    rng_registry_digest: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class CounterfactualWriter:
    def __init__(
        self,
        root: str | Path,
        *,
        source_sha256: str,
        phase25_certificate_sha256: str,
        factual_v2_digest: str,
        max_packet_bytes: int,
        max_pending_bytes: int,
        row_group_rows: int,
        resume: bool = False,
        failure_injector: Callable[[str], None] | None = None,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.source_sha256 = source_sha256
        self.phase25_certificate_sha256 = phase25_certificate_sha256
        self.factual_v2_digest = factual_v2_digest
        self.max_packet_bytes = int(max_packet_bytes)
        self.max_pending_bytes = int(max_pending_bytes)
        self.row_group_rows = int(row_group_rows)
        self.failure_injector = failure_injector
        self.receipts: list[CounterfactualPartReceipt] = []
        self._remove_temporary_files()
        if resume:
            self._recover()

    def _remove_temporary_files(self) -> None:
        for path in self.root.rglob("*.tmp.*"):
            path.unlink(missing_ok=True)
        for path in self.root.rglob(".*.tmp.*"):
            path.unlink(missing_ok=True)

    def _recover(self) -> None:
        for receipt_path in sorted(self.root.glob("*/_journal/part-*.json")):
            payload = json.loads(receipt_path.read_text(encoding="utf-8"))
            part = receipt_path.parent.parent / payload["path"]
            if not part.exists() or _sha256(part) != payload["sha256"]:
                raise RuntimeError(f"counterfactual recovery checksum mismatch: {part}")
            if payload["schema_digest"] != COUNTERFACTUAL_SCHEMA_DIGEST:
                raise RuntimeError(f"counterfactual recovery schema mismatch: {part}")
            self.receipts.append(CounterfactualPartReceipt(**payload))

    def write_packet(self, packet: TablePacket) -> CounterfactualPartReceipt:
        if packet.nbytes > self.max_packet_bytes:
            raise MemoryError(
                f"{packet.table_name} packet {packet.nbytes:,} exceeds {self.max_packet_bytes:,}"
            )
        if packet.nbytes > self.max_pending_bytes:
            raise MemoryError("counterfactual pending-byte bound cannot hold packet")
        import pyarrow as pa
        import pyarrow.parquet as pq

        table_root = self.root / packet.table_name
        journal = table_root / "_journal"
        journal.mkdir(parents=True, exist_ok=True)
        part_index = len(list(table_root.glob("part-*.parquet")))
        final = table_root / f"part-{part_index:06d}.parquet"
        temporary = table_root / f".{final.name}.tmp.{os.getpid()}"
        metadata = {
            b"owl.counterfactual.schema_version": COUNTERFACTUAL_SCHEMA_VERSION.encode(),
            b"owl.counterfactual.schema_digest": COUNTERFACTUAL_SCHEMA_DIGEST.encode(),
            b"owl.phase3.source_sha256": self.source_sha256.encode(),
            b"owl.phase25.certificate_sha256": self.phase25_certificate_sha256.encode(),
            b"owl.cadc.factual_v2_digest": self.factual_v2_digest.encode(),
            b"owl.counterfactual.rng_registry_digest": registry_manifest()[
                "registry_digest"
            ].encode(),
        }
        table = pa.Table.from_pydict(packet.columns)
        table = table.replace_schema_metadata(metadata)
        pq.write_table(
            table,
            temporary,
            compression="zstd",
            row_group_size=max(1, min(self.row_group_rows, max(1, table.num_rows))),
            use_dictionary=True,
            write_statistics=True,
        )
        if self.failure_injector is not None:
            self.failure_injector(packet.table_name)
        os.replace(temporary, final)
        rng_digest = str(registry_manifest()["registry_digest"])
        receipt = CounterfactualPartReceipt(
            table_name=packet.table_name,
            part_id=hashlib.sha256(
                f"{packet.table_name}:{part_index}:{_sha256(final)}".encode()
            ).hexdigest(),
            path=final.name,
            rows=packet.rows,
            packet_bytes=packet.nbytes,
            file_bytes=final.stat().st_size,
            sha256=_sha256(final),
            schema_digest=COUNTERFACTUAL_SCHEMA_DIGEST,
            source_sha256=self.source_sha256,
            phase25_certificate_sha256=self.phase25_certificate_sha256,
            factual_v2_digest=self.factual_v2_digest,
            rng_registry_digest=rng_digest,
        )
        _atomic_json(journal / f"part-{part_index:06d}.json", asdict(receipt))
        self.receipts.append(receipt)
        return receipt

    def write_packets(
        self, packets: Iterable[TablePacket]
    ) -> tuple[CounterfactualPartReceipt, ...]:
        for packet in packets:
            self.write_packet(packet)
        _atomic_json(
            self.root / "counterfactual_manifest.json",
            {
                "schema_version": COUNTERFACTUAL_SCHEMA_VERSION,
                "schema_digest": COUNTERFACTUAL_SCHEMA_DIGEST,
                "source_sha256": self.source_sha256,
                "phase25_certificate_sha256": self.phase25_certificate_sha256,
                "factual_v2_digest": self.factual_v2_digest,
                "rng_registry": registry_manifest(),
                "parts": [asdict(receipt) for receipt in self.receipts],
            },
        )
        return tuple(self.receipts)
