from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Literal


class TransferKind(StrEnum):
    METRIC = "metric"
    QISKIT = "qiskit"
    VISUAL = "visual"
    CHECKPOINT = "checkpoint"
    SHADOW = "shadow"
    DISTRIBUTED_VERIFY = "distributed_verify"
    SNAPSHOT = "snapshot"
    COMPATIBILITY = "compatibility"
    COUNTERFACTUAL_SOURCE = "counterfactual_source"
    COUNTERFACTUAL_BRANCH = "counterfactual_branch"
    COUNTERFACTUAL_HASH = "counterfactual_hash"
    COUNTERFACTUAL_OUTCOME = "counterfactual_outcome"
    COUNTERFACTUAL_EVENT = "counterfactual_event"
    COUNTERFACTUAL_CONTRIBUTION = "counterfactual_contribution"


TransferDirection = Literal["h2d", "d2h", "d2d"]
SynchronizationMode = Literal["none", "event", "stream", "device"]


@dataclass(frozen=True)
class TransferRecord:
    kind: TransferKind
    direction: TransferDirection
    tick: int
    bytes: int
    source_stream: str
    synchronization: SynchronizationMode
    scheduled: bool
    graph_compatible: bool
    reason: str

    def __post_init__(self) -> None:
        if self.tick < 0:
            raise ValueError("transfer tick must be non-negative")
        if self.bytes < 0:
            raise ValueError("transfer byte count must be non-negative")
        if not self.reason.strip():
            raise ValueError("transfer reason must be non-empty")
        if self.graph_compatible and self.synchronization in {"stream", "device"}:
            raise ValueError("synchronous host transfers cannot be graph-compatible")


@dataclass
class TransferLedger:
    """Per-run ownership ledger for host/device transfers.

    Every production-reachable copy is represented by a :class:`TransferRecord`.
    The aggregate counters are retained for compatibility with  reports,
    while strict certification consumes the structured records and rejects any
    unscheduled transfer.
    """

    h2d_bytes: int = 0
    d2h_bytes: int = 0
    d2d_bytes: int = 0
    metric_bytes: int = 0
    frame_bytes: int = 0
    qiskit_bytes: int = 0
    snapshot_bytes: int = 0
    checkpoint_bytes: int = 0
    shadow_bytes: int = 0
    distributed_verify_bytes: int = 0
    compatibility_bytes: int = 0
    async_copy_count: int = 0
    dropped_visual_frames: int = 0
    compute_waits_for_io: int = 0
    records: list[TransferRecord] = field(default_factory=list)

    def record(
        self,
        *,
        kind: TransferKind,
        direction: TransferDirection,
        tick: int,
        nbytes: int,
        source_stream: str,
        synchronization: SynchronizationMode,
        scheduled: bool,
        graph_compatible: bool,
        reason: str,
    ) -> TransferRecord:
        record = TransferRecord(
            kind=kind,
            direction=direction,
            tick=int(tick),
            bytes=int(nbytes),
            source_stream=str(source_stream),
            synchronization=synchronization,
            scheduled=bool(scheduled),
            graph_compatible=bool(graph_compatible),
            reason=str(reason),
        )
        self.records.append(record)
        if direction == "d2h":
            self.d2h_bytes += record.bytes
        elif direction == "h2d":
            self.h2d_bytes += record.bytes
        else:
            self.d2d_bytes += record.bytes
        if synchronization == "event":
            self.async_copy_count += 1
        self._record_kind_bytes(kind, record.bytes)
        return record

    def _record_kind_bytes(self, kind: TransferKind, nbytes: int) -> None:
        attr = {
            TransferKind.METRIC: "metric_bytes",
            TransferKind.VISUAL: "frame_bytes",
            TransferKind.QISKIT: "qiskit_bytes",
            TransferKind.SNAPSHOT: "snapshot_bytes",
            TransferKind.CHECKPOINT: "checkpoint_bytes",
            TransferKind.SHADOW: "shadow_bytes",
            TransferKind.DISTRIBUTED_VERIFY: "distributed_verify_bytes",
            TransferKind.COMPATIBILITY: "compatibility_bytes",
        }.get(kind)
        if attr is not None:
            setattr(self, attr, int(getattr(self, attr)) + int(nbytes))

    def record_d2h(
        self,
        nbytes: int,
        *,
        kind: str | TransferKind,
        tick: int = 0,
        source_stream: str = "transfer",
        synchronization: SynchronizationMode = "event",
        scheduled: bool = True,
        graph_compatible: bool = False,
        reason: str | None = None,
    ) -> TransferRecord:
        normalized = _normalize_kind(kind)
        return self.record(
            kind=normalized,
            direction="d2h",
            tick=tick,
            nbytes=nbytes,
            source_stream=source_stream,
            synchronization=synchronization,
            scheduled=scheduled,
            graph_compatible=graph_compatible,
            reason=reason or f"scheduled {normalized.value} device-to-host transfer",
        )

    def record_h2d(
        self,
        nbytes: int,
        *,
        kind: str | TransferKind = TransferKind.COMPATIBILITY,
        tick: int = 0,
        source_stream: str = "transfer",
        synchronization: SynchronizationMode = "event",
        scheduled: bool = True,
        graph_compatible: bool = False,
        reason: str | None = None,
    ) -> TransferRecord:
        normalized = _normalize_kind(kind)
        return self.record(
            kind=normalized,
            direction="h2d",
            tick=tick,
            nbytes=nbytes,
            source_stream=source_stream,
            synchronization=synchronization,
            scheduled=scheduled,
            graph_compatible=graph_compatible,
            reason=reason or f"scheduled {normalized.value} host-to-device transfer",
        )

    def record_d2d(
        self,
        nbytes: int,
        *,
        kind: str | TransferKind = TransferKind.COUNTERFACTUAL_BRANCH,
        tick: int = 0,
        source_stream: str = "counterfactual-source-copy",
        synchronization: SynchronizationMode = "event",
        scheduled: bool = True,
        graph_compatible: bool = False,
        reason: str | None = None,
    ) -> TransferRecord:
        normalized = _normalize_kind(kind)
        return self.record(
            kind=normalized,
            direction="d2d",
            tick=tick,
            nbytes=nbytes,
            source_stream=source_stream,
            synchronization=synchronization,
            scheduled=scheduled,
            graph_compatible=graph_compatible,
            reason=reason or f"scheduled {normalized.value} device-to-device copy",
        )

    @property
    def unscheduled_records(self) -> tuple[TransferRecord, ...]:
        return tuple(record for record in self.records if not record.scheduled)

    @property
    def graph_incompatible_records(self) -> tuple[TransferRecord, ...]:
        return tuple(record for record in self.records if not record.graph_compatible)

    def assert_production_safe(self) -> None:
        if self.unscheduled_records:
            details = ", ".join(
                f"{record.kind.value}@tick{record.tick}:{record.bytes}B"
                for record in self.unscheduled_records
            )
            raise RuntimeError(f"unscheduled host/device transfers detected: {details}")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "owl.transfer-ledger.v1",
            "h2d_bytes": self.h2d_bytes,
            "d2h_bytes": self.d2h_bytes,
            "d2d_bytes": self.d2d_bytes,
            "metric_bytes": self.metric_bytes,
            "frame_bytes": self.frame_bytes,
            "qiskit_bytes": self.qiskit_bytes,
            "snapshot_bytes": self.snapshot_bytes,
            "checkpoint_bytes": self.checkpoint_bytes,
            "shadow_bytes": self.shadow_bytes,
            "distributed_verify_bytes": self.distributed_verify_bytes,
            "compatibility_bytes": self.compatibility_bytes,
            "async_copy_count": self.async_copy_count,
            "dropped_visual_frames": self.dropped_visual_frames,
            "compute_waits_for_io": self.compute_waits_for_io,
            "unscheduled_count": len(self.unscheduled_records),
            "records": [asdict(record) for record in self.records],
        }

    def write(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return output_path


def _normalize_kind(value: str | TransferKind) -> TransferKind:
    if isinstance(value, TransferKind):
        return value
    aliases = {
        "metrics": TransferKind.METRIC,
        "metric": TransferKind.METRIC,
        "frame": TransferKind.VISUAL,
        "visual": TransferKind.VISUAL,
        "qiskit": TransferKind.QISKIT,
        "snapshot": TransferKind.SNAPSHOT,
        "checkpoint": TransferKind.CHECKPOINT,
        "shadow": TransferKind.SHADOW,
        "distributed": TransferKind.DISTRIBUTED_VERIFY,
        "distributed_verify": TransferKind.DISTRIBUTED_VERIFY,
        "compatibility": TransferKind.COMPATIBILITY,
        "counterfactual_source": TransferKind.COUNTERFACTUAL_SOURCE,
        "counterfactual_branch": TransferKind.COUNTERFACTUAL_BRANCH,
        "counterfactual_hash": TransferKind.COUNTERFACTUAL_HASH,
        "counterfactual_outcome": TransferKind.COUNTERFACTUAL_OUTCOME,
        "counterfactual_event": TransferKind.COUNTERFACTUAL_EVENT,
        "counterfactual_contribution": TransferKind.COUNTERFACTUAL_CONTRIBUTION,
    }
    try:
        return aliases[str(value)]
    except KeyError as exc:
        raise ValueError(f"unknown transfer kind: {value!r}") from exc
