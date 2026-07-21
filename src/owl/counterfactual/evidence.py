"""Branch-local event and contribution packet extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BranchEvidencePacket:
    tick: int
    event_arrays: dict[str, Any]
    contribution_arrays: dict[str, Any]
    event_codes: tuple[int, ...]
    contribution_codes: tuple[int, ...]
    contribution_fields: tuple[str, ...]
    event_overflow: Any

    @property
    def nbytes(self) -> int:
        arrays = (*self.event_arrays.values(), *self.contribution_arrays.values())
        return sum(int(getattr(value, "nbytes", 0)) for value in arrays)


def capture_branch_evidence(buffer: Any) -> BranchEvidencePacket:
    event_names = (
        "event_active",
        "event_stage_code",
        "event_reason_code",
        "event_source_y",
        "event_source_x",
        "event_target_y",
        "event_target_x",
        "event_target_ow_id",
        "event_payload",
    )
    contribution_names = (
        "contribution_delta",
        "tick_start",
        "tick_end",
    )
    return BranchEvidencePacket(
        tick=int(buffer.tick),
        event_arrays={name: buffer.arrays[name].copy() for name in event_names},
        contribution_arrays={name: buffer.arrays[name].copy() for name in contribution_names},
        event_codes=buffer.event_codes,
        contribution_codes=buffer.contribution_codes,
        contribution_fields=buffer.contribution_fields,
        event_overflow=buffer.arrays["event_overflow"].copy(),
    )


def transfer_branch_evidence(backend: Any, packet: BranchEvidencePacket) -> BranchEvidencePacket:
    """Materialize one bounded evidence packet at the declared D2H boundary."""
    return BranchEvidencePacket(
        tick=packet.tick,
        event_arrays={name: backend.asnumpy(value) for name, value in packet.event_arrays.items()},
        contribution_arrays={
            name: backend.asnumpy(value) for name, value in packet.contribution_arrays.items()
        },
        event_codes=packet.event_codes,
        contribution_codes=packet.contribution_codes,
        contribution_fields=packet.contribution_fields,
        event_overflow=backend.asnumpy(packet.event_overflow),
    )
