"""Registry-derived GPU data-layout ledger and checked field accessors.

This module does not decide scientific semantics. It converts the authoritative
field registry into an auditable layout plan used by persistent GPU execution.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .field_registry import FIELD_REGISTRY, FieldSpec


@dataclass(frozen=True)
class LayoutEntry:
    name: str
    shape_kind: str
    dtype: str
    owner: str
    layout_group: str
    moves_with_cell: bool
    clears_on_death: bool
    visual_role: str | None
    audit_role: str | None
    record_role: str | None
    proposed_layout: str
    reason: str


def _proposed_layout(spec: FieldSpec) -> tuple[str, str]:
    if spec.shape_kind in {"cell", "action", "channel"} and spec.moves_with_cell:
        return (
            f"persistent_slab:{spec.layout_group}",
            "cell-resident field participates in deterministic movement/reproduction/death",
        )
    if spec.shape_kind in {"field", "channel"}:
        return (
            "named_contiguous_plane",
            "stencil/environment access benefits from direct contiguous planes",
        )
    if spec.shape_kind == "action":
        return (
            "named_action_tensor",
            "decision/authority kernels require canonical trailing action dimension",
        )
    if spec.shape_kind == "patch":
        return ("named_patch_tensor", "patch reductions use exact block tiling")
    if spec.shape_kind == "global":
        return ("named_global_tensor", "small global reduction output")
    if spec.shape_kind == "event":
        return ("fixed_capacity_event_buffer", "graph-safe sparse event algebra")
    return ("named_array", "no safe packing advantage established")


def build_layout_ledger(state: Any | None = None) -> list[LayoutEntry]:
    specs: Iterable[FieldSpec] = FIELD_REGISTRY.values()
    out: list[LayoutEntry] = []
    for spec in specs:
        proposed, reason = _proposed_layout(spec)
        out.append(
            LayoutEntry(
                name=spec.name,
                shape_kind=spec.shape_kind,
                dtype=str(spec.dtype),
                owner=str(getattr(spec, "owner", "world")),
                layout_group=str(spec.layout_group),
                moves_with_cell=bool(spec.moves_with_cell),
                clears_on_death=bool(spec.clears_on_death),
                visual_role=spec.visual_role,
                audit_role=spec.audit_role,
                record_role=spec.record_role,
                proposed_layout=proposed,
                reason=reason,
            )
        )
    return out


def get_registered_array(device_state: Any, name: str) -> Any:
    """Return a device array only when ``name`` is registry-classified."""
    known = {entry.name for entry in build_layout_ledger(device_state)}
    if name not in known:
        raise KeyError(f"unregistered device field: {name}")
    if not hasattr(device_state, name):
        raise AttributeError(f"registered field is absent from device state: {name}")
    return getattr(device_state, name)


def write_layout_ledger(path: str | Path, state: Any | None = None) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    entries = build_layout_ledger(state)
    if output.suffix.lower() == ".json":
        output.write_text(
            json.dumps([asdict(x) for x in entries], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        lines = [
            "# GPU Data Layout Ledger",
            "",
            "| Field | Shape | Dtype | Group | Move | Clear | Proposed layout | Reason |",
            "|---|---|---|---|---:|---:|---|---|",
        ]
        for x in entries:
            lines.append(
                f"| `{x.name}` | `{x.shape_kind}` | `{x.dtype}` | `{x.layout_group}` | "
                f"{int(x.moves_with_cell)} | {int(x.clears_on_death)} | "
                f"`{x.proposed_layout}` | {x.reason} |"
            )
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output
