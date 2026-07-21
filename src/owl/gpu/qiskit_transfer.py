"""Packed bounded transfers for Qiskit validation and authoritative execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class PackedField:
    name: str
    dtype: str
    shape: tuple[int, ...]
    offset: int
    nbytes: int


@dataclass(frozen=True)
class PackedQiskitLayout:
    fields: tuple[PackedField, ...]
    total_bytes: int
    rows: int
    actions: int


@dataclass(frozen=True)
class UnpackedQiskitRows:
    probabilities: npt.NDArray[np.float64]
    phases: npt.NDArray[np.float64] | None
    authority: npt.NDArray[np.bool_] | None
    parent: npt.NDArray[np.float64] | None
    ow_ids: npt.NDArray[np.int64]
    flat_indices: npt.NDArray[np.int64] | None


def pack_qiskit_rows(
    xp: Any,
    *,
    probabilities: Any,
    phases: Any | None,
    authority: Any | None,
    parent: Any | None,
    ow_ids: Any,
    flat_indices: Any | None = None,
) -> tuple[Any, PackedQiskitLayout]:
    """Pack selected Qiskit rows into one device-resident byte slab.

    The slab preserves integer identity exactly and is intentionally bounded by
    the selected-row count.  It is a scheduled host boundary and must never be
    invoked from a captured CUDA graph.
    """

    normalized: list[tuple[str, Any]] = [
        ("probabilities", xp.ascontiguousarray(probabilities, dtype=xp.float64)),
        ("ow_ids", xp.ascontiguousarray(ow_ids, dtype=xp.int64)),
    ]
    if phases is not None:
        normalized.append(("phases", xp.ascontiguousarray(phases, dtype=xp.float64)))
    if authority is not None:
        normalized.append(("authority", xp.ascontiguousarray(authority, dtype=xp.uint8)))
    if parent is not None:
        normalized.append(("parent", xp.ascontiguousarray(parent, dtype=xp.float64)))
    if flat_indices is not None:
        normalized.append(("flat_indices", xp.ascontiguousarray(flat_indices, dtype=xp.int64)))

    fields: list[PackedField] = []
    offset = 0
    for name, array in normalized:
        alignment = max(1, int(array.dtype.itemsize))
        offset = ((offset + alignment - 1) // alignment) * alignment
        nbytes = int(array.nbytes)
        fields.append(
            PackedField(
                name=name,
                dtype=str(array.dtype),
                shape=tuple(int(value) for value in array.shape),
                offset=offset,
                nbytes=nbytes,
            )
        )
        offset += nbytes

    slab = xp.empty((offset,), dtype=xp.uint8)
    for field, (_, array) in zip(fields, normalized, strict=True):
        slab[field.offset : field.offset + field.nbytes] = array.view(xp.uint8).reshape(-1)

    rows = int(probabilities.shape[0])
    actions = int(probabilities.shape[-1]) if probabilities.ndim > 1 else 1
    return slab, PackedQiskitLayout(
        fields=tuple(fields),
        total_bytes=offset,
        rows=rows,
        actions=actions,
    )


def unpack_qiskit_rows(
    host_slab: npt.NDArray[np.uint8],
    layout: PackedQiskitLayout,
) -> UnpackedQiskitRows:
    values: dict[str, npt.NDArray[Any]] = {}
    contiguous = np.ascontiguousarray(host_slab, dtype=np.uint8)
    for field in layout.fields:
        raw = contiguous[field.offset : field.offset + field.nbytes]
        values[field.name] = (
            np.frombuffer(raw, dtype=np.dtype(field.dtype)).reshape(field.shape).copy()
        )

    probabilities = np.asarray(values["probabilities"], dtype=np.float64)
    phases_value = values.get("phases")
    authority_value = values.get("authority")
    parent_value = values.get("parent")
    flat_indices_value = values.get("flat_indices")
    return UnpackedQiskitRows(
        probabilities=probabilities,
        phases=None if phases_value is None else np.asarray(phases_value, dtype=np.float64),
        authority=(
            None
            if authority_value is None
            else np.asarray(authority_value, dtype=np.uint8).astype(bool)
        ),
        parent=None if parent_value is None else np.asarray(parent_value, dtype=np.float64),
        ow_ids=np.asarray(values["ow_ids"], dtype=np.int64),
        flat_indices=(
            None if flat_indices_value is None else np.asarray(flat_indices_value, dtype=np.int64)
        ),
    )
