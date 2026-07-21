"""Bounded NumPy column builders for replay state, decisions, and action math."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from owl.record.replay_schema import (
    SELECTED_DECISION_FIELDS,
    CompiledReplaySchema,
    ReplayShapeClass,
)


@dataclass(frozen=True)
class DictionaryEncodedColumn:
    indices: np.ndarray
    dictionary: tuple[str, ...]


@dataclass(frozen=True)
class MaskedColumn:
    values: np.ndarray
    invalid: np.ndarray


@dataclass(frozen=True)
class LivingIndex:
    flat: np.ndarray
    y: np.ndarray
    x: np.ndarray
    ow_id: np.ndarray
    selected: np.ndarray

    @property
    def count(self) -> int:
        return int(self.flat.size)


@dataclass(frozen=True)
class ColumnarBatch:
    table_name: str
    tick: int
    columns: Mapping[str, Any]
    schema: Any
    row_count: int
    schema_digest: str

    def to_record_batch(self, *, full_validation: bool = False) -> Any:
        import pyarrow as pa

        arrays: list[Any] = []
        for field in self.schema:
            value = self.columns[field.name]
            if isinstance(value, DictionaryEncodedColumn):
                index_type = field.type.index_type
                indices = pa.array(value.indices, type=index_type, from_pandas=False)
                dictionary = pa.array(value.dictionary, type=field.type.value_type)
                array = pa.DictionaryArray.from_arrays(indices, dictionary)
            elif isinstance(value, MaskedColumn):
                array = pa.array(
                    value.values,
                    type=field.type,
                    mask=value.invalid,
                    from_pandas=False,
                    safe=True,
                )
            else:
                array = pa.array(value, type=field.type, from_pandas=False, safe=True)
            arrays.append(array)
        batch = pa.RecordBatch.from_arrays(arrays, schema=self.schema)
        batch.validate(full=full_validation)
        if batch.num_rows != self.row_count:
            raise RuntimeError(
                f"{self.table_name} batch row mismatch: {batch.num_rows} != {self.row_count}"
            )
        return batch


def build_living_index(
    arrays: Mapping[str, np.ndarray],
    *,
    world_shape: tuple[int, int],
) -> LivingIndex:
    h, w = world_shape
    health = np.asarray(arrays["health"])
    occupancy = np.asarray(arrays.get("occupancy", np.full((h, w), -1, dtype=np.int64)))
    if health.shape == (h * w,):
        health = health.reshape(h, w)
    if occupancy.shape == (h * w,):
        occupancy = occupancy.reshape(h, w)
    if health.shape != (h, w) or occupancy.shape != (h, w):
        raise ValueError("health and occupancy must match the compiled world shape")
    living_flat = np.flatnonzero(((health > 0) & (occupancy >= 0)).reshape(-1, order="C"))
    living_flat = np.ascontiguousarray(living_flat.astype(np.int64, copy=False))
    y = np.ascontiguousarray((living_flat // w).astype(np.int32, copy=False))
    x = np.ascontiguousarray((living_flat % w).astype(np.int32, copy=False))
    ow_id = np.ascontiguousarray(
        occupancy.reshape(-1, order="C")[living_flat].astype(np.int64, copy=False)
    )
    selected_source = arrays.get("raqic_readout", arrays.get("readout"))
    if selected_source is None:
        selected = np.full(living_flat.size, -1, dtype=np.int16)
    else:
        selected_array = np.asarray(selected_source)
        selected = np.ascontiguousarray(selected_array.reshape(-1, order="C")[living_flat])
    return LivingIndex(flat=living_flat, y=y, x=x, ow_id=ow_id, selected=selected)


class NumPyReplayBatchBuilder:
    """Materialize replay tables through compiled NumPy operations only."""

    def __init__(
        self,
        compiled: CompiledReplaySchema,
        *,
        condition: str,
        seed: int,
        action_names: tuple[str, ...],
        max_batch_rows: int = 131_072,
        max_batch_bytes: int = 128 * 1024 * 1024,
        full_validation: bool = False,
    ) -> None:
        self.compiled = compiled
        self.condition = str(condition)
        self.seed = int(seed)
        self.action_names = tuple(action_names)
        self.max_batch_rows = max(1, int(max_batch_rows))
        self.max_batch_bytes = max(1, int(max_batch_bytes))
        self.full_validation = bool(full_validation)
        self._action_dtype = (
            np.int16 if compiled.action_count <= np.iinfo(np.int16).max else np.int32
        )
        self._action_template: np.ndarray[Any, np.dtype[Any]] = np.arange(
            compiled.action_count, dtype=self._action_dtype
        )

    @staticmethod
    def _estimated_width(specs: tuple[Any, ...]) -> int:
        width = 0
        for spec in specs:
            if spec.numpy_dtype is None:
                width += 2
            else:
                width += max(1, np.dtype(spec.numpy_dtype).itemsize)
        return max(width, 1)

    def _row_limit(self, *, action_granular: bool) -> int:
        specs = self.compiled.action_math_specs if action_granular else self.compiled.decision_specs
        by_bytes = max(1, self.max_batch_bytes // self._estimated_width(specs))
        limit = max(1, min(self.max_batch_rows, by_bytes))
        if action_granular:
            actions = self.compiled.action_count
            if actions <= 0:
                return 0
            limit = max(actions, limit - (limit % actions))
        return limit

    def _dictionary_condition(self, rows: int) -> DictionaryEncodedColumn:
        return DictionaryEncodedColumn(np.zeros(rows, dtype=np.int8), (self.condition,))

    def _base_ow_columns(
        self,
        living: LivingIndex,
        slc: slice,
        *,
        tick: int,
    ) -> dict[str, Any]:
        rows = int(living.flat[slc].size)
        return {
            "condition": self._dictionary_condition(rows),
            "seed": np.full(rows, self.seed, dtype=np.int64),
            "tick": np.full(rows, int(tick), dtype=np.int64),
            "ow_id": np.ascontiguousarray(living.ow_id[slc], dtype=np.int64),
            "y": np.ascontiguousarray(living.y[slc], dtype=np.int32),
            "x": np.ascontiguousarray(living.x[slc], dtype=np.int32),
        }

    @staticmethod
    def _world_scalar(
        array: np.ndarray, cells: np.ndarray, world_shape: tuple[int, int]
    ) -> np.ndarray:
        h, w = world_shape
        value = np.asarray(array)
        if value.shape == (h, w):
            value = value.reshape(h * w, order="C")
        elif value.shape != (h * w,):
            raise ValueError(f"expected world scalar for {value.shape}")
        result: np.ndarray[Any, np.dtype[Any]] = np.ascontiguousarray(value[cells])
        return result

    @staticmethod
    def _world_action(
        array: np.ndarray,
        cells: np.ndarray,
        *,
        world_shape: tuple[int, int],
        action_count: int,
    ) -> np.ndarray:
        h, w = world_shape
        value = np.asarray(array)
        if value.shape == (h, w, action_count):
            value = value.reshape(h * w, action_count, order="C")
        elif value.shape != (h * w, action_count):
            raise ValueError(f"expected world-action array, got {value.shape}")
        result: np.ndarray[Any, np.dtype[Any]] = np.ascontiguousarray(
            value[cells, :].reshape(-1, order="C")
        )
        return result

    def iter_state_batches(
        self,
        arrays: Mapping[str, np.ndarray],
        *,
        tick: int,
        living: LivingIndex | None = None,
    ) -> Iterator[ColumnarBatch]:
        living = living or build_living_index(arrays, world_shape=self.compiled.world_shape)
        limit = self._row_limit(action_granular=False)
        for start in range(0, living.count, limit):
            stop = min(living.count, start + limit)
            slc = slice(start, stop)
            cells = living.flat[slc]
            columns = self._base_ow_columns(living, slc, tick=tick)
            for spec in self.compiled.state_specs:
                if spec.name in columns or spec.source_field is None:
                    continue
                columns[spec.name] = self._world_scalar(
                    np.asarray(arrays[spec.source_field]), cells, self.compiled.world_shape
                )
            yield ColumnarBatch(
                table_name="ow_state",
                tick=int(tick),
                columns=columns,
                schema=self.compiled.state_schema,
                row_count=stop - start,
                schema_digest=self.compiled.schema_digest,
            )

    def iter_decision_batches(
        self,
        arrays: Mapping[str, np.ndarray],
        *,
        tick: int,
        living: LivingIndex | None = None,
    ) -> Iterator[ColumnarBatch]:
        living = living or build_living_index(arrays, world_shape=self.compiled.world_shape)
        limit = self._row_limit(action_granular=False)
        action_count = self.compiled.action_count
        authority_name = (
            "_authority_bool"
            if "_authority_bool" in arrays
            else "authority"
            if "authority" in arrays
            else None
        )
        for start in range(0, living.count, limit):
            stop = min(living.count, start + limit)
            slc = slice(start, stop)
            cells = living.flat[slc]
            selected = np.ascontiguousarray(living.selected[slc])
            selected64 = selected.astype(np.int64, copy=False)
            valid = (selected64 >= 0) & (selected64 < action_count)
            safe_selected = np.where(valid, selected64, 0)
            columns = self._base_ow_columns(living, slc, tick=tick)
            for spec in self.compiled.state_specs:
                if spec.name in columns or spec.source_field is None:
                    continue
                columns[spec.name] = self._world_scalar(
                    np.asarray(arrays[spec.source_field]), cells, self.compiled.world_shape
                )
            columns["selected_action"] = selected
            if (
                authority_name is not None
                and "legal_action_count" in self.compiled.decision_schema.names
            ):
                authority = np.asarray(arrays[authority_name])
                h, w = self.compiled.world_shape
                authority2 = authority.reshape(h * w, action_count, order="C")
                columns["legal_action_count"] = np.ascontiguousarray(
                    np.count_nonzero(authority2[cells, :], axis=1).astype(np.int32)
                )
            for source, destination in SELECTED_DECISION_FIELDS.items():
                if destination not in self.compiled.decision_schema.names or source not in arrays:
                    continue
                value = np.asarray(arrays[source])
                h, w = self.compiled.world_shape
                value2 = value.reshape(h * w, action_count, order="C")[cells, :]
                gathered = value2[np.arange(value2.shape[0]), safe_selected]
                columns[destination] = MaskedColumn(
                    np.ascontiguousarray(gathered), np.ascontiguousarray(~valid)
                )
            for name in self.compiled.decision_schema.names:
                if name in columns or name not in arrays:
                    continue
                value = np.asarray(arrays[name])
                if value.ndim <= 2:
                    columns[name] = self._world_scalar(value, cells, self.compiled.world_shape)
            yield ColumnarBatch(
                table_name="ow_decisions",
                tick=int(tick),
                columns=columns,
                schema=self.compiled.decision_schema,
                row_count=stop - start,
                schema_digest=self.compiled.schema_digest,
            )

    def iter_action_math_batches(
        self,
        arrays: Mapping[str, np.ndarray],
        *,
        tick: int,
        living: LivingIndex | None = None,
        sampled: bool = False,
    ) -> Iterator[ColumnarBatch]:
        schema = self.compiled.action_math_schema
        if schema is None:
            return
        living = living or build_living_index(arrays, world_shape=self.compiled.world_shape)
        if sampled:
            keep = (living.ow_id % 32) == 0
            living = LivingIndex(
                flat=np.ascontiguousarray(living.flat[keep]),
                y=np.ascontiguousarray(living.y[keep]),
                x=np.ascontiguousarray(living.x[keep]),
                ow_id=np.ascontiguousarray(living.ow_id[keep]),
                selected=np.ascontiguousarray(living.selected[keep]),
            )
        action_count = self.compiled.action_count
        h, w = self.compiled.world_shape
        start = 0
        while start < living.count:
            row_limit = self._row_limit(action_granular=True)
            cells_per_batch = max(1, row_limit // action_count)
            stop = min(living.count, start + cells_per_batch)
            cells = living.flat[start:stop]
            local_count = int(cells.size)
            rows = local_count * action_count
            action_index = np.tile(self._action_template, local_count)
            ow_id: np.ndarray[Any, np.dtype[Any]] = np.repeat(
                living.ow_id[start:stop], action_count
            )
            selected: np.ndarray[Any, np.dtype[Any]] = np.repeat(
                living.selected[start:stop], action_count
            )
            columns: dict[str, Any] = {
                "condition": self._dictionary_condition(rows),
                "seed": np.full(rows, self.seed, dtype=np.int64),
                "tick": np.full(rows, int(tick), dtype=np.int64),
                "ow_id": np.ascontiguousarray(ow_id, dtype=np.int64),
                "action_index": np.ascontiguousarray(action_index),
                "action_name": DictionaryEncodedColumn(
                    np.ascontiguousarray(action_index.astype(np.int16, copy=False)),
                    self.action_names,
                ),
                "selected": np.ascontiguousarray(
                    action_index.astype(np.int64) == selected.astype(np.int64)
                ),
            }
            y: np.ndarray[Any, np.dtype[Any]] = living.y[start:stop].astype(np.int64, copy=False)
            x: np.ndarray[Any, np.dtype[Any]] = living.x[start:stop].astype(np.int64, copy=False)
            for spec in self.compiled.action_math_specs:
                name = spec.name
                if name in columns or spec.source_field is None:
                    continue
                source_name = spec.source_field
                if source_name == "_authority_bool" and source_name not in arrays:
                    source_name = "authority"
                if source_name == "authority" and source_name not in arrays:
                    source_name = "_authority_bool"
                if source_name not in arrays:
                    raise ValueError(f"required action-math field disappeared: {source_name}")
                value = np.asarray(arrays[source_name])
                if spec.shape_class == ReplayShapeClass.WORLD_ACTION:
                    columns[name] = self._world_action(
                        value,
                        cells,
                        world_shape=self.compiled.world_shape,
                        action_count=action_count,
                    )
                elif spec.shape_class == ReplayShapeClass.WORLD_SCALAR:
                    scalar = self._world_scalar(value, cells, self.compiled.world_shape)
                    columns[name] = np.ascontiguousarray(np.repeat(scalar, action_count))
                elif spec.shape_class == ReplayShapeClass.GLOBAL_ACTION:
                    columns[name] = np.ascontiguousarray(
                        np.tile(value.reshape(action_count), local_count)
                    )
                elif spec.shape_class == ReplayShapeClass.PATCH_ACTION:
                    ph, pw = int(value.shape[0]), int(value.shape[1])
                    patch_size_y, patch_size_x = h // ph, w // pw
                    patch_y: np.ndarray[Any, np.dtype[Any]] = y // patch_size_y
                    patch_x: np.ndarray[Any, np.dtype[Any]] = x // patch_size_x
                    columns[name] = np.ascontiguousarray(
                        value[patch_y, patch_x, :].reshape(rows, order="C")
                    )
                else:
                    raise ValueError(f"unsupported action shape class: {spec.shape_class}")
            yield ColumnarBatch(
                table_name="ow_action_math",
                tick=int(tick),
                columns=columns,
                schema=schema,
                row_count=rows,
                schema_digest=self.compiled.schema_digest,
            )
            start = stop
