"""Stage action-math columnar batches with vectorized CuPy operations.

This module never changes scientific values.  It performs only masks, gathers,
repeats, broadcasts, and bounded device-to-host copies.  CuPy is imported lazily
so CPU and standalone-viewer installations remain supported.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from owl.record.action_math_batch import (
    ColumnarBatch,
    DictionaryEncodedColumn,
)
from owl.record.replay_schema import CompiledReplaySchema, ReplayShapeClass


def _authoritative_device_array_map(device_source: object) -> dict[str, object]:
    """Return device arrays without patch/global shadowing world-scale fields.

    ``DeviceState.arrays`` contains authoritative world-scale simulation fields.
    ``patch_arrays`` and ``global_arrays`` contain lower-dimensional diagnostics.
    Replay materialization must not let diagnostic maps replace fields such as
    health, occupancy, or selected_action.
    """

    arrays: dict[str, object] = dict(getattr(device_source, "arrays", {}))
    for map_name in ("patch_arrays", "global_arrays"):
        for name, value in dict(getattr(device_source, map_name, {})).items():
            arrays.setdefault(name, value)
    return arrays


class DeviceArraySource(Protocol):
    arrays: Mapping[str, Any]
    xp: Any
    tick: int


@dataclass(frozen=True)
class GPUStagingTelemetry:
    batches: int = 0
    rows: int = 0
    transfer_bytes: int = 0
    transfer_count: int = 0


@dataclass(frozen=True)
class CADCHostPacket:
    """Immutable host-side transfer boundary for one committed CADC tick."""

    tick: int
    stage_code: int
    schema_digest: str
    schema_version: str
    world_shape: tuple[int, int]
    channel_count: int
    event_codes: tuple[int, ...]
    contribution_codes: tuple[int, ...]
    contribution_fields: tuple[str, ...]
    arrays: dict[str, np.ndarray]
    transfer_bytes: int
    transfer_count: int
    source_backend: str


def collect_cadc_host_packet(source: Any) -> CADCHostPacket:
    """Cross the explicit CADC D2H boundary without row materialization."""
    buffer = source.metadata.get("cadc_device_buffer")
    if buffer is None:
        raise RuntimeError("device source has no CADC factual buffer")
    arrays: dict[str, np.ndarray] = {}
    transfer_bytes = 0
    transfer_count = 0
    is_gpu = bool(getattr(source.backend, "is_gpu", False))
    if is_gpu:
        cp = _cupy()
        stream = cp.cuda.Stream(non_blocking=True)
        with stream:
            for name, value in buffer.arrays.items():
                host = _pinned_array(
                    tuple(int(item) for item in value.shape), np.dtype(value.dtype)
                )
                value.get(out=host, stream=stream, blocking=False)
                arrays[name] = host
                transfer_bytes += int(host.nbytes)
                transfer_count += 1
            ready = cp.cuda.Event()
            ready.record(stream)
        # One packet-level synchronization replaces one blocking copy per
        # array. No device synchronization is performed per OW or row.
        ready.synchronize()
    else:
        for name, value in buffer.arrays.items():
            arrays[name] = np.array(value, copy=True, order="C")
            transfer_bytes += int(arrays[name].nbytes)
    for name, array in arrays.items():
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
            arrays[name] = array
        array.flags.writeable = False
    return CADCHostPacket(
        tick=int(buffer.tick),
        stage_code=int(buffer.stage_code),
        schema_digest=str(buffer.schema_digest),
        schema_version=str(buffer.schema_version),
        world_shape=(int(buffer.world_shape[0]), int(buffer.world_shape[1])),
        channel_count=int(buffer.channel_count),
        event_codes=tuple(int(item) for item in buffer.event_codes),
        contribution_codes=tuple(int(item) for item in buffer.contribution_codes),
        contribution_fields=tuple(str(item) for item in buffer.contribution_fields),
        arrays=arrays,
        transfer_bytes=transfer_bytes,
        transfer_count=transfer_count,
        source_backend=str(getattr(source.backend, "name", "unknown")),
    )


def cupy_available() -> bool:
    """Return whether the optional CuPy runtime can be imported."""

    try:
        import cupy  # noqa: F401
    except (ImportError, OSError):
        return False
    return True


def _cupy() -> Any:
    import cupy as cp

    return cp


def _pinned_array(shape: tuple[int, ...], dtype: np.dtype[Any]) -> np.ndarray:
    cp = _cupy()
    count = int(np.prod(shape, dtype=np.int64))
    memory = cp.cuda.alloc_pinned_memory(count * int(dtype.itemsize))
    result: np.ndarray[Any, np.dtype[Any]] = np.frombuffer(
        memory, dtype=dtype, count=count
    ).reshape(shape)
    return result


def collect_device_arrays(source: Any) -> dict[str, Any]:
    """Return a read-only name-to-device-array view without scientific mutation.

    World-scale ``source.arrays`` is authoritative. Patch/global diagnostic maps
    are additive only and must not overwrite world fields such as health,
    occupancy, readout, or selected action.
    """

    output: dict[str, Any] = _authoritative_device_array_map(source)
    for name in (
        "health",
        "occupancy",
        "readout",
        "possibility",
        "authority",
        "_authority_bool",
    ):
        value = getattr(source, name, None)
        if value is not None and hasattr(value, "shape"):
            output.setdefault(name, value)
    return output


class CuPyActionMathBatchBuilder:
    """GPU gather and pinned-transfer implementation of the canonical action table."""

    def __init__(
        self,
        compiled: CompiledReplaySchema,
        *,
        condition: str,
        seed: int,
        action_names: tuple[str, ...],
        max_batch_rows: int = 131_072,
        max_batch_bytes: int = 128 * 1024 * 1024,
        max_pinned_pool_bytes: int | None = None,
        full_validation: bool = False,
    ) -> None:
        if compiled.action_math_schema is None:
            raise ValueError("compiled schema has no action-math table")
        self.compiled = compiled
        self.condition = str(condition)
        self.seed = int(seed)
        self.action_names = tuple(action_names)
        self.max_batch_rows = max(1, int(max_batch_rows))
        self.max_batch_bytes = max(1, int(max_batch_bytes))
        self.full_validation = bool(full_validation)
        self.telemetry = GPUStagingTelemetry()
        # The iterator is consumed synchronously by the Parquet sink. A buffer is
        # safe to reuse when iteration resumes after each yield. Cache by column
        # name, shape, and dtype so repeated batches allocate pinned memory once.
        self._pinned_pool: dict[tuple[str, tuple[int, ...], str], np.ndarray] = {}
        self._pinned_pool_bytes = 0
        self._max_pinned_pool_bytes = max(
            1,
            int(
                max_pinned_pool_bytes
                if max_pinned_pool_bytes is not None
                else self.max_batch_bytes * 2
            ),
        )

    def _host_buffer(self, name: str, device: Any) -> np.ndarray:
        shape = tuple(int(value) for value in device.shape)
        dtype = np.dtype(device.dtype)
        key = (name, shape, dtype.str)
        existing = self._pinned_pool.get(key)
        if existing is not None:
            return existing
        required = int(np.prod(shape, dtype=np.int64)) * int(dtype.itemsize)
        if self._pinned_pool_bytes + required > self._max_pinned_pool_bytes:
            # A bounded synchronous iterator normally needs only one shape per
            # column plus one short final batch. Drop stale shape variants first.
            stale = [item for item in self._pinned_pool if item[0] == name]
            for stale_key in stale:
                released = self._pinned_pool.pop(stale_key)
                self._pinned_pool_bytes -= int(released.nbytes)
        if self._pinned_pool_bytes + required > self._max_pinned_pool_bytes:
            raise MemoryError(
                "GPU replay pinned pool budget exceeded: "
                f"{self._pinned_pool_bytes + required} > {self._max_pinned_pool_bytes}"
            )
        allocated = _pinned_array(shape, dtype)
        self._pinned_pool[key] = allocated
        self._pinned_pool_bytes += int(allocated.nbytes)
        return allocated

    def _row_limit(self) -> int:
        width = 0
        for spec in self.compiled.action_math_specs:
            width += 2 if spec.numpy_dtype is None else max(1, np.dtype(spec.numpy_dtype).itemsize)
        by_bytes = max(1, self.max_batch_bytes // max(width, 1))
        rows = min(self.max_batch_rows, by_bytes)
        actions = self.compiled.action_count
        return int(max(int(actions), int(rows - (rows % actions))))

    @staticmethod
    def _world_action(value: Any, cells: Any, h: int, w: int, actions: int) -> Any:
        if tuple(value.shape) == (h, w, actions):
            value = value.reshape(h * w, actions)
        elif tuple(value.shape) != (h * w, actions):
            raise ValueError(f"expected device world-action shape, got {tuple(value.shape)}")
        return value[cells, :].reshape(-1)

    @staticmethod
    def _world_scalar(value: Any, cells: Any, h: int, w: int) -> Any:
        if tuple(value.shape) == (h, w):
            value = value.reshape(h * w)
        elif tuple(value.shape) != (h * w,):
            raise ValueError(f"expected device world-scalar shape, got {tuple(value.shape)}")
        return value[cells]

    def iter_batches(
        self,
        source: DeviceArraySource | Any,
        *,
        tick: int,
        sampled: bool = False,
    ) -> Iterator[ColumnarBatch]:
        cp = _cupy()
        arrays = collect_device_arrays(source)
        h, w = self.compiled.world_shape
        actions = self.compiled.action_count
        health = arrays["health"].reshape(h, w)
        occupancy = arrays["occupancy"].reshape(h, w)
        live_flat = cp.flatnonzero(((health > 0) & (occupancy >= 0)).reshape(-1))
        live_ow = occupancy.reshape(-1)[live_flat]
        if sampled:
            keep = (live_ow % 32) == 0
            live_flat = live_flat[keep]
            live_ow = live_ow[keep]
        selected_source = arrays.get("raqic_readout", arrays.get("readout"))
        if selected_source is None:
            raise ValueError("device action-math recording requires readout")
        selected_live = selected_source.reshape(-1)[live_flat]
        living_count = int(live_flat.size)
        action_dtype = cp.int16 if actions <= np.iinfo(np.int16).max else cp.int32
        action_template = cp.arange(actions, dtype=action_dtype)
        stream = cp.cuda.Stream(non_blocking=True)
        batch_count = 0
        row_total = 0
        byte_total = 0
        transfer_count = 0

        start = 0
        while start < living_count:
            row_limit = self._row_limit()
            cells_per_batch = max(1, row_limit // actions)
            stop = min(living_count, start + cells_per_batch)
            local_count = stop - start
            rows = local_count * actions
            with stream:
                cells = live_flat[start:stop]
                y = cells // w
                x = cells % w
                action_index = cp.tile(action_template, local_count)
                ow_id = cp.repeat(live_ow[start:stop], actions)
                selected = action_index.astype(cp.int64) == cp.repeat(
                    selected_live[start:stop].astype(cp.int64), actions
                )
                device_columns: dict[str, Any] = {
                    "ow_id": ow_id.astype(cp.int64, copy=False),
                    "action_index": action_index,
                    "selected": selected,
                }
                for spec in self.compiled.action_math_specs:
                    name = spec.name
                    if name in {
                        "condition",
                        "seed",
                        "tick",
                        "ow_id",
                        "action_index",
                        "action_name",
                        "selected",
                    }:
                        continue
                    source_name = spec.source_field
                    if source_name == "_authority_bool" and source_name not in arrays:
                        source_name = "authority"
                    if source_name == "authority" and source_name not in arrays:
                        source_name = "_authority_bool"
                    if source_name is None or source_name not in arrays:
                        raise ValueError(f"device field disappeared: {source_name}")
                    value = arrays[source_name]
                    if spec.shape_class == ReplayShapeClass.WORLD_ACTION:
                        column = self._world_action(value, cells, h, w, actions)
                    elif spec.shape_class == ReplayShapeClass.WORLD_SCALAR:
                        column = cp.repeat(self._world_scalar(value, cells, h, w), actions)
                    elif spec.shape_class == ReplayShapeClass.GLOBAL_ACTION:
                        column = cp.tile(value.reshape(actions), local_count)
                    elif spec.shape_class == ReplayShapeClass.PATCH_ACTION:
                        ph, pw = int(value.shape[0]), int(value.shape[1])
                        patch_y = y // (h // ph)
                        patch_x = x // (w // pw)
                        column = value[patch_y, patch_x, :].reshape(rows)
                    else:
                        raise ValueError(f"unsupported device shape class: {spec.shape_class}")
                    device_columns[name] = cp.ascontiguousarray(column)

                host_columns: dict[str, Any] = {
                    "condition": DictionaryEncodedColumn(
                        np.zeros(rows, dtype=np.int8), (self.condition,)
                    ),
                    "seed": np.full(rows, self.seed, dtype=np.int64),
                    "tick": np.full(rows, int(tick), dtype=np.int64),
                    "action_name": DictionaryEncodedColumn(
                        np.tile(np.arange(actions, dtype=np.int16), local_count),
                        self.action_names,
                    ),
                }
                for name, device in device_columns.items():
                    host = self._host_buffer(name, device)
                    cp.asnumpy(device, out=host, stream=stream, blocking=False)
                    host_columns[name] = host
                    byte_total += int(host.nbytes)
                    transfer_count += 1
                ready = cp.cuda.Event()
                ready.record(stream)
            ready.synchronize()
            batch_count += 1
            row_total += rows
            yield ColumnarBatch(
                table_name="ow_action_math",
                tick=int(tick),
                columns=host_columns,
                schema=self.compiled.action_math_schema,
                row_count=rows,
                schema_digest=self.compiled.schema_digest,
            )
            start = stop

        self.telemetry = GPUStagingTelemetry(
            batches=batch_count,
            rows=row_total,
            transfer_bytes=byte_total,
            transfer_count=transfer_count,
        )
