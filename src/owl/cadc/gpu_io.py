"""Projected Parquet, DLPack, pinned-memory, and device-batch interfaces."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


class GPUParquetReader:
    """Read projected Parquet columns with a selected CPU/GPU dataframe engine."""

    def __init__(self, backend: str = "numpy") -> None:
        if backend not in {"numpy", "cupy"}:
            raise ValueError(f"unsupported Phase 4 I/O backend: {backend}")
        self.backend = backend

    def read(
        self,
        paths: Sequence[str | Path],
        *,
        columns: Sequence[str] | None = None,
        filters: Any | None = None,
    ) -> Any:
        """Read and concatenate parts without materializing unrequested columns."""
        normalized = [str(Path(path)) for path in paths]
        if not normalized:
            raise ValueError("at least one Parquet part is required")
        if self.backend == "cupy":
            try:
                import cudf
            except ImportError as exc:
                raise RuntimeError(
                    "GPU Parquet requires the target CUDA-matched RAPIDS cuDF package"
                ) from exc
            frames = [
                cudf.read_parquet(path, columns=list(columns) if columns else None)
                for path in normalized
            ]
            if filters is not None:
                raise NotImplementedError("cuDF filters must be applied as a device predicate")
            return cudf.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
        try:
            import pyarrow.dataset as ds
        except ImportError as exc:
            raise RuntimeError("CPU Parquet ETL requires the recording extra (PyArrow)") from exc
        dataset = ds.dataset(normalized, format="parquet")
        return dataset.to_table(columns=list(columns) if columns else None, filter=filters)

    def read_mapping(
        self,
        paths: Sequence[str | Path],
        *,
        columns: Sequence[str],
    ) -> dict[str, Any]:
        """Return projected columns as backend-native arrays for numerical stages."""
        frame = self.read(paths, columns=columns)
        if self.backend == "cupy":
            return {name: frame[name].values for name in columns}
        return {
            name: frame.column(name).to_numpy(zero_copy_only=False)
            for name in columns
        }


def to_torch_dlpack(value: Any) -> Any:
    """Create a zero-copy Torch tensor from a CUDA array via DLPack."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Torch is required for Phase 4 model training") from exc
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    if not hasattr(value, "__dlpack__"):
        raise TypeError("value does not expose the DLPack protocol")
    return torch.utils.dlpack.from_dlpack(value)


@dataclass(frozen=True)
class DeviceBatch:
    """One bounded backend-native batch and its source-row interval."""
    columns: Mapping[str, Any]
    row_start: int
    row_stop: int
    bytes: int


class PinnedBatchLoader:
    """Bounded deterministic batching with optional pinned host staging."""

    def __init__(
        self,
        columns: Mapping[str, Any],
        *,
        batch_size: int,
        pin_memory: bool,
        max_batch_bytes: int,
    ) -> None:
        if batch_size < 1 or max_batch_bytes < 1:
            raise ValueError("batch and byte bounds must be positive")
        if not columns:
            raise ValueError("batch loader needs at least one column")
        lengths = {len(value) for value in columns.values()}
        if len(lengths) != 1:
            raise ValueError("batch columns must have equal row counts")
        self.columns = dict(columns)
        self.rows = next(iter(lengths))
        self.batch_size = int(batch_size)
        self.pin_memory = bool(pin_memory)
        self.max_batch_bytes = int(max_batch_bytes)

    def __iter__(self) -> Iterator[DeviceBatch]:
        for start in range(0, self.rows, self.batch_size):
            stop = min(self.rows, start + self.batch_size)
            sliced = {name: value[start:stop] for name, value in self.columns.items()}
            total = sum(int(getattr(value, "nbytes", 0)) for value in sliced.values())
            if total > self.max_batch_bytes:
                raise MemoryError(
                    f"Phase 4 batch uses {total:,} bytes; bound is {self.max_batch_bytes:,}"
                )
            if self.pin_memory:
                sliced = {name: _pin(value) for name, value in sliced.items()}
            yield DeviceBatch(sliced, start, stop, total)


def _pin(value: Any) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("pinned batch loading requires Torch") from exc
    tensor = torch.as_tensor(value)
    if tensor.device.type != "cpu":
        return tensor
    return tensor.pin_memory()


def device_memory_snapshot() -> dict[str, int | str]:
    """Return positive CUDA pool/device metadata or a truthful CPU receipt."""
    try:
        import cupy as cp
    except ImportError:
        return {"backend": "cpu", "device_total": 0, "device_free": 0, "pool_used": 0}
    free, total = cp.cuda.runtime.memGetInfo()
    return {
        "backend": "cupy",
        "device_total": int(total),
        "device_free": int(free),
        "pool_used": int(cp.get_default_memory_pool().used_bytes()),
    }
