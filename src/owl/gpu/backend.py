from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any

import numpy as np

from owl_raqic.gpu.backend import GPUBackendInfo, MissingGPUBackendError, detect_cupy


@dataclass(frozen=True)
class ArrayBackend:
    """Small array namespace wrapper for CPU-or-GPU full-stack execution."""

    name: str
    xp: Any
    is_gpu: bool
    info: dict[str, Any]

    def asarray(self, value: Any, dtype: Any | None = None) -> Any:
        return self.xp.asarray(value, dtype=dtype)

    def asnumpy(self, value: Any) -> Any:
        if self.is_gpu:
            return self.xp.asnumpy(value)
        return np.asarray(value)

    def synchronize(self) -> None:
        if self.is_gpu:
            self.xp.cuda.Stream.null.synchronize()


def get_array_backend(
    *, strict: bool = False, allow_fallback: bool = True, force: str | None = None
) -> ArrayBackend:
    """Return CuPy backend when available, otherwise NumPy fallback when allowed."""
    if force is not None and force not in {"numpy", "cupy"}:
        raise ValueError(f"unknown forced array backend: {force}")
    if force == "numpy":
        return ArrayBackend(
            "numpy",
            np,
            False,
            {"available": True, "backend": "numpy", "forced": True},
        )
    info: GPUBackendInfo = detect_cupy()
    if info.available:
        import cupy as cp

        return ArrayBackend("cupy", cp, True, info.to_dict())
    if force == "cupy" or (strict and not allow_fallback):
        raise MissingGPUBackendError(
            "gpu_full requested strict CUDA execution but CuPy/CUDA is unavailable. "
            f"Detection error: {info.error}"
        )
    return ArrayBackend(
        "numpy", np, False, {"available": False, "backend": "numpy", "error": info.error}
    )


def cupy_importable() -> bool:
    return importlib.util.find_spec("cupy") is not None
