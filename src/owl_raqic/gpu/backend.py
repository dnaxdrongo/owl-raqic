from __future__ import annotations

import importlib.util
import traceback
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class GPUBackendInfo:
    available: bool
    backend: str = "cupy"
    device_name: str | None = None
    device_id: int | None = None
    total_memory: int | None = None
    free_memory: int | None = None
    cupy_version: str | None = None
    cuda_runtime_version: int | None = None
    compute_capability: str | None = None
    float64_test_passed: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MissingGPUBackendError(RuntimeError):
    pass


def detect_cupy(run_operation: bool = True) -> GPUBackendInfo:
    """Detect a real CuPy/CUDA runtime, not just an importable module."""
    if importlib.util.find_spec("cupy") is None:
        return GPUBackendInfo(available=False, error="cupy is not installed")
    try:
        import cupy as cp

        device = cp.cuda.Device()
        attrs = cp.cuda.runtime.getDeviceProperties(device.id)
        name = attrs.get("name", b"unknown")
        if isinstance(name, bytes):
            name = name.decode(errors="replace")
        free_mem, total_mem = cp.cuda.runtime.memGetInfo()
        cc = f"{attrs.get('major', '?')}.{attrs.get('minor', '?')}"
        ok64 = False
        if run_operation:
            x = cp.arange(8, dtype=cp.float64)
            y = cp.sum(x * x)
            cp.cuda.Stream.null.synchronize()
            ok64 = bool(abs(float(y.get()) - 140.0) < 1e-12)
        return GPUBackendInfo(
            available=True,
            device_name=str(name),
            device_id=int(device.id),
            total_memory=int(total_mem),
            free_memory=int(free_mem),
            cupy_version=getattr(cp, "__version__", None),
            cuda_runtime_version=int(cp.cuda.runtime.runtimeGetVersion()),
            compute_capability=cc,
            float64_test_passed=ok64,
        )
    except Exception as exc:
        return GPUBackendInfo(
            available=False,
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=5)}",
        )


def require_cupy() -> Any:
    info = detect_cupy()
    if not info.available:
        raise MissingGPUBackendError(
            "CuPy/CUDA GPU backend is unavailable. Install the GPU extra on a Linux CUDA host "
            "with `pip install -e .[gpu]` or use a CPU RAQIC mode. "
            f"Detection error: {info.error}"
        )
    import cupy as cp

    return cp


def get_device_info() -> dict[str, Any]:
    return detect_cupy().to_dict()


def get_memory_info() -> dict[str, int | None]:
    info = detect_cupy(run_operation=False)
    return {"free_memory": info.free_memory, "total_memory": info.total_memory}


def synchronize() -> None:
    cp = require_cupy()
    cp.cuda.Stream.null.synchronize()


def asnumpy(x: Any) -> Any:
    cp = require_cupy()
    return cp.asnumpy(x)
