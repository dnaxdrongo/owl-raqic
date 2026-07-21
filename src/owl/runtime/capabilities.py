from __future__ import annotations

import importlib.util
from dataclasses import asdict, dataclass
from typing import Any, cast

from .json_types import json_native


@dataclass(frozen=True)
class RuntimeCapabilities:
    """Runtime features detected without importing optional packages eagerly."""

    cupy_available: bool
    cuda_device_count: int
    qiskit_available: bool
    aer_available: bool
    aer_gpu_available: bool
    pygame_available: bool
    vispy_available: bool
    nccl_available: bool
    details: dict[str, Any]

    @property
    def has_cuda(self) -> bool:
        return self.cupy_available and self.cuda_device_count > 0

    def to_dict(self) -> dict[str, Any]:
        data = json_native(asdict(self))
        data["has_cuda"] = bool(self.has_cuda)
        return cast(dict[str, Any], data)


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def detect_runtime_capabilities() -> RuntimeCapabilities:
    """Detect optional runtime capabilities conservatively.

    Import failures are reported through ``details``. Merely having a package
    installed does not imply that a CUDA device or Aer GPU method is usable.
    """

    details: dict[str, Any] = {}
    cupy_available = _module_available("cupy")
    cuda_device_count = 0
    nccl_available = False
    if cupy_available:
        try:
            import cupy as cp

            cuda_device_count = int(cp.cuda.runtime.getDeviceCount())
            nccl_available = bool(
                getattr(cp.cuda, "nccl", None) is not None
                and hasattr(cp.cuda.nccl, "NcclCommunicator")
            )
            details["cupy_version"] = getattr(cp, "__version__", "unknown")
            details["cuda_runtime_version"] = int(cp.cuda.runtime.runtimeGetVersion())
        except Exception as exc:
            details["cupy_error"] = f"{type(exc).__name__}: {exc}"
            cuda_device_count = 0
            nccl_available = False

    qiskit_available = _module_available("qiskit")
    aer_available = _module_available("qiskit_aer")
    aer_gpu_available = False
    if aer_available:
        try:
            from qiskit_aer import AerSimulator

            simulator = AerSimulator()
            devices = tuple(str(x) for x in simulator.available_devices())
            details["aer_devices"] = devices
            details["aer_methods"] = tuple(str(x) for x in simulator.available_methods())
            aer_gpu_available = any(device.upper() == "GPU" for device in devices)
        except Exception as exc:
            details["aer_error"] = f"{type(exc).__name__}: {exc}"

    return RuntimeCapabilities(
        cupy_available=cupy_available,
        cuda_device_count=cuda_device_count,
        qiskit_available=qiskit_available,
        aer_available=aer_available,
        aer_gpu_available=aer_gpu_available,
        pygame_available=_module_available("pygame"),
        vispy_available=_module_available("vispy"),
        nccl_available=nccl_available,
        details=details,
    )
