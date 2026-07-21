from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from owl.gpu.commands import CommandKind, GPUCommand

_COMMAND_CODE = {
    CommandKind.INJECT_FOOD: 1,
    CommandKind.INJECT_TOXIN: 2,
}


@dataclass(frozen=True)
class CommandKernelKey:
    source_sha256: str
    options: tuple[str, ...]
    device_id: int
    compute_capability: str


_KERNEL_CACHE: dict[CommandKernelKey, Any] = {}
_KERNEL_COMPILE_COUNT = 0
_KERNEL_OPTIONS = ("--std=c++11",)


@dataclass
class DeviceCommandBuffer:
    command_type: Any
    y: Any
    x: Any
    amount: Any
    active: Any
    count: Any
    dummy64: Any
    dummy32: Any

    @classmethod
    def create(cls, backend: Any, capacity: int) -> DeviceCommandBuffer:
        xp = backend.xp
        capacity = max(1, int(capacity))
        return cls(
            command_type=xp.zeros((capacity,), dtype=xp.int32),
            y=xp.zeros((capacity,), dtype=xp.int32),
            x=xp.zeros((capacity,), dtype=xp.int32),
            amount=xp.zeros((capacity,), dtype=xp.float64),
            active=xp.zeros((capacity,), dtype=xp.uint8),
            count=xp.zeros((1,), dtype=xp.int32),
            dummy64=xp.zeros((1,), dtype=xp.float64),
            dummy32=xp.zeros((1,), dtype=xp.float32),
        )

    @property
    def capacity(self) -> int:
        return int(self.command_type.shape[0])

    def encode(self, commands: list[GPUCommand], backend: Any) -> list[dict[str, Any]]:
        # Aggregate state-mutating commands and copy them into fixed arrays.
        aggregated: dict[tuple[CommandKind, int, int], float] = {}
        metadata: list[dict[str, Any]] = []
        for command in commands:
            if command.kind not in _COMMAND_CODE:
                raise ValueError(f"not a device-state command: {command.kind}")
            if not command.state_mutating:
                raise PermissionError(f"{command.kind.value} requires state_mutating=True")
            y = int(command.payload["y"])
            x = int(command.payload["x"])
            amount = float(command.payload.get("amount", 0.0))
            key = (command.kind, y, x)
            aggregated[key] = aggregated.get(key, 0.0) + amount
        if len(aggregated) > self.capacity:
            raise OverflowError(
                f"device command capacity exceeded: {len(aggregated)} > {self.capacity}"
            )
        kinds = np.zeros((self.capacity,), dtype=np.int32)
        yy = np.zeros((self.capacity,), dtype=np.int32)
        xx = np.zeros((self.capacity,), dtype=np.int32)
        amounts = np.zeros((self.capacity,), dtype=np.float64)
        active = np.zeros((self.capacity,), dtype=np.uint8)
        for row, ((kind, y, x), amount) in enumerate(
            sorted(
                aggregated.items(),
                key=lambda item: (
                    int(item[0][1]),
                    int(item[0][2]),
                    item[0][0].value,
                ),
            )
        ):
            kinds[row] = _COMMAND_CODE[kind]
            yy[row] = y
            xx[row] = x
            amounts[row] = amount
            active[row] = 1
            metadata.append(
                {
                    "kind": kind.value,
                    "y": y,
                    "x": x,
                    "amount": amount,
                    "queued_for_device": True,
                }
            )
        xp = backend.xp
        self.command_type[...] = xp.asarray(kinds)
        self.y[...] = xp.asarray(yy)
        self.x[...] = xp.asarray(xx)
        self.amount[...] = xp.asarray(amounts)
        self.active[...] = xp.asarray(active)
        self.count[0] = len(aggregated)
        return metadata


def _kernel(cp: Any) -> Any:
    global _KERNEL_COMPILE_COUNT
    device = cp.cuda.Device()
    source = r"""
    extern "C" __global__
    void apply_commands(
        const int* kind,
        const int* yy,
        const int* xx,
        const double* amount,
        unsigned char* active,
        const int* count,
        double* food64,
        double* toxin64,
        float* food32,
        float* toxin32,
        const int h,
        const int w,
        const int use64
    ) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i >= count[0] || active[i] == 0) return;
        int y = yy[i];
        int x = xx[i];
        if (y >= 0 && y < h && x >= 0 && x < w) {
            int index = y * w + x;
            if (use64) {
                if (kind[i] == 1) food64[index] = fmin(1.0, fmax(0.0, food64[index] + amount[i]));
                if (kind[i] == 2) toxin64[index] = fmin(1.0, fmax(0.0, toxin64[index] + amount[i]));
            } else {
                float delta = (float)amount[i];
                if (kind[i] == 1) food32[index] = fminf(1.0f, fmaxf(0.0f, food32[index] + delta));
                if (kind[i] == 2) toxin32[index] = fminf(1.0f, fmaxf(0.0f, toxin32[index] + delta));
            }
        }
        active[i] = 0;
    }
    """
    key = CommandKernelKey(
        source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        options=_KERNEL_OPTIONS,
        device_id=int(device.id),
        compute_capability=str(getattr(device, "compute_capability", "unknown")),
    )
    cached = _KERNEL_CACHE.get(key)
    if cached is not None:
        return cached
    module = cp.RawModule(code=source, options=_KERNEL_OPTIONS)
    cached = module.get_function("apply_commands")
    _KERNEL_CACHE[key] = cached
    _KERNEL_COMPILE_COUNT += 1
    return cached


def warm_command_kernel(cp: Any) -> None:
    _kernel(cp)


def command_kernel_compile_count() -> int:
    return int(_KERNEL_COMPILE_COUNT)


def apply_device_commands(ds: Any, buffer: DeviceCommandBuffer) -> None:
    xp = ds.xp
    if not ds.is_gpu:
        count = int(np.asarray(buffer.count)[0])
        for i in range(count):
            if not bool(buffer.active[i]):
                continue
            y, x = int(buffer.y[i]), int(buffer.x[i])
            if 0 <= y < ds.food.shape[0] and 0 <= x < ds.food.shape[1]:
                target = ds.food if int(buffer.command_type[i]) == 1 else ds.toxin
                target[y, x] = np.clip(
                    target[y, x] + float(buffer.amount[i]),
                    0.0,
                    1.0,
                )
            buffer.active[i] = 0
        return

    cp = xp
    capacity = buffer.capacity
    threads = 128
    blocks = (capacity + threads - 1) // threads
    use64 = int(ds.food.dtype == cp.float64)
    dummy64 = buffer.dummy64
    dummy32 = buffer.dummy32
    _kernel(cp)(
        (blocks,),
        (threads,),
        (
            buffer.command_type,
            buffer.y,
            buffer.x,
            buffer.amount,
            buffer.active,
            buffer.count,
            ds.food if use64 else dummy64,
            ds.toxin if use64 else dummy64,
            dummy32 if use64 else ds.food,
            dummy32 if use64 else ds.toxin,
            np.int32(ds.food.shape[0]),
            np.int32(ds.food.shape[1]),
            np.int32(use64),
        ),
    )
