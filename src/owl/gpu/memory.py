from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FullGPUMemoryEstimate:
    height: int
    width: int
    actions: int
    channels: int
    bytes_estimated: int

    @property
    def mb(self) -> float:
        return self.bytes_estimated / (1024.0**2)


def estimate_full_gpu_memory(
    height: int,
    width: int,
    actions: int,
    channels: int,
    dtype_bytes: int = 8,
    scratch_factor: float = 3.0,
) -> FullGPUMemoryEstimate:
    cell = height * width
    action = cell * actions
    channel = cell * channels
    base_scalars = 48 * cell * dtype_bytes
    action_arrays = 8 * action * dtype_bytes
    channel_arrays = 6 * channel * dtype_bytes
    scratch = int((base_scalars + action_arrays + channel_arrays) * scratch_factor)
    return FullGPUMemoryEstimate(
        height,
        width,
        actions,
        channels,
        int(base_scalars + action_arrays + channel_arrays + scratch),
    )
