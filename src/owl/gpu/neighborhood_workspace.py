"""Epoch-scoped reusable neighborhood buffers for reference and optimized stencils."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from owl.gpu.stencil import central_gradient, neighbor_sum_8


@dataclass
class NeighborhoodWorkspace:
    xp: Any
    boundary_mode: str
    source_epoch: int = -1
    buffers: dict[str, Any] = field(default_factory=dict)

    def begin_epoch(self, epoch: int, boundary_mode: str | None = None) -> None:
        if boundary_mode is not None and boundary_mode != self.boundary_mode:
            self.boundary_mode = str(boundary_mode)
            self.buffers.clear()
        if int(epoch) != self.source_epoch:
            self.source_epoch = int(epoch)
            self.buffers.clear()

    def invalidate(self) -> None:
        self.source_epoch = -1
        self.buffers.clear()

    def _output_like(self, name: str, source: Any) -> Any:
        output = self.buffers.get(name)
        if output is None or output.shape != source.shape or output.dtype != source.dtype:
            output = self.xp.empty_like(source)
            self.buffers[name] = output
        return output

    def mean8(self, name: str, source: Any) -> Any:
        output = self._output_like(f"mean8:{name}", source)
        output[...] = neighbor_sum_8(source, self.xp, self.boundary_mode) / 8.0
        return output

    def gradient(self, name: str, source: Any) -> tuple[Any, Any]:
        gradient_y = self._output_like(f"gradient_y:{name}", source)
        gradient_x = self._output_like(f"gradient_x:{name}", source)
        reference_y, reference_x = central_gradient(source, self.xp, self.boundary_mode)
        gradient_y[...] = reference_y
        gradient_x[...] = reference_x
        return gradient_y, gradient_x

    @property
    def allocated_bytes(self) -> int:
        return sum(int(buffer.nbytes) for buffer in self.buffers.values())
