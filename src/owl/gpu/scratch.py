from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ScratchSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str
    role: str = "scratch"


@dataclass
class ScratchManager:
    """Preallocated scratch buffers for persistent GPU/NumPy execution."""

    backend: Any
    specs: dict[str, ScratchSpec] = field(default_factory=dict)
    buffers: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def for_config(cls, backend: Any, cfg: Any) -> ScratchManager:
        h = int(cfg.world.height)
        w = int(cfg.world.width)
        a = 22
        c = int(cfg.communication.num_channels)
        ph = h // int(cfg.world.patch_size)
        pw = w // int(cfg.world.patch_size)
        probability_dtype = (
            "float64"
            if getattr(cfg.raqic, "full_gpu_precision", "audit64") in ("audit64", "mixed")
            else "float32"
        )
        # OWL physical fields are float32 in the reference state. Raw CUDA
        # kernels require output buffers with the exact same dtype as inputs.
        physical_dtype = "float32"
        specs = {
            "stencil_sum": ScratchSpec("stencil_sum", (h, w), physical_dtype, "stencil"),
            "local_alive": ScratchSpec("local_alive", (h, w), physical_dtype, "stencil"),
            "local_food": ScratchSpec("local_food", (h, w), physical_dtype, "stencil"),
            "local_toxin": ScratchSpec("local_toxin", (h, w), physical_dtype, "stencil"),
            "phase_sin_sum": ScratchSpec("phase_sin_sum", (h, w), physical_dtype, "stencil"),
            "phase_cos_sum": ScratchSpec("phase_cos_sum", (h, w), physical_dtype, "stencil"),
            "food_next": ScratchSpec("food_next", (h, w), physical_dtype, "environment"),
            "toxin_next": ScratchSpec("toxin_next", (h, w), physical_dtype, "environment"),
            "signal_next": ScratchSpec("signal_next", (h, w, c), physical_dtype, "communication"),
            "authority": ScratchSpec("authority", (h, w, a), "bool", "authority"),
            "policy_logits": ScratchSpec("policy_logits", (h * w, a), probability_dtype, "policy"),
            "policy_probs": ScratchSpec("policy_probs", (h * w, a), probability_dtype, "policy"),
            "movement_source": ScratchSpec("movement_source", (h * w,), "int64", "movement"),
            "movement_target": ScratchSpec("movement_target", (h * w,), "int64", "movement"),
            "event_type": ScratchSpec(
                "event_type", (int(cfg.raqic.full_gpu_sparse_event_capacity),), "int32", "topology"
            ),
            "event_yx": ScratchSpec(
                "event_yx", (int(cfg.raqic.full_gpu_sparse_event_capacity), 2), "int32", "topology"
            ),
            "patch_summary": ScratchSpec("patch_summary", (ph, pw), probability_dtype, "patch"),
            "rgba_frame": ScratchSpec("rgba_frame", (h, w, 4), "uint8", "visual"),
        }
        return cls(backend=backend, specs=specs)

    def allocate_all(self) -> None:
        xp = self.backend.xp
        dtype_map = {
            "float64": xp.float64,
            "float32": xp.float32,
            "int64": xp.int64,
            "int32": xp.int32,
            "uint8": xp.uint8,
            "bool": bool,
        }
        for name, spec in self.specs.items():
            self.buffers[name] = xp.zeros(spec.shape, dtype=dtype_map[spec.dtype])

    def get(self, name: str) -> Any:
        if name not in self.buffers:
            spec = self.specs[name]
            xp = self.backend.xp
            dtype = getattr(xp, spec.dtype) if spec.dtype != "bool" else bool
            self.buffers[name] = xp.zeros(spec.shape, dtype=dtype)
        return self.buffers[name]

    def memory_bytes(self) -> int:
        return int(sum(getattr(buf, "nbytes", 0) for buf in self.buffers.values()))

    def spec_bytes(self) -> int:
        size = {"float64": 8, "float32": 4, "int64": 8, "int32": 4, "uint8": 1, "bool": 1}
        return int(
            sum(
                int(__import__("math").prod(spec.shape)) * size[spec.dtype]
                for spec in self.specs.values()
            )
        )
