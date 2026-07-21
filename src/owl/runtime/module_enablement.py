from __future__ import annotations

from dataclasses import dataclass
from typing import Any

KNOWN_MODULES = frozenset(
    {
        "environment",
        "sensing",
        "utility",
        "authority",
        "movement",
        "collision",
        "feeding",
        "health",
        "communication",
        "memory",
        "phase",
        "integration",
        "aggregation",
        "topdown",
        "reproduction",
        "death",
        "topology",
        "recording",
        "visualization",
    }
)

_REQUIRED_FOR_RAQIC = frozenset(
    {
        "environment",
        "sensing",
        "utility",
        "authority",
        "phase",
        "integration",
        "aggregation",
        "topdown",
    }
)


@dataclass(frozen=True)
class ModuleEnablement:
    enabled: frozenset[str]

    @classmethod
    def from_config(cls, cfg: Any) -> ModuleEnablement:
        values = frozenset(str(x) for x in getattr(cfg.raqic, "full_gpu_physical_modules", ()))
        unknown = values - KNOWN_MODULES
        if unknown:
            raise ValueError("unknown full_gpu_physical_modules: " + ", ".join(sorted(unknown)))
        if getattr(cfg.raqic, "enabled", False):
            missing = _REQUIRED_FOR_RAQIC - values
            if missing:
                raise ValueError(
                    "RAQIC full-GPU execution requires modules: " + ", ".join(sorted(missing))
                )
        return cls(values)

    def has(self, name: str) -> bool:
        if name not in KNOWN_MODULES:
            raise KeyError(name)
        return name in self.enabled

    def require(self, name: str) -> None:
        if not self.has(name):
            raise RuntimeError(f"GPU physical module {name!r} is disabled")
