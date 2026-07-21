from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from typing import Any

import numpy as np

from owl.core.state import WorldState
from owl.gpu.backend import ArrayBackend, get_array_backend
from owl.gpu.field_registry import PATCH_FIELDS
from owl.raqic.precision import (
    RAQIC_AUDIT_REAL_FIELDS,
    precision_mode,
    raqic_backend_complex_dtype,
    raqic_backend_real_dtype,
)


def _is_array(value: Any) -> bool:
    return isinstance(value, np.ndarray) or value.__class__.__module__.startswith("cupy")


@dataclass
class OWLDeviceState:
    """GPU-resident mirror of WorldState dense arrays.

    The high-level scheduler can remain CPU Python, but this object keeps the
    large simulation fields in an array namespace (`numpy` fallback or `cupy`).
    """

    backend: ArrayBackend
    arrays: dict[str, Any] = field(default_factory=dict)
    patch_arrays: dict[str, Any] = field(default_factory=dict)
    global_arrays: dict[str, Any] = field(default_factory=dict)
    scalars: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def xp(self) -> Any:
        return self.backend.xp

    @property
    def is_gpu(self) -> bool:
        return bool(self.backend.is_gpu)

    @property
    def tick(self) -> int:
        return int(self.scalars.get("tick", 0))

    @tick.setter
    def tick(self, value: int) -> None:
        self.scalars["tick"] = int(value)
        if "_device_tick" in self.arrays:
            self.arrays["_device_tick"][...] = self.xp.asarray(int(value), dtype=self.xp.uint64)

    @classmethod
    def from_world_state(
        cls,
        state: WorldState,
        cfg: Any,
        *,
        strict: bool = False,
        allow_fallback: bool = True,
        force_backend: str | None = None,
    ) -> OWLDeviceState:
        backend = get_array_backend(
            strict=strict,
            allow_fallback=allow_fallback,
            force=force_backend,
        )
        ds = cls(backend=backend)
        ds.refresh_from_cpu(state)
        ds.apply_precision_policy(cfg)
        ds.metadata["cfg_mode"] = getattr(getattr(cfg, "raqic", None), "mode", "unknown")
        ds.metadata["backend_info"] = backend.info
        ds.metadata["cfg"] = cfg
        ds.metadata["defer_host_metrics"] = False
        return ds

    def apply_precision_policy(self, cfg: Any) -> None:
        """Apply the declared numerical precision without changing OWL physics.

        OWL's executable scientific reference stores ecological, phase, patch,
        and top-down state in float32.  The ``audit64`` contract applies to the
        dense RAQIC score/probability/instrument calculations, not to a silent
        promotion of every physical state array.  Promoting all fields changed
        phase-reduction trajectories and, more seriously, selected float64 CUDA
        kernels while the preallocated physical scratch buffers were float32.

        Keep physical state at its source dtype and promote only RAQIC numerical
        evidence that is explicitly evaluated under the audit64 tolerance.
        Dense decision workspaces are independently allocated at the configured
        precision by :mod:`owl_raqic.gpu`.
        """
        precision = precision_mode(cfg)
        xp = self.xp

        if precision in {"balanced32", "fast32"}:
            for mapping in (self.arrays, self.patch_arrays, self.global_arrays):
                for name, value in list(mapping.items()):
                    kind = getattr(value.dtype, "kind", "")
                    if kind == "f":
                        mapping[name] = value.astype(xp.float32, copy=False)
                    elif kind == "c":
                        mapping[name] = value.astype(xp.complex64, copy=False)
            self.metadata["precision_policy"] = "all_float32"
            self.metadata["precision_promoted_fields"] = []
            self.metadata["raqic_real_dtype"] = str(xp.dtype(xp.float32))
            return

        real_dtype = raqic_backend_real_dtype(cfg, xp)
        complex_dtype = raqic_backend_complex_dtype(cfg, xp)
        promoted: list[str] = []
        for mapping in (self.arrays, self.patch_arrays, self.global_arrays):
            for name, value in list(mapping.items()):
                kind = getattr(value.dtype, "kind", "")
                if name not in RAQIC_AUDIT_REAL_FIELDS or kind not in {"f", "c"}:
                    continue
                dtype = complex_dtype if kind == "c" else real_dtype
                mapping[name] = value.astype(dtype, copy=False)
                promoted.append(name)

        self.metadata["precision_policy"] = (
            "raqic_audit64_physical_source" if precision == "audit64" else "mixed"
        )
        self.metadata["precision_promoted_fields"] = sorted(set(promoted))
        self.metadata["raqic_real_dtype"] = str(xp.dtype(real_dtype))

    def refresh_from_cpu(self, state: WorldState, fields: list[str] | None = None) -> None:
        selected = set(fields) if fields is not None else None
        for f in dataclass_fields(state):
            name = f.name
            if selected is not None and name not in selected:
                continue
            value = getattr(state, name)
            if isinstance(value, np.ndarray):
                self.arrays[name] = self.backend.asarray(value)
        for name in PATCH_FIELDS:
            if hasattr(state.patches, name):
                value = getattr(state.patches, name)
                if isinstance(value, np.ndarray):
                    self.patch_arrays[name] = self.backend.asarray(value)
        # Store compact global arrays and scalars.
        for name in ("signal_pressure", "policy_bias", "intention_scores"):
            if hasattr(state.global_state, name):
                value = getattr(state.global_state, name)
                if isinstance(value, np.ndarray):
                    self.global_arrays[name] = self.backend.asarray(value)
        for name in ("tick", "next_ow_id", "global_crisis", "global_carrying_pressure"):
            if hasattr(state, name):
                self.scalars[name] = getattr(state, name)
        self.scalars["global_integration"] = float(state.global_state.integration)
        self.scalars["global_readout"] = int(state.global_state.readout)
        self.scalars["global_intention"] = int(state.global_state.intention)
        self.scalars["global_fragmentation"] = float(state.global_state.fragmentation)
        self.scalars["global_diversity"] = float(state.global_state.diversity)
        self.scalars["global_complexity"] = float(state.global_state.complexity)
        if "_device_tick" not in self.arrays:
            self.arrays["_device_tick"] = self.backend.asarray(
                np.asarray(int(state.tick), dtype=np.uint64)
            )
        else:
            self.arrays["_device_tick"][...] = self.xp.asarray(
                int(state.tick), dtype=self.xp.uint64
            )

    def write_back_to_cpu(self, state: WorldState, fields: list[str] | None = None) -> None:
        selected = set(fields) if fields is not None else None
        for name, value in self.arrays.items():
            if selected is not None and name not in selected:
                continue
            if hasattr(state, name):
                setattr(
                    state,
                    name,
                    self.backend.asnumpy(value).astype(getattr(state, name).dtype, copy=False)
                    if isinstance(getattr(state, name), np.ndarray)
                    else self.backend.asnumpy(value),
                )
        for name, value in self.patch_arrays.items():
            if hasattr(state.patches, name):
                current = getattr(state.patches, name)
                if isinstance(current, np.ndarray):
                    setattr(
                        state.patches,
                        name,
                        self.backend.asnumpy(value).astype(current.dtype, copy=False),
                    )
        for name, value in self.global_arrays.items():
            if hasattr(state.global_state, name):
                current = getattr(state.global_state, name)
                if isinstance(current, np.ndarray):
                    setattr(
                        state.global_state,
                        name,
                        self.backend.asnumpy(value).astype(current.dtype, copy=False),
                    )
        for name in ("tick", "next_ow_id", "global_crisis", "global_carrying_pressure"):
            if name in self.scalars and hasattr(state, name):
                setattr(state, name, self.scalars[name])
        if "global_integration" in self.scalars:
            state.global_state.integration = float(self.scalars["global_integration"])
        if "global_readout" in self.scalars:
            state.global_state.readout = int(self.scalars["global_readout"])
        if "global_intention" in self.scalars:
            state.global_state.intention = int(self.scalars["global_intention"])
        if "global_fragmentation" in self.scalars:
            state.global_state.fragmentation = float(self.scalars["global_fragmentation"])
        if "global_diversity" in self.scalars:
            state.global_state.diversity = float(self.scalars["global_diversity"])
        if "global_complexity" in self.scalars:
            state.global_state.complexity = float(self.scalars["global_complexity"])
        scalar_to_global = {
            "global_crisis": "crisis",
            "global_carrying_pressure": "carrying_pressure",
            "global_starvation_pressure": "starvation_pressure",
            "global_food_deficit": "food_deficit",
        }
        for scalar_name, field_name in scalar_to_global.items():
            if scalar_name in self.scalars and hasattr(state.global_state, field_name):
                setattr(state.global_state, field_name, float(self.scalars[scalar_name]))

        # Sparse Python event records are not dense GPU arrays, so they travel
        # through metadata and are restored when a CPU mirror/checkpoint is built.
        if hasattr(state, "event_queue"):
            state.event_queue = list(self.metadata.get("event_queue", []))

    def checkpoint_to_cpu(self, state: WorldState) -> None:
        self.write_back_to_cpu(state)

    def synchronize(self) -> None:
        self.backend.synchronize()

    def assert_shapes(self) -> None:
        h, w = self.arrays["health"].shape
        assert self.arrays["possibility"].shape[:2] == (h, w)
        assert self.arrays["signal"].shape[:2] == (h, w)

    def memory_estimate(self) -> dict[str, int]:
        total = 0
        for value in (
            list(self.arrays.values())
            + list(self.patch_arrays.values())
            + list(self.global_arrays.values())
        ):
            nbytes = int(getattr(value, "nbytes", 0))
            total += nbytes
        return {"tracked_array_bytes": total}

    def device_summary(self) -> dict[str, Any]:
        return {
            "backend": self.backend.name,
            "is_gpu": self.is_gpu,
            "arrays": sorted(self.arrays),
            "patch_arrays": sorted(self.patch_arrays),
            "memory": self.memory_estimate(),
            "backend_info": self.backend.info,
        }

    def __getattr__(self, name: str) -> Any:
        if "arrays" in self.__dict__ and name in self.__dict__["arrays"]:
            return self.__dict__["arrays"][name]
        if "patch_arrays" in self.__dict__ and name.startswith("patch_"):
            key = name.removeprefix("patch_")
            if key in self.__dict__["patch_arrays"]:
                return self.__dict__["patch_arrays"][key]
        raise AttributeError(name)
