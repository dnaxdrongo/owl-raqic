from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from owl.engine.loop import step as cpu_step


@dataclass
class ShadowParity:
    tick: int
    field_residuals: dict[str, float]
    exact_event_matches: dict[str, bool]
    worst_locations: dict[str, tuple[int, ...]]
    field_tolerances: dict[str, float]
    passed: bool
    field_absolute_tolerances: dict[str, float] = field(default_factory=dict)
    field_relative_tolerances: dict[str, float] = field(default_factory=dict)
    field_limits_at_worst: dict[str, float] = field(default_factory=dict)
    field_residual_ratios: dict[str, float] = field(default_factory=dict)
    left_values_at_worst: dict[str, float] = field(default_factory=dict)
    right_values_at_worst: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DeviceShadowSnapshot:
    """Exact device-state snapshot for a scheduled dense NumPy shadow tick.

    The ordinary ``WorldState`` mirror can use lower-precision storage and does
    not contain every device-only counter.  Exact execution parity therefore
    starts from the authoritative device arrays rather than a checkpoint
    round-trip.
    """

    state_template: Any
    arrays: dict[str, np.ndarray]
    patch_arrays: dict[str, np.ndarray]
    global_arrays: dict[str, np.ndarray]
    scalars: dict[str, Any]


_FLOAT_FIELDS = (
    "food",
    "toxin",
    "health",
    "resource",
    "memory",
    "phase",
    "integration",
    "possibility",
    "raqic_probabilities",
    "patch_integration",
    "patch_coherence",
    "patches.integration",
    "patches.coherence",
    "patches.synchrony",
    "patches.cross_scale",
    "patches.policy_bias",
    "global_state.integration",
    "global_state.fragmentation",
    "global_state.diversity",
    "global_state.complexity",
    "global_state.policy_bias",
    "global_state.intention_scores",
    "active_sense_food_memory",
    "active_sense_toxin_memory",
    "active_sense_alive_memory",
    "action_target_distance",
    "action_target_confidence",
    "action_direction_score",
    "action_direction_distance_delta",
    "action_direction_hazard",
    "action_direction_opportunity",
)

# Numerical floors are equation-specific, not a blanket shadow waiver. Phase
# is a recursively wrapped float32 oscillator, so independent CPU and promoted
# accelerated evaluations may accumulate several ulps over long trajectories
# while preserving all discrete readouts. Reduction fields likewise inherit
# float32 input error.
_FIELD_EPS_MULTIPLIERS = {
    "phase": 64.0,
    "resource": 32.0,
    "food": 32.0,
    "patches.synchrony": 32.0,
    "patches.coherence": 16.0,
    "patches.cross_scale": 16.0,
}

# RAQIC probabilities are stored and evaluated in audit64, but their scores are
# functions of the established float32 physical state. Independent NumPy and
# CuPy GEMM/exp/reduction kernels may therefore differ by approximately one
# source-state relative ULP without changing the probability simplex or any
# sampled/readout event. A combined absolute/relative criterion preserves the
# strict 1e-8 floor near zero while allowing at most one float32 epsilon of
# scale-relative drift away from zero.
_FIELD_RELATIVE_TOLERANCES = {
    "raqic_probabilities": float(np.finfo(np.float32).eps),
}
_CIRCULAR_FLOAT_FIELDS = {"phase"}


def _circular_difference(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.abs(np.arctan2(np.sin(left - right), np.cos(left - right)))


_EXACT_FIELDS = (
    "readout",
    "raqic_readout",
    "occupancy",
    "parent_id",
    "patches.intention",
    "global_state.intention",
    "global_state.readout",
    "active_sense_ttl",
    "active_sense_new_cell_count",
    "active_sense_new_target_count",
    "flee_compiled_action",
    "pursue_compiled_action",
    "compiled_execution_action",
    "action_target_y",
    "action_target_x",
    "action_target_ow_id",
    "action_target_kind",
    "action_target_source",
    "action_direction_y",
    "action_direction_x",
    "action_direction_executable",
)


class ShadowReference(StrEnum):
    IMPLEMENTATION_NUMPY = "implementation_numpy"
    SCIENTIFIC_CPU = "scientific_cpu"


class CPUShadowAuditor:
    def __init__(
        self,
        cfg: Any,
        *,
        ticks: tuple[int, ...],
        tolerance: float = 1e-8,
        strict: bool = True,
        output_dir: str | Path = "reports/cpu_gpu_shadow",
        reference_mode: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.ticks = {int(tick) for tick in ticks}
        self.tolerance = float(tolerance)
        self.strict = bool(strict)
        self.output_dir = Path(output_dir)
        raw_reference = str(
            reference_mode or getattr(cfg.raqic, "full_gpu_shadow_reference", "scientific_cpu")
        )
        aliases = {
            "dense_numpy_exact": ShadowReference.IMPLEMENTATION_NUMPY.value,
            "legacy_cpu_semantic": ShadowReference.SCIENTIFIC_CPU.value,
        }
        self.reference_mode = aliases.get(raw_reference, raw_reference)
        if self.reference_mode not in {
            ShadowReference.IMPLEMENTATION_NUMPY.value,
            ShadowReference.SCIENTIFIC_CPU.value,
        }:
            raise ValueError(f"unknown CPU shadow reference mode: {self.reference_mode}")
        self.reports: list[dict[str, Any]] = []

    def due(self, next_tick: int) -> bool:
        return int(next_tick) in self.ticks

    def prepare_cpu_state(self, state: Any) -> Any:
        return copy.deepcopy(state)

    def prepare_device_snapshot(self, ds: Any, state_template: Any) -> DeviceShadowSnapshot:
        """Copy the exact pre-tick device state at an explicit audit boundary."""
        return DeviceShadowSnapshot(
            state_template=copy.deepcopy(state_template),
            arrays={
                name: np.array(ds.backend.asnumpy(value), copy=True)
                for name, value in ds.arrays.items()
            },
            patch_arrays={
                name: np.array(ds.backend.asnumpy(value), copy=True)
                for name, value in ds.patch_arrays.items()
            },
            global_arrays={
                name: np.array(ds.backend.asnumpy(value), copy=True)
                for name, value in ds.global_arrays.items()
            },
            scalars=copy.deepcopy(ds.scalars),
        )

    @staticmethod
    def _restore_device_snapshot(run: Any, snapshot: DeviceShadowSnapshot) -> None:
        """Restore values into preallocated arrays without invalidating slabs."""
        for name, value in snapshot.arrays.items():
            if name in run.ds.arrays and run.ds.arrays[name].shape == value.shape:
                run.ds.arrays[name][...] = value
            else:
                run.ds.arrays[name] = np.array(value, copy=True)
        for name, value in snapshot.patch_arrays.items():
            if name in run.ds.patch_arrays and run.ds.patch_arrays[name].shape == value.shape:
                run.ds.patch_arrays[name][...] = value
            else:
                run.ds.patch_arrays[name] = np.array(value, copy=True)
        for name, value in snapshot.global_arrays.items():
            if name in run.ds.global_arrays and run.ds.global_arrays[name].shape == value.shape:
                run.ds.global_arrays[name][...] = value
            else:
                run.ds.global_arrays[name] = np.array(value, copy=True)
        run.ds.scalars.clear()
        run.ds.scalars.update(copy.deepcopy(snapshot.scalars))
        if run.slab_manager is not None:
            run.slab_manager.assert_views_current(run.ds)

    def run_cpu_reference(self, state: Any, *, tick: int) -> Any:
        if self.reference_mode == ShadowReference.SCIENTIFIC_CPU.value:
            cfg = self.cfg.model_copy(deep=True)
            cfg.raqic.mode = "cpu_audit"
            cfg.raqic.full_gpu_cpu_shadow_ticks = 0
            cfg.raqic.qiskit_decision_mode = "off"
            rng = np.random.default_rng(int(cfg.world.seed) + int(tick))
            cpu_step(state, cfg, rng)
            return state

        # Dense NumPy shadow: execute the same array-first tick program on CPU.
        # This is the exact execution-parity oracle for CuPy kernels. The
        # independent scalar/Kraus CPU path remains available through
        # ``scientific_cpu`` and the RAQIC baseline tests.
        from owl.gpu.run_context import PersistentOWLDeviceRun
        from owl.runtime.capabilities import RuntimeCapabilities
        from owl.runtime.execution_plan import compile_execution_plan

        cfg = self.cfg.model_copy(deep=True)
        cfg.raqic.mode = "gpu_full"
        cfg.raqic.full_gpu_strict = False
        cfg.raqic.strict_gpu = False
        cfg.raqic.fallback_on_backend_error = True
        cfg.raqic.full_gpu_no_silent_fallback = False
        cfg.raqic.full_gpu_transfer_policy = "persistent_mirror"
        cfg.raqic.full_gpu_execution_tier = "persistent"
        cfg.raqic.full_gpu_graph_mode = "off"
        cfg.raqic.full_gpu_graph_requirement = "allow_partial"
        cfg.raqic.full_gpu_cpu_shadow_ticks = 0
        cfg.raqic.qiskit_decision_mode = "off"
        cfg.raqic.gpu_validate_qiskit = False
        cfg.raqic.full_gpu_validation_every = 0
        cfg.raqic.full_gpu_visual_backend = "none"
        cfg.visualization.enabled = False
        cfg.recording.enabled = False
        runtime = RuntimeCapabilities(
            cupy_available=False,
            cuda_device_count=0,
            qiskit_available=False,
            aer_available=False,
            aer_gpu_available=False,
            pygame_available=False,
            vispy_available=False,
            nccl_available=False,
            details={"shadow": "forced_numpy"},
        )
        plan = compile_execution_plan(cfg, runtime)
        snapshot = state if isinstance(state, DeviceShadowSnapshot) else None
        initial_state = copy.deepcopy(snapshot.state_template) if snapshot is not None else state
        run = PersistentOWLDeviceRun.from_config(
            cfg,
            initial_state=initial_state,
            plan=plan,
            force_backend="numpy",
        )
        if snapshot is not None:
            self._restore_device_snapshot(run, snapshot)
        try:
            run.step()
            return copy.deepcopy(run.checkpoint(count=False))
        finally:
            run.close(checkpoint=False)

    def compare(self, cpu_state: Any, gpu_state: Any, *, tick: int) -> ShadowParity:
        residuals: dict[str, float] = {}
        exact: dict[str, bool] = {}
        worst: dict[str, tuple[int, ...]] = {}
        passed = True
        tolerances: dict[str, float] = {}
        absolute_tolerances: dict[str, float] = {}
        relative_tolerances: dict[str, float] = {}
        limits_at_worst: dict[str, float] = {}
        residual_ratios: dict[str, float] = {}
        left_values_at_worst: dict[str, float] = {}
        right_values_at_worst: dict[str, float] = {}

        def resolve(obj: Any, path: Any) -> Any:
            value = obj
            for part in path.split("."):
                if not hasattr(value, part):
                    return None
                value = getattr(value, part)
            return value

        for name in _FLOAT_FIELDS:
            left_value = resolve(cpu_state, name)
            right_value = resolve(gpu_state, name)
            if left_value is None or right_value is None:
                continue
            left = np.asarray(left_value)
            right = np.asarray(right_value)
            if left.shape != right.shape:
                residuals[name] = float("inf")
                passed = False
                continue
            left64 = left.astype(np.float64)
            right64 = right.astype(np.float64)
            diff = (
                _circular_difference(left64, right64)
                if name in _CIRCULAR_FLOAT_FIELDS
                else np.abs(left64 - right64)
            )
            value = float(np.nanmax(diff)) if diff.size else 0.0
            residuals[name] = value
            worst_index: tuple[int, ...] | None = None
            if diff.size:
                worst_index = tuple(
                    int(x) for x in np.unravel_index(int(np.nanargmax(diff)), diff.shape)
                )
                worst[name] = worst_index
            dtype_eps = 0.0
            if np.issubdtype(left.dtype, np.floating):
                dtype_eps = float(np.finfo(left.dtype).eps)
            if np.issubdtype(right.dtype, np.floating):
                dtype_eps = max(dtype_eps, float(np.finfo(right.dtype).eps))
            multiplier = float(_FIELD_EPS_MULTIPLIERS.get(name, 8.0))
            absolute_tolerance = max(self.tolerance, multiplier * dtype_eps)
            # Global/patch summaries are reductions of float32 cell state even
            # when represented as Python/float64 scalars after checkpoint.
            # Use the float32 state precision floor rather than pretending the
            # reduction inputs were native float64.
            if name.startswith(("global_state.", "patches.")):
                absolute_tolerance = max(absolute_tolerance, 8.0 * float(np.finfo(np.float32).eps))
            relative_tolerance = float(_FIELD_RELATIVE_TOLERANCES.get(name, 0.0))
            allowed = absolute_tolerance + relative_tolerance * np.abs(right64)
            absolute_tolerances[name] = absolute_tolerance
            relative_tolerances[name] = relative_tolerance
            tolerances[name] = float(np.nanmax(allowed)) if allowed.size else absolute_tolerance
            if worst_index is not None:
                limit_at_worst = float(allowed[worst_index])
                left_values_at_worst[name] = float(left64[worst_index])
                right_values_at_worst[name] = float(right64[worst_index])
            else:
                limit_at_worst = absolute_tolerance
            limits_at_worst[name] = limit_at_worst
            safe_allowed = np.maximum(allowed, np.finfo(np.float64).tiny)
            ratio = diff / safe_allowed
            residual_ratios[name] = float(np.nanmax(ratio)) if ratio.size else 0.0
            if (
                not np.isfinite(value)
                or not np.all(np.isfinite(allowed))
                or bool(np.any(diff > allowed))
            ):
                passed = False
        for name in _EXACT_FIELDS:
            left_value = resolve(cpu_state, name)
            right_value = resolve(gpu_state, name)
            if left_value is None or right_value is None:
                continue
            left = np.asarray(left_value)
            right = np.asarray(right_value)
            same = bool(left.shape == right.shape and np.array_equal(left, right))
            exact[name] = same
            if not same:
                passed = False
        return ShadowParity(
            tick=int(tick),
            field_residuals=residuals,
            exact_event_matches=exact,
            worst_locations=worst,
            field_tolerances=tolerances,
            passed=passed,
            field_absolute_tolerances=absolute_tolerances,
            field_relative_tolerances=relative_tolerances,
            field_limits_at_worst=limits_at_worst,
            field_residual_ratios=residual_ratios,
            left_values_at_worst=left_values_at_worst,
            right_values_at_worst=right_values_at_worst,
        )

    def record(self, parity: ShadowParity) -> dict[str, Any]:
        payload = parity.to_dict()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"cpu_gpu_shadow_tick_{parity.tick:08d}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self.reports.append(payload)
        if self.strict and not parity.passed:
            raise RuntimeError(
                f"CPU/GPU shadow parity failed at tick {parity.tick}; report: {path}"
            )
        return payload

    def summary(self) -> dict[str, Any]:
        return {
            "runs": len(self.reports),
            "passed": all(report["passed"] for report in self.reports) if self.reports else None,
            "ticks": sorted(self.ticks),
            "tolerance": self.tolerance,
            "reference_mode": self.reference_mode,
        }
