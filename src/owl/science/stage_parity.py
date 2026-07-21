from __future__ import annotations

import copy
import json
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FieldComparison:
    field: str
    exact: bool
    max_abs: float | None
    mean_abs: float | None
    max_rel: float | None
    worst_index: tuple[int, ...] | None
    passed: bool
    left_shape: tuple[int, ...]
    right_shape: tuple[int, ...]
    missing_side: str | None = None
    left_dtype: str | None = None
    right_dtype: str | None = None
    left_value_at_worst: Any | None = None
    right_value_at_worst: Any | None = None
    left_support_at_worst: float | None = None
    right_support_at_worst: float | None = None


@dataclass(frozen=True)
class StateTraceSnapshot:
    stage: str
    state_hash: str
    values: dict[str, np.ndarray]
    events: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class StageParityReport:
    stage: str
    input_hash: str
    cpu_outputs: dict[str, str]
    array_outputs: dict[str, str]
    comparisons: tuple[FieldComparison, ...]
    event_comparison: dict[str, Any]
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["comparisons"] = [asdict(item) for item in self.comparisons]
        return value


@dataclass(frozen=True)
class StageDifferentialCertificate:
    backend: str
    stages_expected: tuple[str, ...]
    stages_cpu: tuple[str, ...]
    stages_array: tuple[str, ...]
    reports: tuple[StageParityReport, ...]
    first_divergent_stage: str | None
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "stages_expected": list(self.stages_expected),
            "stages_cpu": list(self.stages_cpu),
            "stages_array": list(self.stages_array),
            "reports": [item.to_dict() for item in self.reports],
            "first_divergent_stage": self.first_divergent_stage,
            "passed": self.passed,
        }


_EXACT_FIELD_SUFFIXES = {
    "tick",
    "next_ow_id",
    "occupancy",
    "parent_id",
    "readout",
    "raqic_readout",
    "raqic_shadow_readout",
    "obstacle",
    "intention",
    "global_readout",
    "global_intention",
    "_authority_bool",
    "last_movement_action",
}
_IGNORE_FIELDS = {
    # Sparse topology queues are compared through normalized event records.
    "event_queue",
    "ow_records",
    "pre_utilities",
    "pre_authority",
    "pre_parent_bias",
    # Backend provenance is expected to differ between CPU and GPU and is not
    # part of the scientific trajectory. It remains available in raw reports.
    "raqic_backend_code",
}


def array_hash(value: Any) -> str:
    arr = np.ascontiguousarray(np.asarray(value))
    h = sha256()
    h.update(str(arr.dtype).encode())
    h.update(str(arr.shape).encode())
    h.update(arr.view(np.uint8).tobytes())
    return h.hexdigest()


def _scalar_array(value: Any) -> np.ndarray | None:
    if isinstance(value, (bool, int, float, np.bool_, np.integer, np.floating)):
        return np.asarray(value)
    return None


def flatten_state(state: Any, *, extras: Mapping[str, Any] | None = None) -> dict[str, np.ndarray]:
    """Flatten scientific state into stable path->NumPy values.

    Validation uses this representation for both the independent CPU object and
    a CPU mirror copied from the device state.  Runtime-only metadata, queues,
    and opaque objects are intentionally excluded.
    """
    out: dict[str, np.ndarray] = {}

    def visit(value: Any, prefix: str) -> None:
        if prefix.rsplit(".", 1)[-1] in _IGNORE_FIELDS:
            return
        if isinstance(value, np.ndarray):
            out[prefix] = np.asarray(value).copy()
            return
        scalar = _scalar_array(value)
        if scalar is not None:
            out[prefix] = scalar.copy()
            return
        if is_dataclass(value):
            for item in fields(value):
                name = f"{prefix}.{item.name}" if prefix else item.name
                visit(getattr(value, item.name), name)

    visit(state, "")
    # Remove an empty prefix if a scalar root was ever supplied.
    out.pop("", None)
    for name, value in (extras or {}).items():
        if value is None:
            continue
        out[f"trace.{name}"] = np.asarray(value).copy()
    return out


def normalize_events(state: Any) -> tuple[dict[str, Any], ...]:
    records = []
    for event in list(getattr(state, "event_queue", ()) or ()):
        if is_dataclass(event) and not isinstance(event, type):
            payload = asdict(event)
        elif isinstance(event, Mapping):
            payload = dict(event)
        else:
            payload = {"repr": repr(event)}
        # JSON normalization gives deterministic tuple/list representation and
        # ordering for nested payload dictionaries.
        payload = json.loads(json.dumps(payload, sort_keys=True, default=str))
        records.append(payload)
    return tuple(records)


def _event_values_close(left: Any, right: Any, *, atol: float, rtol: float) -> bool:
    """Recursively compare normalized event values under scientific semantics."""
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return False
        return all(_event_values_close(left[key], right[key], atol=atol, rtol=rtol) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _event_values_close(a, b, atol=atol, rtol=rtol)
            for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, (float, np.floating)) or isinstance(right, (float, np.floating)):
        try:
            return bool(np.isclose(float(left), float(right), atol=atol, rtol=rtol))
        except (TypeError, ValueError):
            return False
    if isinstance(left, (int, np.integer)) or isinstance(right, (int, np.integer)):
        return type(left) is type(right) and int(left) == int(right)
    return bool(left == right)


def _compare_events(
    cpu_events: tuple[dict[str, Any], ...],
    array_events: tuple[dict[str, Any], ...],
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    """Compare ordered sparse events exactly except for floating payload values."""
    count_match = len(cpu_events) == len(array_events)
    mismatch_index: int | None = None
    for index, (left, right) in enumerate(zip(cpu_events, array_events, strict=False)):
        if not _event_values_close(left, right, atol=atol, rtol=rtol):
            mismatch_index = index
            break
    if mismatch_index is None and not count_match:
        mismatch_index = min(len(cpu_events), len(array_events))
    passed = count_match and mismatch_index is None
    return {
        "passed": passed,
        "cpu_count": len(cpu_events),
        "array_count": len(array_events),
        "first_mismatch_index": mismatch_index,
        "first_cpu_only": (
            cpu_events[mismatch_index]
            if mismatch_index is not None and mismatch_index < len(cpu_events)
            else None
        ),
        "first_array_only": (
            array_events[mismatch_index]
            if mismatch_index is not None and mismatch_index < len(array_events)
            else None
        ),
        "float_atol": float(atol),
        "float_rtol": float(rtol),
    }


def snapshot_state(
    stage: str, state: Any, *, extras: Mapping[str, Any] | None = None
) -> StateTraceSnapshot:
    values = flatten_state(state, extras=extras)
    h = sha256()
    for name in sorted(values):
        h.update(name.encode())
        h.update(array_hash(values[name]).encode())
    events = normalize_events(state)
    h.update(json.dumps(events, sort_keys=True).encode())
    return StateTraceSnapshot(stage=stage, state_hash=h.hexdigest(), values=values, events=events)


class ScientificTraceCollector:
    """Ordered validation-only stage snapshot collector."""

    def __init__(self, *, backend: str) -> None:
        self.backend = str(backend)
        self.snapshots: list[StateTraceSnapshot] = []
        self._seen: set[str] = set()

    def capture(self, stage: str, state: Any, *, extras: Mapping[str, Any] | None = None) -> None:
        if stage in self._seen:
            raise RuntimeError(f"scientific stage {stage!r} captured more than once in one tick")
        self.snapshots.append(snapshot_state(stage, state, extras=extras))
        self._seen.add(stage)

    @property
    def stages(self) -> tuple[str, ...]:
        return tuple(item.stage for item in self.snapshots)

    def by_stage(self) -> dict[str, StateTraceSnapshot]:
        return {item.stage: item for item in self.snapshots}


_CIRCULAR_FIELDS = {
    "phase",
    "patches.phase",
    "raqic_phase",
    "trace._parent_phase",
}


def _is_circular_field(name: str) -> bool:
    return name in _CIRCULAR_FIELDS


def _circular_delta(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return principal signed angular displacement in radians."""
    return np.asarray(np.arctan2(np.sin(left - right), np.cos(left - right)))


def _json_value_at(array: np.ndarray, index: tuple[int, ...] | None) -> Any | None:
    if index is None or not array.size:
        return None
    value = np.asarray(array[index])
    if value.ndim == 0:
        return value.item()
    return value.tolist()


def compare_field(
    name: str, left: Any, right: Any, *, exact: bool, atol: float, rtol: float
) -> FieldComparison:
    a = np.asarray(left)
    b = np.asarray(right)
    left_shape = tuple(int(v) for v in a.shape)
    right_shape = tuple(int(v) for v in b.shape)
    left_dtype = str(a.dtype)
    right_dtype = str(b.dtype)
    if a.shape != b.shape:
        return FieldComparison(
            field=name,
            exact=exact,
            max_abs=None,
            mean_abs=None,
            max_rel=None,
            worst_index=None,
            passed=False,
            left_shape=left_shape,
            right_shape=right_shape,
            left_dtype=left_dtype,
            right_dtype=right_dtype,
        )
    if exact:
        passed = bool(np.array_equal(a, b))
        worst = None
        if not passed and a.size:
            worst = tuple(int(v) for v in np.argwhere(a != b)[0])
        return FieldComparison(
            field=name,
            exact=exact,
            max_abs=0.0 if passed else None,
            mean_abs=0.0 if passed else None,
            max_rel=0.0 if passed else None,
            worst_index=worst,
            passed=passed,
            left_shape=left_shape,
            right_shape=right_shape,
            left_dtype=left_dtype,
            right_dtype=right_dtype,
            left_value_at_worst=_json_value_at(a, worst),
            right_value_at_worst=_json_value_at(b, worst),
        )
    af = a.astype(np.float64, copy=False)
    bf = b.astype(np.float64, copy=False)
    if _is_circular_field(name):
        diff = np.abs(_circular_delta(af, bf))
        rel = diff / np.pi
        finite = np.isfinite(diff) & np.isfinite(rel)
        passed = bool(np.all(finite & ((diff <= atol) | (rel <= rtol))))
    else:
        diff = np.abs(af - bf)
        denom = np.maximum(np.abs(af), np.abs(bf))
        rel = diff / np.maximum(denom, np.finfo(np.float64).tiny)
        passed = bool(np.allclose(af, bf, atol=atol, rtol=rtol, equal_nan=False))
    max_abs = float(np.nanmax(diff)) if diff.size else 0.0
    mean_abs = float(np.nanmean(diff)) if diff.size else 0.0
    max_rel = float(np.nanmax(rel)) if rel.size else 0.0
    worst = (
        tuple(int(v) for v in np.unravel_index(int(np.nanargmax(diff)), diff.shape))
        if diff.size
        else None
    )
    return FieldComparison(
        field=name,
        exact=exact,
        max_abs=max_abs,
        mean_abs=mean_abs,
        max_rel=max_rel,
        worst_index=worst,
        passed=passed,
        left_shape=left_shape,
        right_shape=right_shape,
        left_dtype=left_dtype,
        right_dtype=right_dtype,
        left_value_at_worst=_json_value_at(a, worst),
        right_value_at_worst=_json_value_at(b, worst),
    )


def _is_exact(name: str, a: np.ndarray, b: np.ndarray) -> bool:
    suffix = name.rsplit(".", 1)[-1]
    return suffix in _EXACT_FIELD_SUFFIXES or a.dtype.kind in "biu" or b.dtype.kind in "biu"


def compare_snapshots(
    cpu: StateTraceSnapshot,
    array: StateTraceSnapshot,
    *,
    input_hash: str,
    atol: float = 1e-5,
    rtol: float = 1e-6,
    strict_key_sets: bool = True,
) -> StageParityReport:
    comparisons: list[FieldComparison] = []
    cpu_keys = set(cpu.values)
    array_keys = set(array.values)
    for name in sorted(cpu_keys | array_keys):
        if name not in cpu.values:
            comparisons.append(
                FieldComparison(
                    name,
                    False,
                    None,
                    None,
                    None,
                    None,
                    False,
                    (),
                    tuple(array.values[name].shape),
                    "cpu",
                )
            )
            continue
        if name not in array.values:
            comparisons.append(
                FieldComparison(
                    name,
                    False,
                    None,
                    None,
                    None,
                    None,
                    False,
                    tuple(cpu.values[name].shape),
                    (),
                    "array",
                )
            )
            continue
        a, b = cpu.values[name], array.values[name]
        comparison = compare_field(name, a, b, exact=_is_exact(name, a, b), atol=atol, rtol=rtol)
        if name == "patches.phase" and comparison.worst_index is not None:
            left_support = cpu.values.get("patches.synchrony")
            right_support = array.values.get("patches.synchrony")
            index = comparison.worst_index
            comparison = replace(
                comparison,
                left_support_at_worst=(
                    float(np.asarray(left_support)[index]) if left_support is not None else None
                ),
                right_support_at_worst=(
                    float(np.asarray(right_support)[index]) if right_support is not None else None
                ),
            )
        comparisons.append(comparison)
    if not strict_key_sets:
        comparisons = [
            item
            for item in comparisons
            if item.missing_side is None or item.field.startswith("trace.")
        ]
    event_comparison = _compare_events(cpu.events, array.events, atol=atol, rtol=rtol)
    event_passed = bool(event_comparison["passed"])
    passed = all(item.passed for item in comparisons) and event_passed
    return StageParityReport(
        stage=cpu.stage,
        input_hash=input_hash,
        cpu_outputs={name: array_hash(value) for name, value in cpu.values.items()},
        array_outputs={name: array_hash(value) for name, value in array.values.items()},
        comparisons=tuple(comparisons),
        event_comparison=event_comparison,
        passed=passed,
    )


def _filtered_snapshot(snapshot: StateTraceSnapshot, writes: Iterable[str]) -> StateTraceSnapshot:
    wanted = tuple(writes)
    broad_cell = any(name in {"cell_fields", "bounded_fields"} for name in wanted)
    values: dict[str, np.ndarray] = {}
    for key, value in snapshot.values.items():
        if key.startswith("trace."):
            values[key] = value
            continue
        if broad_cell and not key.startswith(("patches.", "global_state.")):
            values[key] = value
            continue
        for name in wanted:
            if name == "patches" and key.startswith("patches."):
                values[key] = value
                break
            if name == "global_state" and key.startswith("global_state."):
                values[key] = value
                break
            if key == name or key.endswith("." + name):
                values[key] = value
                break
    return StateTraceSnapshot(snapshot.stage, snapshot.state_hash, values, snapshot.events)


def compare_stage_collectors(
    cpu: ScientificTraceCollector,
    array: ScientificTraceCollector,
    expected_stages: Iterable[str],
    *,
    initial_hash: str,
    atol: float = 1e-5,
    rtol: float = 1e-6,
) -> StageDifferentialCertificate:
    expected = tuple(expected_stages)
    cpu_map = cpu.by_stage()
    array_map = array.by_stage()
    from owl.science.stage_contract import STAGE_CONTRACTS

    writes_by_stage = {item.name: item.writes for item in STAGE_CONTRACTS}
    reports: list[StageParityReport] = []
    previous_hash = initial_hash
    first_divergent: str | None = None
    for stage in expected:
        left = cpu_map.get(stage)
        right = array_map.get(stage)
        if left is None or right is None:
            empty_left = left or StateTraceSnapshot(stage, "", {}, ())
            empty_right = right or StateTraceSnapshot(stage, "", {}, ())
            report = compare_snapshots(
                empty_left,
                empty_right,
                input_hash=previous_hash,
                atol=atol,
                rtol=rtol,
                strict_key_sets=False,
            )
        else:
            left_filtered = _filtered_snapshot(left, writes_by_stage.get(stage, ()))
            right_filtered = _filtered_snapshot(right, writes_by_stage.get(stage, ()))
            report = compare_snapshots(
                left_filtered,
                right_filtered,
                input_hash=previous_hash,
                atol=atol,
                rtol=rtol,
                strict_key_sets=False,
            )
        reports.append(report)
        if not report.passed and first_divergent is None:
            first_divergent = stage
        if left is not None:
            previous_hash = left.state_hash
    passed = first_divergent is None and cpu.stages == expected and array.stages == expected
    return StageDifferentialCertificate(
        backend=array.backend,
        stages_expected=expected,
        stages_cpu=cpu.stages,
        stages_array=array.stages,
        reports=tuple(reports),
        first_divergent_stage=first_divergent,
        passed=passed,
    )


def compare_state_fields(
    cpu_state: Any,
    array_state: Any,
    fields: Iterable[str],
    *,
    exact_fields: Iterable[str] = (),
    atol: float = 1e-10,
    rtol: float = 1e-10,
) -> tuple[FieldComparison, ...]:
    exact_set = set(exact_fields)
    out = []
    for name in fields:
        if not hasattr(cpu_state, name) or not hasattr(array_state, name):
            out.append(
                FieldComparison(name, name in exact_set, None, None, None, None, False, (), ())
            )
            continue
        out.append(
            compare_field(
                name,
                getattr(cpu_state, name),
                getattr(array_state, name),
                exact=name in exact_set,
                atol=atol,
                rtol=rtol,
            )
        )
    return tuple(out)


def write_report(
    report: StageParityReport | StageDifferentialCertificate, path: str | Path
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


@contextmanager
def _patched(module: Any, name: str, replacement: Callable[..., Any]) -> Iterator[None]:
    original = getattr(module, name)
    setattr(module, name, replacement)
    try:
        yield
    finally:
        setattr(module, name, original)


def _cpu_state_from_args(args: tuple[Any, ...], kwargs: Mapping[str, Any] | None = None) -> Any:
    for value in args:
        if value.__class__.__name__ == "WorldState":
            return value
    for value in (kwargs or {}).values():
        if value.__class__.__name__ == "WorldState":
            return value
    raise RuntimeError("instrumented CPU stage did not receive WorldState")


@contextmanager
def instrument_cpu_tick(collector: ScientificTraceCollector) -> Iterator[None]:
    """Instrument the public CPU composition root without changing runtime code."""
    import owl.engine.loop as loop

    phase_outputs: dict[str, Any] = {}

    def simple(name: str, stage: str, *, extras_from_result: str | None = None) -> Any:
        original = getattr(loop, name)

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            state = _cpu_state_from_args(args, kwargs)
            extras = {extras_from_result: result} if extras_from_result else None
            collector.capture(stage, state, extras=extras)
            return result

        return wrapped

    def parent_wrapper(*args: Any, **kwargs: Any) -> Any:
        result = parent_original(*args, **kwargs)
        state = _cpu_state_from_args(args, kwargs)
        collector.capture(
            "parent_context", state, extras={"_parent_bias": result[0], "_parent_phase": result[1]}
        )
        return result

    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        result = sync_original(*args, **kwargs)
        phase_outputs["_synchrony_current"] = result
        return result

    def coherence_wrapper(*args: Any, **kwargs: Any) -> Any:
        result = coherence_original(*args, **kwargs)
        phase_outputs["_coherence_current"] = result
        return result

    def cross_wrapper(*args: Any, **kwargs: Any) -> Any:
        result = cross_original(*args, **kwargs)
        phase_outputs["_cross_scale_current"] = result
        state = _cpu_state_from_args(args, kwargs)
        collector.capture("phase", state, extras={**phase_outputs, "_cross_scale_current": result})
        return result

    def post_wrapper(*args: Any, **kwargs: Any) -> Any:
        result = post_original(*args, **kwargs)
        state = _cpu_state_from_args(args, kwargs)
        collector.capture("aggregation", state)
        collector.capture("topdown_dispatch", state)
        return result

    parent_original = loop._ensure_parent_context
    sync_original = loop.compute_local_synchrony
    coherence_original = loop.compute_cell_coherence
    cross_original = loop.compute_cross_scale_coupling
    post_original = loop._post_state_refresh

    replacements = {
        "update_environment": simple("update_environment", "environment"),
        "compute_signal_reception": simple("compute_signal_reception", "sensing"),
        "_ensure_parent_context": parent_wrapper,
        "compute_local_synchrony": sync_wrapper,
        "compute_cell_coherence": coherence_wrapper,
        "compute_cross_scale_coupling": cross_wrapper,
        "compute_utilities": simple(
            "compute_utilities", "utility", extras_from_result="pre_utilities"
        ),
        "compute_authority": simple(
            "compute_authority", "authority", extras_from_result="pre_authority"
        ),
        "apply_decision_policy": simple("apply_decision_policy", "decision"),
        "apply_movement": simple("apply_movement", "movement"),
        "resolve_collisions": simple("resolve_collisions", "collision"),
        "apply_inhibition": simple("apply_inhibition", "inhibition"),
        "apply_feeding": simple("apply_feeding", "feeding"),
        "apply_repair_and_integrate": simple("apply_repair_and_integrate", "health_actions"),
        "emit_signals": simple("emit_signals", "communication_emit"),
        "apply_reproduction": simple("apply_reproduction", "reproduction"),
        "apply_topology_events": simple("apply_topology_events", "topology"),
        "apply_metabolism_damage": simple("apply_metabolism_damage", "metabolism"),
        "update_memory": simple("update_memory", "memory"),
        "update_signal_memory": simple("update_signal_memory", "signal_memory"),
        "update_integration": simple("update_integration", "integration"),
        "update_channel_trust": simple("update_channel_trust", "trust"),
        "apply_death": simple("apply_death", "death"),
        "clip_life_fields": simple("clip_life_fields", "clip"),
        "_post_state_refresh": post_wrapper,
    }
    with ExitStack() as stack:
        for name, replacement in replacements.items():
            stack.enter_context(_patched(loop, name, replacement))
        yield


def _device_snapshot(run: Any, stage: str, extras: Mapping[str, Any] | None = None) -> None:
    run.streams.compute.synchronize()
    mirror = copy.deepcopy(run.state)
    run.ds.write_back_to_cpu(mirror)
    converted: dict[str, Any] = {}
    for name, value in (extras or {}).items():
        converted[name] = run.ds.backend.asnumpy(value) if hasattr(value, "shape") else value
    run._scientific_trace_collector.capture(stage, mirror, extras=converted)


@contextmanager
def instrument_device_tick(run: Any, collector: ScientificTraceCollector) -> Iterator[None]:
    """Instrument eager NumPy/CuPy device stages for differential validation.

    Graph capture is intentionally rejected: synchronous snapshots and Python
    callbacks are validation operations and are forbidden inside a captured
    replay region.
    """
    import owl.gpu.run_context as rc

    if str(run.graph_manager.mode) != "off":
        raise RuntimeError("stage differential tracing requires graph mode off")
    run._scientific_trace_collector = collector
    previous_scientific_flag = run.ds.metadata.get("scientific_stage_parity")
    run.ds.metadata["scientific_stage_parity"] = True

    phase_outputs: dict[str, Any] = {}

    def simple(name: str, stage: str, *, extras_names: tuple[str, ...] = ()) -> Any:
        original = getattr(rc, name)

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            extras = {item: run.ds.arrays.get(item) for item in extras_names}
            _device_snapshot(run, stage, extras)
            return result

        return wrapped

    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        result = sync_original(*args, **kwargs)
        phase_outputs["_synchrony_current"] = result
        return result

    def coherence_wrapper(*args: Any, **kwargs: Any) -> Any:
        result = coherence_original(*args, **kwargs)
        phase_outputs["_coherence_current"] = result
        return result

    def cross_wrapper(*args: Any, **kwargs: Any) -> Any:
        result = cross_original(*args, **kwargs)
        phase_outputs["_cross_scale_current"] = result
        _device_snapshot(run, "phase", phase_outputs)
        return result

    def threshold_wrapper(*args: Any, **kwargs: Any) -> Any:
        result = threshold_original(*args, **kwargs)
        _device_snapshot(
            run,
            "parent_context",
            {
                "_parent_bias": run.ds.arrays.get("pre_parent_bias"),
                "_parent_phase": run.ds.arrays.get("_parent_phase"),
            },
        )
        return result

    def aggregate_global_wrapper(*args: Any, **kwargs: Any) -> Any:
        return aggregate_global_original(*args, **kwargs)

    def dispatch_wrapper(*args: Any, **kwargs: Any) -> Any:
        result = dispatch_original(*args, **kwargs)
        if bool(kwargs.get("force_global", False)):
            # CPU post-refresh performs aggregation and top-down policy update
            # in one monolithic helper. Capture the same completed epoch for
            # both contract labels until that helper is split.
            _device_snapshot(run, "aggregation")
            _device_snapshot(run, "topdown_dispatch")
        return result

    sync_original = rc.compute_local_synchrony_gpu
    coherence_original = rc.compute_cell_coherence_gpu
    cross_original = rc.compute_cross_scale_coupling_gpu
    threshold_original = rc.apply_threshold_modulation_gpu
    aggregate_global_original = rc.aggregate_global_gpu
    dispatch_original = rc.dispatch_parent_context_gpu

    replacements = {
        "update_environment_gpu": simple("update_environment_gpu", "environment"),
        "compute_sensing_bundle_gpu": simple("compute_sensing_bundle_gpu", "sensing"),
        "apply_threshold_modulation_gpu": threshold_wrapper,
        "compute_local_synchrony_gpu": sync_wrapper,
        "compute_cell_coherence_gpu": coherence_wrapper,
        "compute_cross_scale_coupling_gpu": cross_wrapper,
        "compute_utilities_gpu": simple(
            "compute_utilities_gpu", "utility", extras_names=("pre_utilities",)
        ),
        "compute_authority_gpu": simple(
            "compute_authority_gpu", "authority", extras_names=("pre_authority",)
        ),
        "run_raqic_gpu_stage": simple("run_raqic_gpu_stage", "decision"),
        "apply_movement_gpu": simple("apply_movement_gpu", "movement"),
        "resolve_collisions_gpu": simple("resolve_collisions_gpu", "collision"),
        "apply_inhibition_gpu": simple("apply_inhibition_gpu", "inhibition"),
        "apply_feeding_gpu": simple("apply_feeding_gpu", "feeding"),
        "apply_repair_and_integrate_gpu": simple(
            "apply_repair_and_integrate_gpu", "health_actions"
        ),
        "emit_signals_gpu": simple("emit_signals_gpu", "communication_emit"),
        "apply_reproduction_gpu": simple("apply_reproduction_gpu", "reproduction"),
        "apply_topology_events_gpu": simple("apply_topology_events_gpu", "topology"),
        "apply_metabolism_damage_gpu": simple("apply_metabolism_damage_gpu", "metabolism"),
        "update_memory_gpu": simple("update_memory_gpu", "memory"),
        "update_signal_memory_gpu": simple("update_signal_memory_gpu", "signal_memory"),
        "update_integration_gpu": simple("update_integration_gpu", "integration"),
        "update_channel_trust_gpu": simple("update_channel_trust_gpu", "trust"),
        "apply_death_gpu": simple("apply_death_gpu", "death"),
        "clip_life_fields_gpu": simple("clip_life_fields_gpu", "clip"),
        "aggregate_global_gpu": aggregate_global_wrapper,
        "dispatch_parent_context_gpu": dispatch_wrapper,
    }
    try:
        with ExitStack() as stack:
            for name, replacement in replacements.items():
                stack.enter_context(_patched(rc, name, replacement))
            yield
    finally:
        if previous_scientific_flag is None:
            run.ds.metadata.pop("scientific_stage_parity", None)
        else:
            run.ds.metadata["scientific_stage_parity"] = previous_scientific_flag
        if hasattr(run, "_scientific_trace_collector"):
            delattr(run, "_scientific_trace_collector")
