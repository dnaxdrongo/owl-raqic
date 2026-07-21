from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from owl.gpu.graph_safety import CaptureAllocationGuard, GraphSafetyManifest


@dataclass
class SegmentRecord:
    name: str
    callback: Callable[[], Any]
    graph_safe: bool = False
    graph: Any = None
    capture_count: int = 0
    replay_count: int = 0
    run_count: int = 0
    reason: str = ""
    pointer_stable: bool | None = None
    pointer_changes: tuple[str, ...] = ()


@dataclass(frozen=True)
class GraphCoverage:
    required_segments: tuple[str, ...]
    captured_segments: tuple[str, ...]
    replay_counts: dict[str, int]
    uncovered_reasons: dict[str, str]

    @property
    def full_tick(self) -> bool:
        return set(self.required_segments) == set(self.captured_segments)

    def to_dict(self) -> dict[str, Any]:
        return {
            "required_segments": list(self.required_segments),
            "captured_segments": list(self.captured_segments),
            "replay_counts": dict(self.replay_counts),
            "uncovered_reasons": dict(self.uncovered_reasons),
            "full_tick": self.full_tick,
        }


@dataclass
class GpuTickGraphManager:
    """CUDA Graph segment manager with explicit, auditable status.

    Segments must opt in as ``graph_safe``.  This prevents a callback that
    allocates dynamically or synchronizes with the host from being captured by
    accident.  When capture is unavailable the callback is still executed.
    """

    backend: Any
    mode: str = "off"
    requirement: str = "allow_partial"
    required_segments: tuple[str, ...] = ("predecision", "decision", "actions", "postdecision")
    segments: dict[str, SegmentRecord] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)
    invalidation_count: int = 0
    pointer_snapshot: Callable[[], dict[str, int]] | None = None
    safety_manifest: GraphSafetyManifest | None = None
    allocation_guard: CaptureAllocationGuard | None = None

    @property
    def graphs(self) -> dict[str, Any]:
        return {name: rec.graph for name, rec in self.segments.items() if rec.graph is not None}

    def can_capture(self) -> tuple[bool, str]:
        if self.mode == "off":
            return False, "graph mode off"
        if not getattr(self.backend, "is_gpu", False):
            return False, "CUDA backend unavailable"
        stream_cls = getattr(getattr(self.backend.xp, "cuda", None), "Stream", None)
        if stream_cls is None or not hasattr(stream_cls, "begin_capture"):
            return False, "CuPy stream capture unavailable"
        return True, "ok"

    def prepare_segments(
        self, mapping: dict[str, tuple[Callable[[], Any], bool] | Callable[[], Any]]
    ) -> None:
        for name, item in mapping.items():
            if isinstance(item, tuple):
                callback, graph_safe = item
            else:
                callback, graph_safe = item, False
            self.segments[name] = SegmentRecord(
                name=name, callback=callback, graph_safe=bool(graph_safe)
            )

    def warmup(self, callback: Any | None = None, count: int = 1) -> None:
        callbacks = (
            [callback] if callback is not None else [rec.callback for rec in self.segments.values()]
        )
        for _ in range(max(0, int(count))):
            for cb in callbacks:
                cb()
            self.backend.synchronize()

    def capture_segment(
        self,
        name: str,
        callback: Any | None = None,
        *,
        stream: Any | None = None,
        graph_safe: bool | None = None,
    ) -> bool:
        rec = self.segments.get(name)
        if rec is None:
            if callback is None:
                raise KeyError(name)
            rec = SegmentRecord(name=name, callback=callback, graph_safe=bool(graph_safe))
            self.segments[name] = rec
        elif callback is not None:
            rec.callback = callback
        if graph_safe is not None:
            rec.graph_safe = bool(graph_safe)

        ok, reason = self.can_capture()
        if not ok:
            rec.reason = reason
            self.reasons[name] = reason
            return False
        if self.safety_manifest is not None and not self.safety_manifest.segment_approved(name):
            reason = "graph safety manifest rejected segment: " + self.safety_manifest.reason(name)
            rec.reason = reason
            self.reasons[name] = reason
            return False
        if not rec.graph_safe:
            reason = "segment not declared graph-safe"
            rec.reason = reason
            self.reasons[name] = reason
            return False

        cp = self.backend.xp
        capture_stream = stream or cp.cuda.Stream(non_blocking=True)
        before = self.pointer_snapshot() if self.pointer_snapshot is not None else None
        pool_before = (
            self.allocation_guard.snapshot() if self.allocation_guard is not None else None
        )
        try:
            with capture_stream:
                capture_stream.begin_capture()
                rec.callback()
                graph = capture_stream.end_capture()
            after = self.pointer_snapshot() if self.pointer_snapshot is not None else None
            pool_after = (
                self.allocation_guard.snapshot() if self.allocation_guard is not None else None
            )
            if (
                pool_before is not None
                and pool_after is not None
                and self.allocation_guard is not None
            ):
                pool_ok, pool_reason = self.allocation_guard.compare(name, pool_before, pool_after)
                if not pool_ok:
                    rec.graph = None
                    rec.reason = pool_reason
                    self.reasons[name] = pool_reason
                    return False
            if before is not None and after is not None:
                changed = tuple(
                    sorted(
                        key for key in set(before) | set(after) if before.get(key) != after.get(key)
                    )
                )
                rec.pointer_stable = not changed
                rec.pointer_changes = changed
                if changed:
                    rec.graph = None
                    reason = "capture changed persistent device addresses: " + ", ".join(
                        changed[:12]
                    )
                    rec.reason = reason
                    self.reasons[name] = reason
                    return False
            else:
                rec.pointer_stable = None
                rec.pointer_changes = ()
            rec.graph = graph
            rec.capture_count += 1
            rec.reason = ""
            self.reasons.pop(name, None)
            return True
        except Exception as exc:  # pragma: no cover - GPU host specific
            reason = f"capture failed: {type(exc).__name__}: {exc}"
            rec.reason = reason
            self.reasons[name] = reason
            rec.graph = None
            rec.pointer_stable = False if before is not None else None
            return False

    def capture_available_segments(self, *, stream: Any | None = None) -> dict[str, bool]:
        return {name: self.capture_segment(name, stream=stream) for name in self.segments}

    def replay_segment(self, name: str, *, stream: Any | None = None) -> bool:
        rec = self.segments.get(name)
        if rec is None or rec.graph is None:
            return False
        target_stream = stream
        try:
            if target_stream is not None:
                rec.graph.launch(stream=target_stream)
            else:
                rec.graph.launch()
        except TypeError:  # CuPy-version compatibility
            rec.graph.launch(stream=target_stream or self.backend.xp.cuda.Stream.null)
        rec.replay_count += 1
        return True

    def replay_or_run_segment(self, name: str, *, stream: Any | None = None) -> Any:
        rec = self.segments[name]
        if self.replay_segment(name, stream=stream):
            return None
        rec.run_count += 1
        if stream is None:
            return rec.callback()
        with stream:
            return rec.callback()

    def invalidate(self, reason: str = "invalidated") -> None:
        for rec in self.segments.values():
            rec.graph = None
            rec.reason = reason
            rec.pointer_stable = None
            rec.pointer_changes = ()
        self.reasons["global"] = reason
        self.invalidation_count += 1

    @property
    def replay_count(self) -> int:
        return sum(rec.replay_count for rec in self.segments.values())

    def coverage(self) -> GraphCoverage:
        captured = tuple(
            sorted(name for name, rec in self.segments.items() if rec.graph is not None)
        )
        replay_counts = {name: int(rec.replay_count) for name, rec in sorted(self.segments.items())}
        reasons = {
            name: rec.reason or "not captured"
            for name, rec in sorted(self.segments.items())
            if name in self.required_segments and rec.graph is None
        }
        return GraphCoverage(
            required_segments=tuple(self.required_segments),
            captured_segments=captured,
            replay_counts=replay_counts,
            uncovered_reasons=reasons,
        )

    def assert_requirement(self) -> None:
        coverage = self.coverage()
        if self.requirement == "full_tick":
            if not coverage.full_tick:
                raise RuntimeError(
                    "full-tick CUDA graph requirement is not satisfied: "
                    f"{coverage.uncovered_reasons}"
                )
            missing_replay = [
                name for name in self.required_segments if coverage.replay_counts.get(name, 0) <= 0
            ]
            if missing_replay:
                raise RuntimeError(
                    f"captured CUDA graph segments have not replayed: {missing_replay}"
                )

    def graph_status(self) -> dict[str, Any]:
        ok, reason = self.can_capture()
        return {
            "requested_mode": self.mode,
            "requirement": self.requirement,
            "coverage": self.coverage().to_dict(),
            "can_capture": ok,
            "reason": reason if not ok else "",
            "captured_segments": sorted(
                name for name, rec in self.segments.items() if rec.graph is not None
            ),
            "disabled_reasons": dict(self.reasons),
            "invalidation_count": self.invalidation_count,
            "safety_manifest": None
            if self.safety_manifest is None
            else self.safety_manifest.to_dict(),
            "allocation_guard": None
            if self.allocation_guard is None
            else self.allocation_guard.to_dict(),
            "capture_contract": {
                "pointer_snapshot_enabled": self.pointer_snapshot is not None,
                "captured_segments_pointer_stable": all(
                    rec.pointer_stable is not False
                    for rec in self.segments.values()
                    if rec.graph is not None
                ),
            },
            "segments": {
                name: {
                    "graph_safe": rec.graph_safe,
                    "captured": rec.graph is not None,
                    "capture_count": rec.capture_count,
                    "replay_count": rec.replay_count,
                    "run_count": rec.run_count,
                    "reason": rec.reason,
                    "pointer_stable": rec.pointer_stable,
                    "pointer_changes": list(rec.pointer_changes),
                }
                for name, rec in sorted(self.segments.items())
            },
        }
