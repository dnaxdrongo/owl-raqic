"""Operation-level CUDA graph safety contracts.

This module deliberately separates *declared* capture support from observed
runtime evidence.  A full-tick plan is eligible for capture only when every
operation in each segment has an explicit audit record and the allocation
snapshot remains stable after warm-up/capture.  Unknown operations fail closed.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class GraphOperationAudit:
    callable_name: str
    allocates: bool = False
    host_sync: bool = False
    d2h_transfer: bool = False
    python_device_branch: bool = False
    capture_sensitive_library: str | None = None
    replay_python_side_effect: bool = False
    approved: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GraphSegmentAudit:
    name: str
    operations: tuple[GraphOperationAudit, ...]

    @property
    def approved(self) -> bool:
        return bool(self.operations) and all(op.approved for op in self.operations)

    @property
    def failures(self) -> tuple[str, ...]:
        return tuple(
            f"{op.callable_name}: {op.reason or 'not approved'}"
            for op in self.operations
            if not op.approved
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "approved": self.approved,
            "failures": list(self.failures),
            "operations": [op.to_dict() for op in self.operations],
        }


@dataclass(frozen=True)
class GraphSafetyManifest:
    segments: dict[str, GraphSegmentAudit]
    schema_version: str = "owl.graph-safety.v1"

    def segment_approved(self, name: str) -> bool:
        audit = self.segments.get(name)
        return bool(audit and audit.approved)

    def reason(self, name: str) -> str:
        audit = self.segments.get(name)
        if audit is None:
            return "segment absent from graph-safety manifest"
        return "; ".join(audit.failures) if audit.failures else "approved"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "passed": all(item.approved for item in self.segments.values()),
            "segments": {name: item.to_dict() for name, item in sorted(self.segments.items())},
        }


@dataclass(frozen=True)
class MemoryPoolSnapshot:
    used_bytes: int | None
    total_bytes: int | None
    pinned_free_blocks: int | None = None

    def to_dict(self) -> dict[str, int | None]:
        return asdict(self)


@dataclass
class CaptureAllocationGuard:
    """Observe CuPy pool growth across capture without importing CuPy eagerly."""

    xp: Any
    allowed_used_growth: int = 0
    allowed_total_growth: int = 0
    records: dict[str, dict[str, Any]] = field(default_factory=dict)

    def snapshot(self) -> MemoryPoolSnapshot:
        get_pool = getattr(self.xp, "get_default_memory_pool", None)
        if get_pool is None:
            return MemoryPoolSnapshot(None, None, None)
        try:
            pool = get_pool()
            used = int(pool.used_bytes())
            total = int(pool.total_bytes())
            pinned_blocks = None
            get_pinned = getattr(self.xp, "get_default_pinned_memory_pool", None)
            if get_pinned is not None:
                pinned = get_pinned()
                nfree = getattr(pinned, "n_free_blocks", None)
                pinned_blocks = int(nfree()) if callable(nfree) else None
            return MemoryPoolSnapshot(used, total, pinned_blocks)
        except Exception:
            return MemoryPoolSnapshot(None, None, None)

    def compare(
        self,
        segment: str,
        before: MemoryPoolSnapshot,
        after: MemoryPoolSnapshot,
    ) -> tuple[bool, str]:
        if before.used_bytes is None or after.used_bytes is None:
            record = {
                "before": before.to_dict(),
                "after": after.to_dict(),
                "observed": False,
                "passed": False,
                "reason": "memory-pool telemetry unavailable",
            }
            self.records[segment] = record
            return False, str(record["reason"])
        used_growth = int(after.used_bytes - before.used_bytes)
        total_growth = int((after.total_bytes or 0) - (before.total_bytes or 0))
        passed = used_growth <= int(self.allowed_used_growth) and total_growth <= int(
            self.allowed_total_growth
        )
        reason = (
            ""
            if passed
            else (
                f"unplanned capture allocation: used_growth={used_growth}, "
                f"total_growth={total_growth}"
            )
        )
        self.records[segment] = {
            "before": before.to_dict(),
            "after": after.to_dict(),
            "observed": True,
            "used_growth": used_growth,
            "total_growth": total_growth,
            "allowed_used_growth": int(self.allowed_used_growth),
            "allowed_total_growth": int(self.allowed_total_growth),
            "passed": passed,
            "reason": reason,
        }
        return passed, reason

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed_used_growth": int(self.allowed_used_growth),
            "allowed_total_growth": int(self.allowed_total_growth),
            "records": dict(self.records),
            "passed": bool(self.records)
            and all(bool(record.get("passed", False)) for record in self.records.values()),
        }


def operation(
    name: str,
    *,
    approved: bool,
    reason: str,
    allocates: bool = False,
    host_sync: bool = False,
    d2h_transfer: bool = False,
    python_device_branch: bool = False,
    capture_sensitive_library: str | None = None,
    replay_python_side_effect: bool = False,
) -> GraphOperationAudit:
    return GraphOperationAudit(
        callable_name=name,
        allocates=allocates,
        host_sync=host_sync,
        d2h_transfer=d2h_transfer,
        python_device_branch=python_device_branch,
        capture_sensitive_library=capture_sensitive_library,
        replay_python_side_effect=replay_python_side_effect,
        approved=approved,
        reason=reason,
    )


def build_default_graph_safety_manifest() -> GraphSafetyManifest:
    """Return the maintained operation inventory for OWL's four segments.

    The approvals are eligibility declarations only.  Runtime pointer and pool
    evidence remain mandatory.  Operations that may allocate temporary arrays
    are marked as such; capture is accepted only after warm-up makes the pool
    stable and the allocation guard observes no growth.
    """

    def vector(name: str, reason: str = "fixed-shape device array operation") -> Any:
        return operation(name, approved=True, reason=reason, allocates=True)

    predecision = (
        vector("apply_device_commands", "fixed-capacity command buffer"),
        vector("capture_tick_start_gpu", "preallocated coordinate snapshots"),
        vector("update_environment_gpu"),
        vector("prepare_sensing_stencil_scratch", "preallocated scratch views"),
        vector("compute_sensing_bundle_gpu"),
        vector("aggregate_patches_gpu"),
        vector("aggregate_global_gpu"),
        vector("dispatch_parent_context_gpu"),
        vector("apply_threshold_modulation_gpu"),
        vector("update_phase_gpu"),
        vector("compute_local_synchrony_gpu"),
        vector("compute_cell_coherence_gpu"),
        vector("compute_cross_scale_coupling_gpu"),
        vector("compute_utilities_gpu"),
        vector("compute_authority_gpu"),
        vector("capture_pre_decision_state_gpu", "preallocated causal audit snapshots"),
    )
    decision = (
        vector(
            "ensure_actualization_graph_buffers_gpu",
            "preallocated v0.9.6 decision buffers",
        ),
        vector(
            "aggregate_action_phase_context_gpu",
            "fixed-shape probability-weighted phasor reductions",
        ),
        vector("run_raqic_gpu_stage", "fixed-shape dense device decision path"),
    )
    actions = (
        vector("apply_movement_graph_static", "fixed-capacity candidate buffers"),
        vector("resolve_collisions_gpu"),
        vector("apply_inhibition_gpu"),
        vector("apply_feeding_gpu"),
        vector("apply_repair_and_integrate_gpu"),
        vector("emit_signals_gpu"),
        vector("apply_reproduction_graph_static", "fixed-capacity candidate buffers"),
        vector("apply_topology_graph_static", "fixed-capacity event buffers"),
    )
    postdecision = (
        vector("apply_metabolism_damage_gpu"),
        vector("update_memory_gpu"),
        vector("update_signal_memory_gpu"),
        vector("update_channel_trust_gpu"),
        vector("update_integration_gpu"),
        vector("apply_death_gpu"),
        vector("clip_life_fields_gpu"),
        vector(
            "quiesce_dead_raqic_fields_gpu",
            "REST terminal RAQIC state for every dead or obstacle cell",
        ),
        vector("aggregate_patches_gpu"),
        vector("aggregate_global_gpu"),
        vector("dispatch_parent_context_gpu"),
    )
    return GraphSafetyManifest(
        {
            "predecision": GraphSegmentAudit("predecision", predecision),
            "decision": GraphSegmentAudit("decision", decision),
            "actions": GraphSegmentAudit("actions", actions),
            "postdecision": GraphSegmentAudit("postdecision", postdecision),
        }
    )


def validate_manifest_callables(
    manifest: GraphSafetyManifest,
    available_callables: Iterable[str],
) -> tuple[bool, tuple[str, ...]]:
    available = set(available_callables)
    missing = tuple(
        sorted(
            op.callable_name
            for segment in manifest.segments.values()
            for op in segment.operations
            if op.callable_name not in available
        )
    )
    return not missing, missing
