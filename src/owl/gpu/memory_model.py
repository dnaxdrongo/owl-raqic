from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AllocationSpec:
    name: str
    bytes: int
    owner: str
    lifetime: str
    shares_storage_with: str | None = None


@dataclass
class MemoryPlan:
    allocations: list[AllocationSpec]
    steady_state_bytes: int
    peak_bytes: int
    allowed_bytes: int | None = None
    safety_margin_bytes: int = 0
    actual_current_bytes: int | None = None
    actual_pool_bytes: int | None = None
    actual_peak_bytes: int | None = None
    unexplained_growth_bytes: int | None = None
    passed: bool = False

    @property
    def estimated_peak_bytes(self) -> int:
        return int(self.peak_bytes)

    def evaluate(self) -> bool:
        estimate_ok = self.allowed_bytes is None or self.peak_bytes <= self.allowed_bytes
        actual_ok = (
            self.actual_peak_bytes is None
            or self.allowed_bytes is None
            or self.actual_peak_bytes <= self.allowed_bytes
        )
        self.passed = bool(estimate_ok and actual_ok)
        return self.passed

    def record_actual(
        self,
        *,
        current_bytes: int | None,
        pool_bytes: int | None,
        peak_bytes: int | None,
        unexplained_growth_bytes: int | None = None,
    ) -> None:
        self.actual_current_bytes = None if current_bytes is None else int(current_bytes)
        self.actual_pool_bytes = None if pool_bytes is None else int(pool_bytes)
        self.actual_peak_bytes = None if peak_bytes is None else int(peak_bytes)
        self.unexplained_growth_bytes = (
            None if unexplained_growth_bytes is None else int(unexplained_growth_bytes)
        )
        self.evaluate()

    def to_dict(self) -> dict[str, Any]:
        self.evaluate()
        owners: dict[str, int] = {}
        for item in self.allocations:
            owners[item.owner] = owners.get(item.owner, 0) + int(item.bytes)
        return {
            "schema_version": "2",
            "allocations": [asdict(item) for item in self.allocations],
            "owners": owners,
            "steady_state_bytes": int(self.steady_state_bytes),
            "peak_bytes": int(self.peak_bytes),
            "estimated_peak_bytes": int(self.peak_bytes),
            "allowed_bytes": self.allowed_bytes,
            "safety_margin_bytes": int(self.safety_margin_bytes),
            "actual_current_bytes": self.actual_current_bytes,
            "actual_pool_bytes": self.actual_pool_bytes,
            "actual_peak_bytes": self.actual_peak_bytes,
            "unexplained_growth_bytes": self.unexplained_growth_bytes,
            "passed": bool(self.passed),
        }


@dataclass(frozen=True)
class CounterfactualMemoryPlan:
    """Calculate bounded single-device source and branch capacity for counterfactual rollouts."""

    factual_fixed_bytes: int
    source_snapshot_bytes: int
    per_branch_state_bytes: int
    per_branch_evidence_bytes: int
    per_branch_scratch_bytes: int
    per_branch_event_bytes: int
    per_branch_contribution_bytes: int
    per_branch_outcome_bytes: int
    hash_workspace_bytes: int
    pinned_packet_bytes: int
    library_reserve_bytes: int
    transient_reserve_bytes: int
    safety_margin_bytes: int
    allowed_bytes: int
    max_active_branches: int
    configured_max_active_branches: int
    passed: bool

    @property
    def per_branch_bytes(self) -> int:
        return (
            self.per_branch_state_bytes
            + self.per_branch_evidence_bytes
            + self.per_branch_scratch_bytes
            + self.per_branch_event_bytes
            + self.per_branch_contribution_bytes
            + self.per_branch_outcome_bytes
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = "owl.counterfactual-memory-plan.v1"
        payload["per_branch_bytes"] = self.per_branch_bytes
        return payload


def build_counterfactual_memory_plan(
    ds: Any,
    cfg: Any,
    *,
    scratch_bytes: int,
    free_device_bytes: int | None = None,
) -> CounterfactualMemoryPlan:
    """Calculate a conservative branch cap without weakening factual preflight."""
    arrays = (*ds.arrays.values(), *ds.patch_arrays.values(), *ds.global_arrays.values())
    state_bytes = sum(int(getattr(value, "nbytes", 0)) for value in arrays)
    cadc_buffer = ds.metadata.get("cadc_device_buffer")
    evidence_arrays = getattr(cadc_buffer, "arrays", {})
    evidence_bytes = sum(
        int(getattr(value, "nbytes", 0)) for value in evidence_arrays.values()
    )
    counterfactual = cfg.counterfactual
    cells = int(ds.health.size)
    fields = 8
    event_capacity = int(counterfactual.event_capacity_per_branch_tick)
    event_bytes = event_capacity * (8 * 8 + 8 * 4)
    contribution_bytes = cells * fields * 8 * 4
    outcome_bytes = len(counterfactual.horizons) * 64 * 8
    hash_workspace = min(state_bytes, 8 * 1024**2)
    pinned = int(counterfactual.max_pending_bytes)
    library = max(64 * 1024**2, int(state_bytes * 0.10))
    transient = max(64 * 1024**2, int(state_bytes * 0.10))
    configured = int(counterfactual.max_device_bytes)
    if free_device_bytes is not None:
        configured = min(
            configured,
            int(float(counterfactual.memory_safety_fraction) * int(free_device_bytes)),
        )
    source_snapshot_bytes = state_bytes + evidence_bytes
    fixed = state_bytes + source_snapshot_bytes + hash_workspace + pinned + library + transient
    margin = max(64 * 1024**2, int(configured * (1.0 - counterfactual.memory_safety_fraction)))
    per_branch = (
        state_bytes
        + evidence_bytes
        + int(scratch_bytes)
        + event_bytes
        + contribution_bytes
        + outcome_bytes
    )
    available = max(0, configured - fixed - margin)
    capacity = available // max(per_branch, 1)
    maximum = min(int(counterfactual.max_active_branches), int(capacity))
    return CounterfactualMemoryPlan(
        factual_fixed_bytes=state_bytes,
        source_snapshot_bytes=source_snapshot_bytes,
        per_branch_state_bytes=state_bytes,
        per_branch_evidence_bytes=evidence_bytes,
        per_branch_scratch_bytes=int(scratch_bytes),
        per_branch_event_bytes=event_bytes,
        per_branch_contribution_bytes=contribution_bytes,
        per_branch_outcome_bytes=outcome_bytes,
        hash_workspace_bytes=hash_workspace,
        pinned_packet_bytes=pinned,
        library_reserve_bytes=library,
        transient_reserve_bytes=transient,
        safety_margin_bytes=margin,
        allowed_bytes=configured,
        max_active_branches=maximum,
        configured_max_active_branches=int(counterfactual.max_active_branches),
        passed=maximum >= 1,
    )


def _storage_identity(array: Any) -> tuple[tuple[str, int], int]:
    """Return owning allocation identity and bytes, not a view offset."""
    root = array
    visited: set[int] = set()
    while True:
        base = getattr(root, "base", None)
        if base is None or base is root or id(base) in visited:
            break
        visited.add(id(root))
        # CuPy may expose a MemoryPointer-like ``base`` without ndarray shape.
        if not hasattr(base, "nbytes") and not hasattr(base, "__array_interface__"):
            break
        root = base
    try:
        ptr = int(root.data.mem.ptr)
        size = int(getattr(root.data.mem, "size", getattr(root, "nbytes", 0)))
        return ("cupy", ptr), size
    except Exception:
        pass
    try:
        ptr = int(root.__array_interface__["data"][0])
        return ("numpy", ptr), int(getattr(root, "nbytes", 0))
    except Exception:
        return ("object", id(root)), int(getattr(root, "nbytes", 0))


def _unique_device_allocations(ds: Any) -> list[AllocationSpec]:
    seen: dict[tuple[str, int], str] = {}
    out: list[AllocationSpec] = []
    groups = (
        ("state", ds.arrays),
        ("patch", ds.patch_arrays),
        ("global", ds.global_arrays),
    )
    for owner, mapping in groups:
        for name, array in mapping.items():
            identity, owned_bytes = _storage_identity(array)
            nbytes = int(owned_bytes)
            if identity in seen:
                out.append(
                    AllocationSpec(
                        name=f"{owner}.{name}",
                        bytes=0,
                        owner=owner,
                        lifetime="run",
                        shares_storage_with=seen[identity],
                    )
                )
            else:
                canonical = f"{owner}.{name}"
                seen[identity] = canonical
                out.append(
                    AllocationSpec(
                        name=canonical,
                        bytes=nbytes,
                        owner=owner,
                        lifetime="run",
                    )
                )
    return out


def _actualization_extension_allocations(ds: Any, cfg: Any) -> list[AllocationSpec]:
    """Return fixed actualization buffers not already owned by device state."""
    rq = cfg.raqic
    enabled = (
        str(getattr(rq, "actualization_variant", "stable_baseline")) != "stable_baseline"
        or bool(getattr(rq, "experimental_shadow_only", False))
        or bool(getattr(rq, "record_actualization_diagnostics", False))
    )
    if not enabled:
        return []
    h, w = ds.health.shape
    n = int(h * w)
    actions = int(ds.possibility.shape[-1])
    audit64 = str(getattr(rq, "full_gpu_precision", "audit64")) == "audit64"
    real_bytes = 8 if audit64 else 4
    complex_bytes = 16 if audit64 else 8
    expected = {
        "_graph_raqic_utilities": n * actions * real_bytes,
        "_graph_raqic_parent_action_phase": n * actions * real_bytes,
        "_graph_raqic_parent_action_coherence": n * actions * real_bytes,
        "_graph_raqic_amplitudes": n * actions * complex_bytes,
        "_graph_raqic_pair_left_scratch": n * complex_bytes,
        "_graph_raqic_pair_right_scratch": n * complex_bytes,
        "_graph_raqic_pre_mixer_probabilities": n * actions * real_bytes,
    }
    return [
        AllocationSpec(
            name=f"gpu.{name}",
            bytes=int(nbytes),
            owner="gpu",
            lifetime="run",
        )
        for name, nbytes in expected.items()
        if name not in ds.arrays
    ]


def build_memory_plan(
    ds: Any,
    cfg: Any,
    *,
    scratch_bytes: int,
    slab_layout: dict[str, Any] | None,
    qiskit_policy: Any | None = None,
    visual_backend: str = "none",
) -> MemoryPlan:
    allocations = _unique_device_allocations(ds)
    allocations.extend(_actualization_extension_allocations(ds, cfg))
    h, w = ds.health.shape
    action_count = int(ds.possibility.shape[-1])
    allocations.append(AllocationSpec("scratch", int(scratch_bytes), "gpu", "run"))
    # Graph-static command, RNG, counters, action candidates, and double-buffer
    # headroom so the preflight calculation includes every fixed allocation.
    # while capture later grew the memory pool.
    command_capacity = int(getattr(cfg.raqic, "full_gpu_command_capacity", 1024))
    allocations.extend(
        [
            AllocationSpec("command_buffers", command_capacity * 8 * 8, "gpu", "run"),
            AllocationSpec("device_rng_and_tick", max(4096, h * w * 8), "gpu", "run"),
            AllocationSpec("device_diagnostic_counters", 256 * 8, "gpu", "run"),
            AllocationSpec("graph_conflict_buffers", h * w * 16 * 2, "gpu", "run"),
        ]
    )

    event_capacity = int(getattr(cfg.raqic, "full_gpu_sparse_event_capacity", 4096))
    event_bytes = event_capacity * (5 * 4 + 3 * 8 + 2 + 2 + 2)
    allocations.append(AllocationSpec("topology_event_buffers", event_bytes, "gpu", "run"))
    visual_event_capacity = int(getattr(cfg.raqic, "full_gpu_visual_event_capacity", 16384))
    allocations.append(
        AllocationSpec(
            "visual_event_buffers",
            visual_event_capacity * 14 * 4,
            "gpu",
            "run",
        )
    )
    if visual_backend != "none":
        allocations.extend(
            [
                AllocationSpec("visual_rgba_device", h * w * 4, "visual", "run"),
                AllocationSpec("visual_rgba_pinned", h * w * 4 * 2, "visual", "run"),
                AllocationSpec("visual_texture", h * w * 4, "visual", "run"),
            ]
        )
    # Metric double buffers and bounded async writer estimates.
    allocations.append(AllocationSpec("metric_pinned_buffers", 64 * 8 * 3, "recording", "run"))
    writer_capacity = int(getattr(cfg.raqic, "full_gpu_writer_queue_capacity", 1024))
    # Compact JSON records are bounded by policy. Large snapshots are written
    # by path after checkpointing and are not retained in the queue.
    allocations.append(
        AllocationSpec(
            "async_writer_backlog",
            writer_capacity * 4096,
            "recording",
            "run",
        )
    )
    recording_level = str(getattr(cfg.raqic, "full_gpu_recording_level_v07", "metrics_plus_events"))
    if recording_level in {"full_snapshot_decimated", "debug_full_every_tick"}:
        state_bytes = sum(
            item.bytes for item in allocations if item.owner in {"state", "patch", "global"}
        )
        allocations.append(
            AllocationSpec("checkpoint_staging", state_bytes, "recording", "transient")
        )

    if qiskit_policy is not None and getattr(qiskit_policy, "per_ow", False):
        from owl_raqic.qiskit_backend.circuit_families import CIRCUIT_FAMILIES

        complex_bytes = (
            16 if str(getattr(cfg.raqic, "full_gpu_precision", "audit64")) == "audit64" else 8
        )
        chunk = int(getattr(qiskit_policy, "chunk_size", 64))
        queue_depth = max(1, int(getattr(qiskit_policy, "job_queue_depth", 1)))
        family_working = []
        for family in getattr(qiskit_policy, "circuit_families", ("static",)):
            spec = CIRCUIT_FAMILIES[str(family)]
            per_row = int(
                spec.memory_estimator(
                    action_count,
                    precision_bytes=complex_bytes,
                    n_positions=max(2, 1 << max(1, int(math.ceil(math.log2(action_count))))),
                )
            )
            # State, simulator scratch, result staging, and concurrent queued jobs.
            bytes_for_family = per_row * chunk * queue_depth * 2
            family_working.append(bytes_for_family)
            allocations.append(
                AllocationSpec(
                    f"qiskit_family_{family}_working", bytes_for_family, "qiskit", "transient"
                )
            )
        allocations.extend(
            [
                AllocationSpec(
                    "qiskit_parameter_buffers",
                    chunk * action_count * 16 * 2 * queue_depth,
                    "qiskit",
                    "run",
                ),
                AllocationSpec(
                    "qiskit_result_buffers",
                    chunk * action_count * 8 * 2 * queue_depth,
                    "qiskit",
                    "run",
                ),
                AllocationSpec(
                    "qiskit_transpile_and_metadata_reserve",
                    max(16 * 1024 * 1024, sum(family_working) // 4),
                    "qiskit",
                    "transient",
                ),
            ]
        )

    devices = tuple(getattr(cfg.raqic, "full_gpu_devices", ()))
    if bool(getattr(cfg.raqic, "full_gpu_multi_gpu", False)) or len(devices) > 1:
        from owl.gpu.distributed.halo_protocol import generate_halo_protocol

        protocol = generate_halo_protocol(ds)
        halo_width = max(1, int(protocol.halo_width))
        halo_row_bytes = 0
        for name in protocol.fields:
            array = ds.arrays.get(name)
            if array is None:
                continue
            trailing = int(np.prod(array.shape[1:], dtype=np.int64))
            halo_row_bytes += trailing * int(array.dtype.itemsize)
        # north/south send+receive, double buffered for overlap.
        allocations.extend(
            [
                AllocationSpec(
                    "nccl_halo_buffers", halo_row_bytes * halo_width * 8, "distributed", "run"
                ),
                AllocationSpec(
                    "distributed_boundary_compare",
                    halo_row_bytes * halo_width * 4,
                    "distributed",
                    "run",
                ),
                AllocationSpec(
                    "distributed_reconciliation", h * w * 32, "distributed", "transient"
                ),
                AllocationSpec(
                    "distributed_global_reduce_buffers",
                    max(4096, action_count * 8 * 16),
                    "distributed",
                    "run",
                ),
            ]
        )
    primary = sum(
        item.bytes for item in allocations if item.owner in {"state", "patch", "global", "gpu"}
    )
    allocations.append(
        AllocationSpec(
            "library_workspace_reserve",
            max(64 * 1024 * 1024, int(primary * 0.10)),
            "library",
            "run",
        )
    )
    steady = sum(item.bytes for item in allocations if item.lifetime == "run")
    transient = sum(item.bytes for item in allocations if item.lifetime == "transient")
    configured_limit = getattr(cfg.raqic, "gpu_memory_limit_mb", None)
    allowed = None if configured_limit is None else int(float(configured_limit) * 1024 * 1024)
    safety_margin = int(max(64 * 1024 * 1024, 0.05 * (steady + transient)))
    plan = MemoryPlan(
        allocations=allocations,
        steady_state_bytes=int(steady),
        peak_bytes=int(steady + transient + safety_margin),
        allowed_bytes=allowed,
        safety_margin_bytes=safety_margin,
    )
    plan.evaluate()
    return plan
