"""Bounded isolated branch scheduler for authoritative micro-rollouts."""

from __future__ import annotations

import copy
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import Lock
from time import perf_counter
from typing import Any

from owl.core.actions import Action
from owl.counterfactual.evidence import (
    BranchEvidencePacket,
    capture_branch_evidence,
    transfer_branch_evidence,
)
from owl.counterfactual.forcing import (
    ForcedActionBatch,
    build_forced_action_batch,
    inject_forced_actions,
)
from owl.counterfactual.outcomes import (
    OutcomeDevicePacket,
    capture_outcome_device,
    transfer_outcome,
)
from owl.counterfactual.rng_registry import branch_seed
from owl.counterfactual.schema import (
    BranchStatus,
    branch_id,
    pair_id,
)
from owl.counterfactual.source import CollectedSource
from owl.counterfactual.state_hash import (
    StateHashResult,
    compare_state_science,
    differing_leaves,
    hash_state,
)
from owl.gpu.commands import GPUCommandQueue
from owl.gpu.graphs import GpuTickGraphManager
from owl.gpu.metrics_slab import DeviceMetricSlab
from owl.gpu.numerical_ledger import NumericalLedger
from owl.gpu.profiler import GPUFullProfiler
from owl.gpu.scratch import ScratchManager
from owl.gpu.streams import StreamBundle
from owl.gpu.transfer_ledger import TransferLedger
from owl.record.cadc_capture import capture_selected_intent
from owl.record.cadc_device_buffer import CADCDeviceBuffer


@dataclass(frozen=True)
class NonExecutableAttempt:
    source_state_id: str
    source_decision_id: str
    branch_id: str
    repeat_index: int
    branch_seed: int
    factual_action: int
    forced_action: int
    policy_legal: bool
    prechoice_executable: bool
    reason_code: int


@dataclass(frozen=True)
class CandidatePair:
    pair_id: str
    source_decision_id: str
    repeat_index: int
    action_a: int
    action_b: int
    branch_a: str
    branch_b: str
    horizon: int


@dataclass
class BranchResult:
    branch_id: str
    source_state_id: str
    source_decision_id: str
    repeat_index: int
    branch_seed: int
    factual_action: int
    forced_action: int
    anchor: bool
    status: BranchStatus
    pre_force_hash: StateHashResult
    post_force_hash: StateHashResult
    force_changed_leaves: tuple[str, ...] = ()
    horizon_hashes: dict[int, StateHashResult] = field(default_factory=dict)
    outcomes: dict[int, OutcomeDevicePacket] = field(default_factory=dict)
    evidence: list[BranchEvidencePacket] = field(default_factory=list)
    validation_passed: bool = False
    anchor_matches: dict[int, bool] = field(default_factory=dict)
    anchor_exact_hash_matches: dict[int, bool] = field(default_factory=dict)
    runtime_seconds: float = 0.0
    failure: str | None = None
    failure_traceback: tuple[str, ...] = ()


@dataclass
class CounterfactualRunResult:
    source_state_id: str
    source_hash: StateHashResult
    branches: list[BranchResult]
    nonexecutable: list[NonExecutableAttempt]
    pairs: list[CandidatePair]


@dataclass
class CounterfactualScheduler:
    """Correctness-first scheduler using whole-array branch views.

    Branches are currently submitted through a bounded Python loop. Each
    scientific stage remains a whole-array NumPy/CuPy operation and no loop is
    introduced over OWs, cells, or candidate rows inside the hot kernels.
    """

    template_run: Any
    cfg: Any
    transfer_ledger: TransferLedger = field(default_factory=TransferLedger)
    active_branch_limit: int | None = None
    _ledger_lock: Lock = field(default_factory=Lock, init=False, repr=False)
    last_worker_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if not bool(self.cfg.counterfactual.enabled):
            raise ValueError("counterfactual scheduler requires enabled configuration")
        if self.template_run.per_ow_qiskit is not None and not bool(
            self.cfg.counterfactual.allow_per_ow_qiskit
        ):
            raise RuntimeError("authoritative per-OW Qiskit is not branch-local certified")
        if self.active_branch_limit is not None and self.active_branch_limit < 1:
            raise ValueError("active branch limit must be positive")

    def _branch_context(
        self,
        collected: CollectedSource,
        seed: int,
    ) -> Any:
        run = copy.copy(self.template_run)
        ds = collected.state.branch_clone()
        branch_cfg = self.cfg.model_copy(deep=True)
        branch_cfg.world.seed = int(seed)
        ds.metadata["cfg"] = branch_cfg
        ds.metadata["graph_static"] = False
        # Persistent factual execution keeps aggregate metrics as backend
        # scalars. Branches must preserve that mode: converting a CuPy scalar
        # to a Python float both synchronizes the device and makes later CuPy
        # ufuncs backend-incompatible.
        ds.metadata["defer_host_metrics"] = True
        ds.metadata["counterfactual_suppress_host_event_queue"] = True
        ds.metadata["counterfactual_suppress_close_reports"] = True
        streams = StreamBundle.create(ds.backend)
        scratch = ScratchManager.for_config(ds.backend, branch_cfg)
        graph_manager = GpuTickGraphManager(ds.backend, mode="off")
        buffer = CADCDeviceBuffer.create(ds, branch_cfg)
        for name, source in collected.factual_evidence.items():
            if name in buffer.arrays:
                buffer.arrays[name][...] = source
        ds.metadata["cadc_device_buffer"] = buffer
        run.cfg = branch_cfg
        run.ds = ds
        run.state = copy.deepcopy(self.template_run.state)
        run.streams = streams
        run.scratch = scratch
        run.graph_manager = graph_manager
        run.profiler = GPUFullProfiler(ds.backend, streams.compute, enabled=True)
        run.metric_slab = DeviceMetricSlab.create(ds.backend)
        run.transfer_ledger = TransferLedger()
        run.numerical_ledger = NumericalLedger.from_config(branch_cfg)
        run.command_queue = GPUCommandQueue(int(branch_cfg.raqic.full_gpu_command_capacity))
        run.device_command_buffer = None
        run.slab_manager = None
        run.qiskit_validation = None
        run.per_ow_qiskit = None
        run.shadow_auditor = None
        run.visual_controller = None
        run.async_writer = None
        run.recording_policy = None
        run.cadc_buffer = buffer
        run.counterfactual_source_observer = None
        run.pending_metric_tickets = []
        run.metrics = []
        run.last_diagnostics = {}
        run.fallback_count = 0 if ds.is_gpu else 1
        run.closed = False
        run.paused = False
        run.checkpoint_requested = False
        run.validation_requested = False
        run.visual_settings = {}
        run._steps_completed = int(ds.tick)
        run._checkpointed_tick = None
        run._checkpoint_counted_tick = None
        run._last_checkpoint = None
        run.checkpoint_count = 0
        run.memory_plan = None
        run.run_paths = None
        run._prepare_graph_segments()
        with self._ledger_lock:
            self.transfer_ledger.record_d2d(
                collected.state.nbytes,
                kind="counterfactual_branch",
                tick=int(ds.tick),
                source_stream="counterfactual-branch-lane",
                synchronization="event",
                scheduled=True,
                graph_compatible=False,
                reason="complete source-to-branch expansion",
            )
        return run

    def _horizons(self, action: int) -> tuple[int, ...]:
        values = {int(item) for item in self.cfg.counterfactual.horizons}
        values.update(
            int(item)
            for item in self.cfg.counterfactual.family_horizons.get(Action(action).name, ())
        )
        return tuple(sorted(values))

    def _execute_branch(
        self,
        collected: CollectedSource,
        decision_index: int,
        decision_id: str,
        forced_action: int,
        repeat_index: int,
        seed: int,
        *,
        anchor: bool,
    ) -> BranchResult:
        decisions = collected.decisions
        bid = branch_id(
            collected.state.source_state_id,
            decision_id,
            repeat_index,
            forced_action,
            seed,
            "anchor" if anchor else "paired",
        )
        run = self._branch_context(collected, seed)
        forced: ForcedActionBatch = build_forced_action_batch(
            decisions, [decision_index], [forced_action]
        )
        pre = hash_state(run.ds)
        result = BranchResult(
            branch_id=bid,
            source_state_id=collected.state.source_state_id,
            source_decision_id=decision_id,
            repeat_index=repeat_index,
            branch_seed=seed,
            factual_action=int(self.template_run.ds.backend.asnumpy(forced.factual_action)[0]),
            forced_action=forced_action,
            anchor=anchor,
            status=BranchStatus.RUNNING,
            pre_force_hash=pre,
            post_force_hash=pre,
        )
        started = perf_counter()
        try:
            valid = inject_forced_actions(run.ds, forced)
            result.validation_passed = bool(run.ds.backend.asnumpy(valid)[0])
            if not result.validation_passed:
                raise RuntimeError("forced action failed source validation")
            result.post_force_hash = hash_state(run.ds)
            result.force_changed_leaves = differing_leaves(pre, result.post_force_hash)
            allowed_force_leaves = {"arrays.readout", "arrays.raqic_readout"}
            if not set(result.force_changed_leaves) <= allowed_force_leaves:
                raise RuntimeError(
                    "forced action mutated fields outside the registered seam: "
                    f"{result.force_changed_leaves}"
                )
            capture_selected_intent(run.cadc_buffer, run.ds, run.cfg)
            horizons = self._horizons(forced_action)
            first_death = run.ds.xp.asarray(-1, dtype=run.ds.xp.int64)
            y = forced.source_y[0]
            x = forced.source_x[0]
            source_baseline = {
                "coordinate_y": y.copy(),
                "coordinate_x": x.copy(),
                "health": run.ds.health[y, x].copy(),
                "resource": run.ds.resource[y, x].copy(),
                "boundary": run.ds.boundary[y, x].copy(),
                "integration": run.ds.integration[y, x].copy(),
                "memory": run.ds.memory[y, x].copy(),
            }

            def capture_completed(offset: int) -> None:
                nonlocal first_death
                run.streams.compute.synchronize()
                ow_id = forced.ow_id[0]
                alive = run.ds.xp.any(
                    (run.ds.occupancy == ow_id) & (run.ds.health > 0.0) & (~run.ds.obstacle)
                )
                first_death = run.ds.xp.where(
                    (first_death < 0) & (~alive), int(run.ds.tick), first_death
                )
                device_evidence = capture_branch_evidence(run.cadc_buffer)
                host_evidence = transfer_branch_evidence(run.ds.backend, device_evidence)
                result.evidence.append(host_evidence)
                if run.ds.is_gpu:
                    evidence_bytes = host_evidence.nbytes
                    run.transfer_ledger.record_d2h(
                        evidence_bytes,
                        kind="counterfactual_event",
                        tick=int(run.ds.tick),
                        source_stream="counterfactual-branch-lane",
                        synchronization="event",
                        scheduled=True,
                        graph_compatible=False,
                        reason="bounded branch event/contribution packet",
                    )
                if offset in horizons:
                    device_outcome = capture_outcome_device(
                        run.ds,
                        forced,
                        horizon=offset,
                        source_tick=decisions.tick,
                        first_death_tick=first_death,
                        source_baseline=source_baseline,
                    )
                    result.outcomes[offset] = transfer_outcome(run.ds.backend, device_outcome)
                    if run.ds.is_gpu:
                        run.transfer_ledger.record_d2h(
                            result.outcomes[offset].nbytes,
                            kind="counterfactual_outcome",
                            tick=int(run.ds.tick),
                            source_stream="counterfactual-branch-lane",
                            synchronization="event",
                            scheduled=True,
                            graph_compatible=False,
                            reason="compact branch horizon outcome vector",
                        )
                    result.horizon_hashes[offset] = hash_state(run.ds)
                    if run.ds.is_gpu:
                        run.transfer_ledger.record_d2h(
                            result.horizon_hashes[offset].device_to_host_bytes,
                            kind="counterfactual_hash",
                            tick=int(run.ds.tick),
                            source_stream="counterfactual-hash",
                            synchronization="stream",
                            scheduled=True,
                            graph_compatible=False,
                            reason="canonical complete-state Merkle leaves",
                        )
                    if anchor and offset in collected.factual_horizons:
                        factual_endpoint = collected.factual_horizons[offset]
                        result.anchor_exact_hash_matches[offset] = (
                            result.horizon_hashes[offset].root == hash_state(factual_endpoint).root
                        )
                        result.anchor_matches[offset] = compare_state_science(
                            run.ds, factual_endpoint
                        ).passed

            with run.streams.compute:
                run._segment_actions()
                run._segment_postdecision()
            capture_completed(1)
            for offset in range(2, max(horizons) + 1):
                run._step_device([])
                capture_completed(offset)
            if (
                anchor
                and self.cfg.counterfactual.require_anchor_equivalence
                and not all(
                    result.anchor_matches.get(horizon, False)
                    for horizon in result.horizon_hashes
                    if horizon in collected.factual_horizons
                )
            ):
                raise AssertionError("selected anchor differs from factual remainder")
            if any(bool(packet.event_overflow[0]) for packet in result.evidence):
                raise OverflowError("branch event buffer overflow")
            result.status = BranchStatus.COMPLETED
        except Exception as exc:
            result.status = BranchStatus.FAILED
            result.failure = f"{type(exc).__name__}: {exc}"
            result.failure_traceback = tuple(traceback.format_exc().splitlines())
        finally:
            result.runtime_seconds = perf_counter() - started
            run.streams.synchronize_all()
            with self._ledger_lock:
                for record in run.transfer_ledger.records:
                    self.transfer_ledger.record(
                        kind=record.kind,
                        direction=record.direction,
                        tick=record.tick,
                        nbytes=record.bytes,
                        source_stream=record.source_stream,
                        synchronization=record.synchronization,
                        scheduled=record.scheduled,
                        graph_compatible=record.graph_compatible,
                        reason=record.reason,
                    )
            run.close(checkpoint=False)
        return result

    def _execute_task(self, task: tuple[Any, ...]) -> BranchResult:
        if self.template_run.ds.is_gpu:
            device = self.template_run.ds.xp.cuda.Device(
                int(self.template_run.ds.xp.cuda.Device().id)
            )
            with device:
                return self._execute_branch(*task[:-1], anchor=bool(task[-1]))
        return self._execute_branch(*task[:-1], anchor=bool(task[-1]))

    def run_source(self, collected: CollectedSource) -> CounterfactualRunResult:
        backend = collected.state.backend
        source_hash = hash_state(collected.state)
        collected.state.source_root = source_hash.root
        ids = collected.decisions.materialize_ids(backend)
        legal = backend.asnumpy(collected.decisions.policy_legal)
        executable = backend.asnumpy(collected.decisions.prechoice_executable)
        reasons = backend.asnumpy(collected.decisions.candidate_reason)
        selected = backend.asnumpy(collected.decisions.selected_action)
        tasks: list[tuple[Any, ...]] = []
        nonexecutable: list[NonExecutableAttempt] = []

        for decision_index, decision_id in enumerate(ids):
            selected_action = int(selected[decision_index])
            if bool(self.cfg.counterfactual.include_selected_anchor) and bool(
                executable[decision_index, selected_action]
            ):
                tasks.append(
                    (
                        collected,
                        decision_index,
                        decision_id,
                        selected_action,
                        -1,
                        int(self.cfg.world.seed),
                        True,
                    )
                )
            for repeat_index in range(int(self.cfg.counterfactual.repeats)):
                seed = branch_seed(
                    int(self.cfg.world.seed), collected.state.source_state_id, repeat_index
                )
                for action in range(len(Action)):
                    if not bool(executable[decision_index, action]):
                        if bool(self.cfg.counterfactual.emit_nonexecutable_candidates):
                            missing_id = branch_id(
                                collected.state.source_state_id,
                                decision_id,
                                repeat_index,
                                action,
                                seed,
                                "nonexecutable",
                            )
                            nonexecutable.append(
                                NonExecutableAttempt(
                                    collected.state.source_state_id,
                                    decision_id,
                                    missing_id,
                                    repeat_index,
                                    seed,
                                    selected_action,
                                    action,
                                    bool(legal[decision_index, action]),
                                    False,
                                    int(reasons[decision_index, action]),
                                )
                            )
                        continue
                    tasks.append(
                        (
                            collected,
                            decision_index,
                            decision_id,
                            action,
                            repeat_index,
                            seed,
                            False,
                        )
                    )
        workers = 1
        if backend.is_gpu:
            memory_safe_limit = (
                int(self.active_branch_limit)
                if self.active_branch_limit is not None
                else int(self.cfg.counterfactual.max_active_branches)
            )
            workers = min(
                int(self.cfg.counterfactual.stream_lanes),
                memory_safe_limit,
                len(tasks),
            )
        self.last_worker_count = workers
        if workers > 1:
            with ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="owl-counterfactual-lane",
            ) as executor:
                branches = list(executor.map(self._execute_task, tasks))
        else:
            branches = [self._execute_task(task) for task in tasks]
        pairs: list[CandidatePair] = []
        grouped: dict[tuple[str, int], list[BranchResult]] = {}
        for branch in branches:
            if branch.anchor or branch.status != BranchStatus.COMPLETED:
                continue
            grouped.setdefault((branch.source_decision_id, branch.repeat_index), []).append(branch)
        for (decision_id, repeat_index), group in grouped.items():
            ordered = sorted(group, key=lambda item: item.forced_action)
            for left_index, left in enumerate(ordered):
                for right in ordered[left_index + 1 :]:
                    for horizon in sorted(set(left.outcomes) & set(right.outcomes)):
                        pairs.append(
                            CandidatePair(
                                pair_id(
                                    decision_id,
                                    repeat_index,
                                    left.forced_action,
                                    right.forced_action,
                                    horizon,
                                ),
                                decision_id,
                                repeat_index,
                                left.forced_action,
                                right.forced_action,
                                left.branch_id,
                                right.branch_id,
                                horizon,
                            )
                        )
        return CounterfactualRunResult(
            collected.state.source_state_id,
            source_hash,
            branches,
            nonexecutable,
            pairs,
        )
