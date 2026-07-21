from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from owl.core.advanced import ensure_action_transition_fields, ensure_advanced_fields
from owl.core.config import SimulationConfig, load_config
from owl.core.init import initialize_world
from owl.core.state import WorldState
from owl.gpu.command_apply import apply_command
from owl.gpu.commands import CommandKind, GPUCommandQueue
from owl.gpu.device_state import OWLDeviceState
from owl.gpu.epochs import FieldEpochs
from owl.gpu.graph_inputs import DeviceCommandBuffer, apply_device_commands, warm_command_kernel
from owl.gpu.graph_safety import (
    CaptureAllocationGuard,
    build_default_graph_safety_manifest,
)
from owl.gpu.graph_static_actions import (
    apply_movement_graph_static,
    apply_reproduction_graph_static,
    apply_topology_graph_static,
    ensure_graph_static_action_buffers,
)
from owl.gpu.graphs import GpuTickGraphManager
from owl.gpu.invariants import assert_invariant_summary, invariant_summary_from_metric
from owl.gpu.kernels.stencil_kernels import warm_stencil_kernels
from owl.gpu.metrics_slab import DeviceMetricSlab
from owl.gpu.numerical_ledger import NumericalLedger
from owl.gpu.nvtx import nvtx_range
from owl.gpu.profiler import GPUFullProfiler
from owl.gpu.qiskit_transfer import pack_qiskit_rows, unpack_qiskit_rows
from owl.gpu.scratch import ScratchManager
from owl.gpu.shadow_audit import CPUShadowAuditor
from owl.gpu.slabs import FieldSlabManager
from owl.gpu.stages.action_transitions_gpu import (
    apply_active_sense_transition_gpu,
    compile_selected_action_transition_gpu,
    prepare_action_transition_context_gpu,
)
from owl.gpu.stages.aggregation_gpu import (
    aggregate_global_gpu as aggregate_global_gpu,
)
from owl.gpu.stages.aggregation_gpu import (
    aggregate_patches_gpu,
)
from owl.gpu.stages.authority_gpu import compute_authority_gpu
from owl.gpu.stages.collision_gpu import apply_inhibition_gpu, resolve_collisions_gpu
from owl.gpu.stages.communication_gpu import (
    emit_signals_gpu,
    update_channel_trust_gpu,
    update_signal_memory_gpu,
)
from owl.gpu.stages.death_gpu import apply_death_gpu
from owl.gpu.stages.environment_gpu import update_environment_gpu
from owl.gpu.stages.feeding_gpu import apply_feeding_gpu
from owl.gpu.stages.health_gpu import (
    apply_metabolism_damage_gpu,
    apply_repair_and_integrate_gpu,
    clip_life_fields_gpu,
)
from owl.gpu.stages.integration_gpu import update_integration_gpu
from owl.gpu.stages.memory_gpu import update_memory_gpu
from owl.gpu.stages.movement_gpu import apply_movement_gpu
from owl.gpu.stages.phase_gpu import (
    compute_cell_coherence_gpu as compute_cell_coherence_gpu,
)
from owl.gpu.stages.phase_gpu import (
    compute_cross_scale_coupling_gpu as compute_cross_scale_coupling_gpu,
)
from owl.gpu.stages.phase_gpu import (
    compute_local_synchrony_gpu as compute_local_synchrony_gpu,
)
from owl.gpu.stages.phase_gpu import (
    update_phase_gpu,
)
from owl.gpu.stages.raqic_gpu_stage import (
    build_raqic_dense_batch_from_device,
    ensure_actualization_graph_buffers_gpu,
    quiesce_dead_raqic_fields_gpu,
    run_raqic_gpu_stage,
)
from owl.gpu.stages.reproduction_gpu import apply_reproduction_gpu
from owl.gpu.stages.scientific_snapshots_gpu import (
    capture_pre_decision_state_gpu,
    capture_tick_start_gpu,
    ensure_scientific_snapshot_buffers,
)
from owl.gpu.stages.sensing_gpu import compute_sensing_bundle_gpu, prepare_sensing_stencil_scratch
from owl.gpu.stages.topdown_gpu import (
    apply_threshold_modulation_gpu as apply_threshold_modulation_gpu,
)
from owl.gpu.stages.topdown_gpu import (
    dispatch_parent_context_gpu as dispatch_parent_context_gpu,
)
from owl.gpu.stages.topology_gpu import apply_topology_events_gpu, detect_topology_events_gpu
from owl.gpu.stages.utility_gpu import compute_utilities_gpu
from owl.gpu.streams import StreamBundle, TransferTicket
from owl.gpu.transfer_ledger import TransferLedger
from owl.raqic.state import ensure_raqic_fields, quiesce_dead_raqic_fields
from owl.record.gpu_async_writer import AsyncGPUWriter
from owl.record.gpu_recording_policy import GPURecordingPolicy
from owl.runtime.module_enablement import ModuleEnablement
from owl.runtime.run_paths import RunPaths, derive_run_paths
from owl_raqic.qiskit_backend.per_ow_executor import PerOWQiskitExecutor
from owl_raqic.qiskit_backend.validation_manager import QiskitValidationManager

_GRAPH_SEGMENTS = ("predecision", "decision", "actions", "postdecision")


def _array_pointer(value: Any) -> int | None:
    data = getattr(value, "data", None)
    ptr = getattr(data, "ptr", None)
    if ptr is not None:
        return int(ptr)
    interface = getattr(value, "__array_interface__", None)
    if interface and interface.get("data"):
        return int(interface["data"][0])
    return None


def _persistent_pointer_snapshot(
    ds: Any, scratch: Any, slab_manager: Any, command_buffer: Any
) -> dict[str, int]:
    """Snapshot addresses that must remain stable across CUDA graph capture."""
    out: dict[str, int] = {}
    groups = (
        ("state", getattr(ds, "arrays", {})),
        ("patch", getattr(ds, "patch_arrays", {})),
        ("global", getattr(ds, "global_arrays", {})),
        ("scratch", getattr(scratch, "buffers", {})),
        ("slab", {} if slab_manager is None else slab_manager.slabs),
        (
            "cadc",
            {}
            if ds.metadata.get("cadc_device_buffer") is None
            else ds.metadata["cadc_device_buffer"].pointer_arrays(),
        ),
    )
    for prefix, mapping in groups:
        for name, value in mapping.items():
            pointer = _array_pointer(value)
            if pointer is not None:
                out[f"{prefix}.{name}"] = pointer
    if command_buffer is not None:
        for name, value in vars(command_buffer).items():
            pointer = _array_pointer(value)
            if pointer is not None:
                out[f"command.{name}"] = pointer
    return out


@dataclass
class PersistentOWLDeviceRun:
    """Persistent device-resident OWL + RAQIC run context.

    The device state owns the hot simulation arrays. CPU state is a checkpoint,
    recording, and audit mirror. Execution-plan metadata is retained so the
    primary CLI and dedicated GPU entry points use the same semantics.
    """

    cfg: SimulationConfig
    state: WorldState
    ds: OWLDeviceState
    profiler: GPUFullProfiler
    scratch: ScratchManager
    streams: StreamBundle
    graph_manager: GpuTickGraphManager
    metric_slab: DeviceMetricSlab
    transfer_ledger: TransferLedger
    numerical_ledger: NumericalLedger
    command_queue: GPUCommandQueue
    device_command_buffer: DeviceCommandBuffer | None = None
    plan: Any = None
    module_enablement: ModuleEnablement | None = None
    slab_manager: FieldSlabManager | None = None
    qiskit_validation: QiskitValidationManager | None = None
    per_ow_qiskit: PerOWQiskitExecutor | None = None
    shadow_auditor: CPUShadowAuditor | None = None
    visual_controller: Any = None
    async_writer: AsyncGPUWriter | None = None
    recording_policy: GPURecordingPolicy | None = None
    cadc_buffer: Any = None
    counterfactual_source_observer: Any = None
    metrics: list[dict[str, Any]] = field(default_factory=list)
    pending_metric_tickets: list[TransferTicket] = field(default_factory=list)
    last_diagnostics: dict[str, Any] = field(default_factory=dict)
    fallback_count: int = 0
    closed: bool = False
    paused: bool = False
    checkpoint_requested: bool = False
    validation_requested: bool = False
    visual_settings: dict[str, Any] = field(default_factory=dict)
    _steps_completed: int = 0
    _checkpointed_tick: int | None = None
    _checkpoint_counted_tick: int | None = None
    _last_checkpoint: WorldState | None = None
    checkpoint_count: int = 0
    memory_plan: Any = None
    run_paths: RunPaths | None = None

    @classmethod
    def from_config(
        cls,
        cfg: SimulationConfig,
        initial_state: WorldState | None = None,
        *,
        plan: Any | None = None,
        force_backend: str | None = None,
        output_root: str | Path | None = None,
        counterfactual_observer: Any | None = None,
    ) -> PersistentOWLDeviceRun:
        if plan is None:
            from owl.runtime.capabilities import detect_runtime_capabilities
            from owl.runtime.execution_plan import compile_execution_plan

            plan = compile_execution_plan(cfg, detect_runtime_capabilities())
        if bool(cfg.counterfactual.enabled) and counterfactual_observer is None:
            raise ValueError(
                "enabled counterfactual collection requires an explicit, "
                "source-hash-bound decision observer"
            )

        rng = np.random.default_rng(cfg.world.seed)
        state = initialize_world(cfg, rng) if initial_state is None else initial_state
        ensure_advanced_fields(state, cfg)
        ensure_action_transition_fields(state, cfg)
        if getattr(cfg.raqic, "enabled", False):
            ensure_raqic_fields(state, cfg)

        strict = bool(getattr(cfg.raqic, "full_gpu_strict", True))
        allow = bool(getattr(plan, "fallback_allowed", False))
        ds = OWLDeviceState.from_world_state(
            state,
            cfg,
            strict=strict,
            allow_fallback=allow,
            force_backend=force_backend,
        )
        ds.metadata["defer_host_metrics"] = True
        ds.metadata["field_epochs"] = FieldEpochs()
        graph_static = bool(
            getattr(plan, "simulation_backend", "") == "gpu_graph"
            and getattr(plan, "graph_requirement", "allow_partial") == "full_tick"
        )
        if graph_static and bool(cfg.action_transitions.enabled):
            raise RuntimeError(
                "owl.action-transitions.v1 requires the persistent segmented GPU tier; "
                "full-tick CUDA graph execution is fail-closed until its movement command "
                "buffer carries selected and compiled action identities separately"
            )
        ds.metadata["graph_static"] = graph_static
        # Allocate coordinate-resident scientific snapshots before any graph
        # capture or slab attachment.
        ensure_scientific_snapshot_buffers(ds)

        cadc_buffer = None
        if bool(getattr(cfg.recording.cadc, "enabled", False)):
            from owl.record.cadc_device_buffer import CADCDeviceBuffer

            cadc_buffer = CADCDeviceBuffer.create(ds, cfg)
            ds.metadata["cadc_device_buffer"] = cadc_buffer

        if ds.is_gpu:
            warm_stencil_kernels(ds.xp)
            if graph_static:
                warm_command_kernel(ds.xp)

        scratch = ScratchManager.for_config(ds.backend, cfg)
        if getattr(cfg.raqic, "full_gpu_memory_policy", "elastic") == "preallocate":
            scratch.allocate_all()
        streams = StreamBundle.create(ds.backend)
        profiler = GPUFullProfiler(
            ds.backend,
            streams.compute,
            enabled=bool(getattr(plan.runtime_settings, "profile_enabled", True)),
        )
        graph_mode = (
            "full_tick"
            if getattr(plan, "simulation_backend", "") == "gpu_graph"
            else getattr(cfg.raqic, "full_gpu_graph_mode", "off")
        )
        graph_manager = GpuTickGraphManager(
            ds.backend,
            mode=str(graph_mode),
            requirement=str(getattr(plan, "graph_requirement", "allow_partial")),
            required_segments=_GRAPH_SEGMENTS,
            safety_manifest=build_default_graph_safety_manifest(),
            allocation_guard=(
                CaptureAllocationGuard(ds.xp) if ds.is_gpu and graph_mode != "off" else None
            ),
        )
        metric_slab = DeviceMetricSlab.create(ds.backend)
        slab_manager = None
        if bool(getattr(cfg.raqic, "full_gpu_fuse_scatter", False)):
            slab_manager = FieldSlabManager.attach(ds)
        if graph_static:
            if slab_manager is None:
                raise RuntimeError("full-tick graph requires persistent field slabs")
            ensure_graph_static_action_buffers(ds, cfg)
            ensure_actualization_graph_buffers_gpu(ds, cfg)

        transfer_ledger = TransferLedger()
        ds.metadata["transfer_ledger"] = transfer_ledger
        validation_manager = None
        qiskit_policy = getattr(plan, "qiskit_policy", None)
        if (
            bool(getattr(cfg.raqic, "gpu_validate_qiskit", False))
            or int(getattr(cfg.raqic, "full_gpu_validation_every", 0)) > 0
            or (qiskit_policy is not None and qiskit_policy.validation_only)
        ):
            validation_manager = QiskitValidationManager.from_config(
                cfg, transfer_ledger=transfer_ledger
            )

        per_ow_executor = None
        if qiskit_policy is not None and qiskit_policy.per_ow:
            per_ow_executor = PerOWQiskitExecutor(
                qiskit_policy,
                seed=int(cfg.world.seed),
            )

        shadow = None
        shadow_ticks = tuple(getattr(plan, "cpu_shadow_ticks", ()))
        if shadow_ticks:
            shadow = CPUShadowAuditor(
                cfg,
                ticks=shadow_ticks,
                tolerance=float(getattr(cfg.raqic, "full_gpu_shadow_tolerance", 1e-8)),
                strict=bool(getattr(cfg.raqic, "full_gpu_shadow_strict", True)),
                reference_mode=str(
                    getattr(
                        cfg.raqic,
                        "full_gpu_shadow_reference",
                        "scientific_cpu",
                    )
                ),
            )

        visual = None
        if getattr(plan, "visual_backend", "none") != "none":
            from owl.viz.controller import VisualController

            visual = VisualController.from_config(
                cfg,
                backend_name=str(plan.visual_backend),
            )

        recording_policy = (
            GPURecordingPolicy.from_config(cfg)
            if bool(getattr(cfg.recording, "enabled", False))
            else None
        )
        device_command_buffer = (
            DeviceCommandBuffer.create(
                ds.backend,
                int(cfg.raqic.full_gpu_command_capacity),
            )
            if graph_static
            else None
        )
        if graph_manager.mode != "off":
            graph_manager.pointer_snapshot = lambda: _persistent_pointer_snapshot(
                ds, scratch, slab_manager, device_command_buffer
            )

        run_paths = derive_run_paths(
            cfg=cfg,
            plan=plan,
            root=(output_root if output_root is not None else "runs"),
        )

        run = cls(
            cfg=cfg,
            state=state,
            ds=ds,
            profiler=profiler,
            scratch=scratch,
            streams=streams,
            graph_manager=graph_manager,
            metric_slab=metric_slab,
            transfer_ledger=transfer_ledger,
            numerical_ledger=NumericalLedger.from_config(cfg),
            command_queue=GPUCommandQueue(int(cfg.raqic.full_gpu_command_capacity)),
            device_command_buffer=device_command_buffer,
            plan=plan,
            module_enablement=ModuleEnablement(frozenset(plan.enabled_modules)),
            slab_manager=slab_manager,
            qiskit_validation=validation_manager,
            per_ow_qiskit=per_ow_executor,
            shadow_auditor=shadow,
            visual_controller=visual,
            recording_policy=recording_policy,
            cadc_buffer=cadc_buffer,
            counterfactual_source_observer=counterfactual_observer,
            run_paths=run_paths,
        )
        run.fallback_count = 0 if ds.is_gpu else 1
        # A restored checkpoint resumes cadence, graph warm-up, recording, and
        # validation from the scientific tick rather than pretending to be a
        # new run at step zero. This keeps restart behavior invariant.
        run._steps_completed = int(state.tick)
        run._memory_preflight()
        run._prepare_graph_segments()
        if cfg.recording.enabled:
            path = run_paths.reports / "metrics.gpu.jsonl"
            run.async_writer = AsyncGPUWriter(
                path,
                max_queue=int(cfg.raqic.full_gpu_writer_queue_capacity),
                overflow_policy=str(cfg.raqic.full_gpu_writer_overflow_policy),
            ).start()
        return run

    def __enter__(self) -> PersistentOWLDeviceRun:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close(checkpoint=False)

    def _memory_preflight(self) -> None:
        from owl.gpu.memory_model import build_memory_plan

        plan = build_memory_plan(
            self.ds,
            self.cfg,
            scratch_bytes=int(self.scratch.spec_bytes()),
            slab_layout=None if self.slab_manager is None else self.slab_manager.layout(),
            qiskit_policy=getattr(self.plan, "qiskit_policy", None),
            visual_backend=getattr(self.plan, "visual_backend", "none"),
        )
        self.memory_plan = plan
        configured_limit = getattr(self.cfg.raqic, "gpu_memory_limit_mb", None)
        candidates: list[int] = []
        if configured_limit is not None:
            candidates.append(int(float(configured_limit) * 1024 * 1024))
        if self.ds.is_gpu:
            free = self.ds.backend.info.get("free_memory")
            if free:
                candidates.append(
                    int(float(self.cfg.raqic.full_gpu_memory_safety_fraction) * int(free))
                )
        plan.allowed_bytes = min(candidates) if candidates else plan.allowed_bytes
        plan.evaluate()
        self.ds.metadata["memory_preflight"] = plan.to_dict()
        if not bool(getattr(self.cfg.raqic, "full_gpu_memory_preflight", True)):
            return
        if not plan.passed:
            raise MemoryError(
                f"memory preflight peak {plan.peak_bytes:,} exceeds allowed "
                f"memory {plan.allowed_bytes:,} bytes"
            )

    def _graph_safe(self, segment: str) -> bool:
        if not self.ds.is_gpu:
            return False
        if self.graph_manager.mode == "off":
            return False
        full_tick = self.graph_manager.requirement == "full_tick"
        if full_tick:
            # Full-tick capture has a stricter static replay contract than
            # partial/experimental capture. All persistent storage and fixed
            # action/event buffers must exist before the first capture.
            if not bool(self.ds.metadata.get("graph_static", False)):
                return False
            if self.slab_manager is None or self.device_command_buffer is None:
                return False
            if str(getattr(self.cfg.raqic, "full_gpu_memory_policy", "")) != "preallocate":
                return False
        if (
            self.graph_manager.safety_manifest is not None
            and not self.graph_manager.safety_manifest.segment_approved(segment)
        ):
            return False
        if self.per_ow_qiskit is not None and segment == "decision":
            return False
        if self.counterfactual_source_observer is not None and segment == "decision":
            return False
        phase_is_canonical = (
            str(getattr(self.cfg.raqic, "full_gpu_phase_mode", "")) == "canonical_device"
        )
        return segment != "decision" or phase_is_canonical

    def _prepare_graph_segments(self) -> None:
        self.graph_manager.prepare_segments(
            {
                "predecision": (self._segment_predecision, self._graph_safe("predecision")),
                "decision": (self._segment_decision, self._graph_safe("decision")),
                "actions": (self._segment_actions, self._graph_safe("actions")),
                "postdecision": (self._segment_postdecision, self._graph_safe("postdecision")),
            }
        )

    def _execute_segment(self, name: str) -> None:
        rec = self.graph_manager.segments[name]
        if self.graph_manager.mode == "off" or not rec.graph_safe:
            self.graph_manager.replay_or_run_segment(name, stream=self.streams.compute)
            return
        if rec.graph is not None:
            self.graph_manager.replay_segment(name, stream=self.streams.compute)
            return
        warmup = int(getattr(self.cfg.raqic, "full_gpu_graph_warmup_ticks", 0))
        if self._steps_completed < warmup:
            self.graph_manager.replay_or_run_segment(name, stream=self.streams.compute)
            return
        captured = self.graph_manager.capture_segment(
            name,
            stream=self.streams.compute,
        )
        if not captured:
            reason = rec.reason
            if self.graph_manager.requirement == "full_tick" or not bool(
                getattr(self.cfg.raqic, "full_gpu_graph_allow_fallback", False)
            ):
                raise RuntimeError(f"CUDA graph capture for segment {name!r} failed: {reason}")
            self.graph_manager.replay_or_run_segment(name, stream=self.streams.compute)

    def _module(self, name: str) -> bool:
        return self.module_enablement is None or self.module_enablement.has(name)

    def _segment_predecision(self) -> None:
        cfg, ds = self.cfg, self.ds
        if self.device_command_buffer is not None:
            apply_device_commands(ds, self.device_command_buffer)
        # CPU captures prev_resource/health/integration before environment and
        # before cells move. These arrays are coordinate-resident.
        capture_tick_start_gpu(ds, cfg)
        if self.cadc_buffer is not None:
            from owl.record.cadc_capture import capture_stage_before, capture_tick_open

            capture_tick_open(self.cadc_buffer, ds)
            capture_stage_before(self.cadc_buffer, ds)
        if self._module("environment"):
            update_environment_gpu(ds, cfg)
        if self.cadc_buffer is not None:
            from owl.record.cadc_capture import capture_stage_after
            from owl.record.cadc_schema import ContributionCode

            capture_stage_after(self.cadc_buffer, ds, ContributionCode.ENVIRONMENT)
        sensing_scratch = None
        if self._module("sensing"):
            sensing_scratch = prepare_sensing_stencil_scratch(ds, cfg, self.scratch)
            compute_sensing_bundle_gpu(ds, cfg, sensing_scratch)
        prepare_action_transition_context_gpu(ds, cfg)
        if self.cadc_buffer is not None:
            from owl.record.cadc_capture import capture_agent_oracle_context

            capture_agent_oracle_context(self.cadc_buffer, ds, cfg)
        if self._module("aggregation"):
            aggregate_patches_gpu(ds, cfg)
            aggregate_global_gpu(ds, cfg, force=False)
        if self._module("topdown"):
            dispatch_parent_context_gpu(ds, cfg)
            apply_threshold_modulation_gpu(ds, cfg)
        if self._module("phase"):
            update_phase_gpu(ds, cfg)
            compute_local_synchrony_gpu(ds, cfg)
            compute_cell_coherence_gpu(ds, cfg)
            compute_cross_scale_coupling_gpu(ds, cfg)
        if self._module("utility"):
            compute_utilities_gpu(ds, cfg)
        if self._module("authority"):
            compute_authority_gpu(ds, cfg)
        capture_pre_decision_state_gpu(ds, cfg)
        if self.cadc_buffer is not None:
            from owl.record.cadc_capture import capture_prechoice_candidates

            capture_prechoice_candidates(self.cadc_buffer, ds, cfg)

    def _apply_per_ow_qiskit(self) -> dict[str, Any] | None:
        if self.per_ow_qiskit is None:
            return None
        ds, cfg = self.ds, self.cfg
        batch = build_raqic_dense_batch_from_device(ds, cfg)
        if batch.n == 0:
            return {
                "expected_count": 0,
                "processed_count": 0,
                "all_ow_accounted": True,
            }
        packed_device, packed_layout = pack_qiskit_rows(
            ds.xp,
            probabilities=ds.raqic_probabilities[batch.yx[:, 0], batch.yx[:, 1], :],
            phases=ds.raqic_phase[batch.yx[:, 0], batch.yx[:, 1], :],
            authority=batch.authority_mask,
            parent=None,
            ow_ids=batch.ow_id,
        )
        packed_host = ds.backend.asnumpy(packed_device)
        unpacked = unpack_qiskit_rows(packed_host, packed_layout)
        p = unpacked.probabilities
        phase = unpacked.phases
        authority = unpacked.authority
        ow_ids = unpacked.ow_ids
        if phase is None or authority is None:
            raise RuntimeError("authoritative Qiskit slab omitted phase or authority")
        self.transfer_ledger.record_d2h(
            packed_layout.total_bytes,
            kind="qiskit",
            tick=int(ds.tick),
            source_stream="qiskit-authoritative",
            synchronization="device",
            scheduled=True,
            graph_compatible=False,
            reason="bounded authoritative per-OW Qiskit slab",
        )
        result = self.per_ow_qiskit.execute(
            p,
            phase,
            authority,
            ow_ids,
            tick=int(ds.tick),
            tolerance=float(cfg.raqic.gpu_probability_tolerance),
        )
        yy, xx = batch.yx[:, 0], batch.yx[:, 1]
        q = ds.backend.asarray(result.authoritative.probabilities)
        readout = ds.backend.asarray(result.authoritative.readouts)
        ds.arrays["raqic_probabilities"][yy, xx, :] = q
        ds.arrays["possibility"][yy, xx, :] = q.astype(ds.possibility.dtype)
        ds.arrays["raqic_readout"][yy, xx] = readout
        ds.arrays["readout"][yy, xx] = readout.astype(ds.readout.dtype)
        payload = result.to_dict()
        ds.metadata["last_per_ow_qiskit"] = payload
        return payload

    def _segment_decision(self) -> None:
        # Replay-varying Python metadata must not be written inside a captured
        # callback. Device arrays/counters are collected after segment replay.
        run_raqic_gpu_stage(self.ds, self.cfg)
        if self.per_ow_qiskit is not None:
            self._apply_per_ow_qiskit()
        if self.cadc_buffer is not None:
            from owl.record.cadc_capture import capture_selected_intent

            capture_selected_intent(self.cadc_buffer, self.ds, self.cfg)
        if self.counterfactual_source_observer is not None:
            decision_event = self.streams.record(self.streams.compute)
            self.counterfactual_source_observer.capture(self, decision_event)

    def _segment_actions(self) -> None:
        cfg, ds = self.cfg, self.ds
        compile_selected_action_transition_gpu(ds, cfg)
        graph_static = bool(ds.metadata.get("graph_static", False))
        capture_before = capture_after = None
        if self.cadc_buffer is not None:
            from owl.record.cadc_capture import capture_stage_after, capture_stage_before

            capture_before = capture_stage_before
            capture_after = capture_stage_after
        if self._module("movement"):
            if capture_before is not None:
                capture_before(self.cadc_buffer, ds)
            if graph_static:
                apply_movement_graph_static(ds, cfg)
            else:
                apply_movement_gpu(ds, cfg)
            if capture_after is not None:
                from owl.record.cadc_schema import ContributionCode

                capture_after(self.cadc_buffer, ds, ContributionCode.MOVEMENT)
        if self._module("collision"):
            if capture_before is not None:
                capture_before(self.cadc_buffer, ds)
            resolve_collisions_gpu(ds, cfg)
            apply_inhibition_gpu(ds, cfg)
            if capture_after is not None:
                from owl.core.actions import Action
                from owl.record.cadc_schema import ContributionCode

                capture_after(
                    self.cadc_buffer,
                    ds,
                    ContributionCode.COLLISION_INHIBITION,
                    actions=(int(Action.INHIBIT),),
                )
        if self._module("feeding"):
            if capture_before is not None:
                capture_before(self.cadc_buffer, ds)
            apply_feeding_gpu(ds, cfg)
            if capture_after is not None:
                from owl.core.actions import Action
                from owl.record.cadc_schema import ContributionCode

                capture_after(
                    self.cadc_buffer,
                    ds,
                    ContributionCode.FEEDING,
                    actions=(int(Action.FEED),),
                )
        if self._module("health"):
            if capture_before is not None:
                capture_before(self.cadc_buffer, ds)
            apply_repair_and_integrate_gpu(ds, cfg)
            if capture_after is not None:
                from owl.core.actions import Action
                from owl.record.cadc_schema import ContributionCode

                capture_after(
                    self.cadc_buffer,
                    ds,
                    ContributionCode.REPAIR_INTEGRATION,
                    actions=(int(Action.INTEGRATE), int(Action.REPAIR)),
                )
        if self._module("communication"):
            if capture_before is not None:
                capture_before(self.cadc_buffer, ds)
            emit_signals_gpu(ds, cfg)
            if capture_after is not None:
                from owl.core.actions import Action
                from owl.record.cadc_schema import ContributionCode

                capture_after(
                    self.cadc_buffer,
                    ds,
                    ContributionCode.COMMUNICATION,
                    actions=(int(Action.COMMUNICATE),),
                )
        if self._module("reproduction"):
            if capture_before is not None:
                capture_before(self.cadc_buffer, ds)
            if graph_static:
                reproduction_result = apply_reproduction_graph_static(ds, cfg)
            else:
                reproduction_result = apply_reproduction_gpu(ds, cfg)
            if capture_after is not None:
                from owl.record.cadc_capture import capture_reproduction_execution
                from owl.record.cadc_schema import ContributionCode

                capture_after(
                    self.cadc_buffer,
                    ds,
                    ContributionCode.REPRODUCTION,
                )
                capture_reproduction_execution(self.cadc_buffer, ds, reproduction_result)
        if self._module("topology"):
            if capture_before is not None:
                capture_before(self.cadc_buffer, ds)
            if graph_static:
                topology_result = apply_topology_graph_static(ds, cfg)
                events = topology_result.get("_cadc_events")
            else:
                events = detect_topology_events_gpu(ds, cfg)
                apply_topology_events_gpu(ds, cfg, events)
            if capture_after is not None:
                from owl.record.cadc_capture import capture_topology_execution
                from owl.record.cadc_schema import ContributionCode

                capture_after(
                    self.cadc_buffer,
                    ds,
                    ContributionCode.TOPOLOGY,
                )
                if events is not None:
                    capture_topology_execution(self.cadc_buffer, ds, events)
        if capture_before is not None and bool(cfg.action_transitions.enabled):
            capture_before(self.cadc_buffer, ds)
        active_sense_result = apply_active_sense_transition_gpu(ds, cfg)
        if self.cadc_buffer is not None and bool(cfg.action_transitions.enabled):
            from owl.record.cadc_capture import capture_action_transition_execution
            from owl.record.cadc_schema import ContributionCode

            capture_after(self.cadc_buffer, ds, ContributionCode.ACTIVE_SENSE)
            capture_action_transition_execution(self.cadc_buffer, ds, cfg, active_sense_result)

    def _segment_postdecision(self) -> None:
        cfg, ds = self.cfg, self.ds
        capture_before = capture_after = None
        if self.cadc_buffer is not None:
            from owl.record.cadc_capture import capture_stage_after, capture_stage_before

            capture_before = capture_stage_before
            capture_after = capture_stage_after
        if self._module("health"):
            if capture_before is not None:
                capture_before(self.cadc_buffer, ds)
            apply_metabolism_damage_gpu(ds, cfg)
            if capture_after is not None:
                from owl.record.cadc_schema import ContributionCode

                capture_after(self.cadc_buffer, ds, ContributionCode.METABOLISM_TOXIN)
        if capture_before is not None:
            capture_before(self.cadc_buffer, ds)
        if self._module("memory"):
            update_memory_gpu(ds, cfg)
        if self._module("communication"):
            update_signal_memory_gpu(ds, cfg)
        if self._module("integration"):
            update_integration_gpu(ds, cfg)
        if self._module("communication"):
            update_channel_trust_gpu(ds, cfg)
        if capture_after is not None:
            from owl.record.cadc_schema import ContributionCode

            capture_after(self.cadc_buffer, ds, ContributionCode.MEMORY_TRUST)
            from owl.record.cadc_capture import capture_information_post_memory

            capture_information_post_memory(self.cadc_buffer, ds)
        if self._module("death"):
            if capture_before is not None:
                capture_before(self.cadc_buffer, ds)
            apply_death_gpu(ds, cfg)
            if capture_after is not None:
                from owl.record.cadc_schema import ContributionCode

                capture_after(self.cadc_buffer, ds, ContributionCode.DEATH_CLEANUP)
        if self._module("health"):
            if capture_before is not None:
                capture_before(self.cadc_buffer, ds)
            clip_life_fields_gpu(ds, cfg)
            if capture_after is not None:
                from owl.record.cadc_schema import ContributionCode

                capture_after(self.cadc_buffer, ds, ContributionCode.CLIPPING)
        if getattr(cfg.raqic, "enabled", False):
            # CPU ``step`` quiesces every dead/obstacle RAQIC row after physical
            # clipping. This includes movement-vacated source cells that are no
            # longer "present" and therefore are not rediscovered by death.
            quiesce_dead_raqic_fields_gpu(ds)
        if self._module("aggregation"):
            aggregate_patches_gpu(ds, cfg)
            aggregate_global_gpu(ds, cfg, force=True)
        if self._module("topdown"):
            # CPU ``_post_state_refresh`` recomputes apex intention and policy
            # after every completed tick, independently of the predecision
            # apex cadence. Preserve that scientific epoch here.
            dispatch_parent_context_gpu(ds, cfg, force_global=True)
        if self.cadc_buffer is not None:
            from owl.record.cadc_capture import finalize_tick_reconciliation

            finalize_tick_reconciliation(self.cadc_buffer, ds)
        if self.counterfactual_source_observer is not None:
            self.counterfactual_source_observer.after_postdecision(self)

    def _device_replay_diagnostics(self) -> dict[str, Any]:
        """Return graph-safe diagnostic references collected outside capture."""
        arrays = self.ds.arrays
        return {
            "raqic": {
                "eligible_device": arrays.get("_raqic_eligible_count"),
                "repaired_device": arrays.get("_raqic_repaired_count"),
                "phase_mode": str(getattr(self.cfg.raqic, "full_gpu_phase_mode", "")),
            },
            "movement": {"moved_device": arrays.get("_graph_moved_count")},
            "reproduction": {"children_device": arrays.get("_graph_children_count")},
            "topology": {"events_device": arrays.get("_graph_topology_count")},
        }

    def _apply_commands(self) -> list[dict[str, Any]]:
        commands = self.command_queue.drain()
        if self.device_command_buffer is None:
            return [apply_command(self, cmd) for cmd in commands]
        state_commands = [
            cmd
            for cmd in commands
            if cmd.kind in {CommandKind.INJECT_FOOD, CommandKind.INJECT_TOXIN}
        ]
        control_commands = [
            cmd
            for cmd in commands
            if cmd.kind not in {CommandKind.INJECT_FOOD, CommandKind.INJECT_TOXIN}
        ]
        results = [apply_command(self, cmd) for cmd in control_commands]
        results.extend(
            self.device_command_buffer.encode(
                state_commands,
                self.ds.backend,
            )
        )
        return results

    def _step_device(
        self,
        command_results: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        cfg, ds = self.cfg, self.ds
        self.profiler.reset()
        ds.tick = int(ds.tick) + 1
        diagnostics: dict[str, Any] = {
            "mode": cfg.raqic.mode,
            "execution_tier": getattr(
                self.plan,
                "simulation_backend",
                getattr(cfg.raqic, "full_gpu_execution_tier", "persistent"),
            ),
            "backend": ds.backend.name,
            "fallback": not ds.is_gpu,
            "persistent": True,
            "commands": list(command_results or []),
        }

        with self.streams.compute:
            for segment in _GRAPH_SEGMENTS:
                with self.profiler.stage(segment), nvtx_range(f"owl.tick.{segment}"):
                    self._execute_segment(segment)
            ds.metadata["field_epochs"].tick(
                "food", "toxin", "signal", "phase", "health", "occupancy", "resource"
            )
            metric_due = ds.tick % int(cfg.raqic.full_gpu_metric_every) == 0
            invariant_due = cfg.debug.assert_invariants and metric_due
            if metric_due:
                self.metric_slab.update(
                    ds,
                    fallback_count=self.fallback_count,
                    graph_replay_count=self.graph_manager.replay_count,
                )

        diagnostics["invariants"] = (
            {"pending_metric_slab": True} if invariant_due else {"deferred": True}
        )
        diagnostics["profile"] = self.profiler.to_dict(resolve_gpu=False)
        diagnostics["graph"] = self.graph_manager.graph_status()
        replay_diagnostics = self._device_replay_diagnostics()
        diagnostics["raqic"] = replay_diagnostics["raqic"]
        diagnostics["actions"] = {
            "movement": replay_diagnostics["movement"],
            "reproduction": replay_diagnostics["reproduction"],
            "topology": replay_diagnostics["topology"],
        }
        diagnostics["per_ow_qiskit"] = ds.metadata.get("last_per_ow_qiskit")
        self.last_diagnostics = diagnostics
        self._steps_completed += 1
        return diagnostics

    def _schedule_metric_transfer(self) -> None:
        ticket = self.metric_slab.transfer_async(
            self.streams,
            metadata={"backend": self.ds.backend.name, "tick": int(self.ds.tick)},
        )
        self.pending_metric_tickets.append(ticket)
        self.transfer_ledger.record_d2h(
            ticket.host_array.nbytes,
            kind="metric",
            tick=int(self.ds.tick),
            source_stream="metrics",
            synchronization="event",
            scheduled=True,
            graph_compatible=False,
            reason="compact device metric slab at configured cadence",
        )

    def _poll_metric_transfers(self, *, block: bool = False) -> None:
        pending = []
        for ticket in self.pending_metric_tickets:
            arr = ticket.result(block=block)
            if arr is None:
                pending.append(ticket)
                continue
            metric = DeviceMetricSlab.decode(
                arr,
                backend=self.ds.backend.name,
                persistent=True,
            )
            self.metrics.append(metric)
            self.numerical_ledger.update_metrics(metric)
            if self.async_writer and (
                self.recording_policy is None
                or self.recording_policy.metric_due(int(metric.get("tick", 0)))
            ):
                self.async_writer.write({"kind": "metrics", **metric})
        self.pending_metric_tickets = pending

    def _shadow_pre_state(self) -> Any:
        if self.shadow_auditor is None:
            return None
        next_tick = int(self.ds.tick) + 1
        if not self.shadow_auditor.due(next_tick):
            return None
        if self.shadow_auditor.reference_mode == "implementation_numpy":
            snapshot = self.shadow_auditor.prepare_device_snapshot(
                self.ds,
                self.state,
            )
            self.transfer_ledger.record_d2h(
                int(self.ds.memory_estimate()["tracked_array_bytes"]),
                kind="shadow",
                tick=int(self.ds.tick),
                source_stream="shadow",
                synchronization="device",
                scheduled=True,
                graph_compatible=False,
                reason="scheduled implementation shadow snapshot",
            )
            return snapshot
        self.checkpoint(
            force=True,
            count=False,
            transfer_kind="shadow",
            transfer_reason="scheduled scientific CPU shadow pre-state",
        )
        return self.shadow_auditor.prepare_cpu_state(self.state)

    def _run_shadow(self, cpu_pre_state: Any, out: dict[str, Any]) -> None:
        if cpu_pre_state is None or self.shadow_auditor is None:
            return
        tick = int(self.ds.tick)
        cpu_post = self.shadow_auditor.run_cpu_reference(
            cpu_pre_state,
            tick=tick,
        )
        self.checkpoint(
            force=True,
            count=False,
            transfer_kind="shadow",
            transfer_reason="scheduled scientific CPU shadow post-state",
        )
        gpu_post = copy.deepcopy(self.state)
        parity = self.shadow_auditor.compare(
            cpu_post,
            gpu_post,
            tick=tick,
        )
        out["cpu_gpu_shadow"] = self.shadow_auditor.record(parity)

    def step(self) -> dict[str, Any]:
        if self.closed:
            raise RuntimeError("PersistentOWLDeviceRun is closed")
        command_results = self._apply_commands()
        if self.paused:
            return {
                "persistent": True,
                "paused": True,
                "commands": command_results,
                "backend": self.ds.backend.name,
            }
        cpu_pre_state = self._shadow_pre_state()
        out = self._step_device(command_results)
        metric_due = self.ds.tick % int(self.cfg.raqic.full_gpu_metric_every) == 0
        invariant_due = bool(self.cfg.debug.assert_invariants and metric_due)
        if metric_due:
            self._schedule_metric_transfer()
        self._poll_metric_transfers(block=invariant_due)
        if invariant_due:
            if not self.metrics or int(self.metrics[-1].get("tick", -1)) != int(self.ds.tick):
                raise RuntimeError(
                    "invariant metric slab did not complete at strict audit boundary"
                )
            invariant_values = invariant_summary_from_metric(self.metrics[-1])
            assert_invariant_summary(invariant_values, self.cfg)
            out["invariants"] = invariant_values

        if self.qiskit_validation is not None and self.qiskit_validation.due(
            self.ds.tick,
            requested=self.validation_requested,
        ):
            with (
                self.profiler.stage("qiskit_validation"),
                nvtx_range("owl.boundary.qiskit_validation"),
            ):
                validation = self.qiskit_validation.validate_device_state(
                    self.ds,
                    tick=self.ds.tick,
                    requested=self.validation_requested,
                )
            out["qiskit_validation"] = validation
            self.validation_requested = False

        self._run_shadow(cpu_pre_state, out)

        if self.visual_controller is not None:
            self.visual_controller.update_settings(self.visual_settings)
        if self.visual_controller is not None and self.visual_controller.render_due(self.ds.tick):
            frame = self.visual_controller.prepare_frame(
                device_state=self.ds,
                diagnostics=out,
            )
            self.transfer_ledger.record_d2h(
                frame.estimated_nbytes(),
                kind="visual",
                tick=int(self.ds.tick),
                source_stream="visual",
                synchronization="device",
                scheduled=True,
                graph_compatible=False,
                reason="bounded visual frame and event payload at render cadence",
            )
            profile_stages = out.get("profile", {}).get("stages", ())
            if isinstance(profile_stages, dict):
                profile_records = profile_stages.values()
            else:
                profile_records = profile_stages
            simulation_ms = sum(
                float(
                    item.get("gpu_ms")
                    or item.get("wall_ms")
                    or item.get("gpu_milliseconds")
                    or item.get("wall_milliseconds")
                    or 0.0
                )
                for item in profile_records
                if isinstance(item, dict)
            )
            self.visual_controller.submit(frame, simulation_ms=simulation_ms)
            out["visual"] = self.visual_controller.summary()

        if self.recording_policy is not None:
            record_payload = self.recording_policy.record_tick(self, out)
            if record_payload is not None:
                out["recording"] = {
                    "level": self.recording_policy.level,
                    "tick": int(self.ds.tick),
                }

        checkpoint_every = int(self.cfg.raqic.full_gpu_checkpoint_every)
        if self.checkpoint_requested or (checkpoint_every and self.ds.tick % checkpoint_every == 0):
            self.checkpoint(force=True)
            self.checkpoint_requested = False
        return out

    def collect_metrics(self) -> dict[str, Any]:
        with self.streams.compute:
            self.metric_slab.update(
                self.ds,
                fallback_count=self.fallback_count,
                graph_replay_count=self.graph_manager.replay_count,
            )
        ticket = self.metric_slab.transfer_async(self.streams)
        arr = ticket.result(block=True)
        assert arr is not None
        metric = DeviceMetricSlab.decode(arr, backend=self.ds.backend.name)
        self.numerical_ledger.update_metrics(metric)
        return metric

    def run(
        self,
        max_steps: int | None = None,
        *,
        checkpoint_final: bool = True,
    ) -> tuple[WorldState, list[dict[str, Any]]]:
        steps = int(self.cfg.world.max_steps if max_steps is None else max_steps)
        for _ in range(steps):
            self.step()
        self._poll_metric_transfers(block=True)
        if self.graph_manager.requirement == "full_tick":
            self.graph_manager.assert_requirement()
        if checkpoint_final:
            self.checkpoint()
        return self.state, self.metrics

    def checkpoint(
        self,
        fields: list[str] | None = None,
        *,
        force: bool = False,
        count: bool = True,
        transfer_kind: str = "checkpoint",
        transfer_reason: str = "explicit full-state checkpoint boundary",
    ) -> WorldState:
        tick = int(self.ds.tick)
        if (
            fields is None
            and not force
            and self._checkpointed_tick == tick
            and self._last_checkpoint is not None
        ):
            # A shadow/audit may have created the cached CPU mirror with
            # ``count=False``. A later normal-completion checkpoint must still
            # be counted exactly once even though no second transfer is needed.
            if count and self._checkpoint_counted_tick != tick:
                self.checkpoint_count += 1
                self._checkpoint_counted_tick = tick
            return self._last_checkpoint
        self.streams.compute.synchronize()
        if "_next_ow_id_device" in self.ds.arrays:
            next_id = int(
                self.ds.backend.asnumpy(self.ds.arrays["_next_ow_id_device"]).reshape(-1)[0]
            )
            self.ds.scalars["next_ow_id"] = next_id
        self.ds.write_back_to_cpu(self.state, fields=fields)
        self.transfer_ledger.record_d2h(
            int(self.ds.memory_estimate()["tracked_array_bytes"]),
            kind=transfer_kind,
            tick=tick,
            source_stream="compute",
            synchronization="stream",
            scheduled=True,
            graph_compatible=False,
            reason=transfer_reason,
        )
        if getattr(self.cfg.raqic, "enabled", False):
            quiesce_dead_raqic_fields(self.state)
        if fields is None:
            self._checkpointed_tick = tick
            self._last_checkpoint = self.state
            if count and self._checkpoint_counted_tick != tick:
                self.checkpoint_count += 1
                self._checkpoint_counted_tick = tick
        return self.state

    def _refresh_memory_evidence(self) -> None:
        if self.memory_plan is None:
            return
        current = pool = peak = None
        if self.ds.is_gpu:
            try:  # pragma: no cover - target CUDA host
                xp = self.ds.xp
                memory_pool = xp.get_default_memory_pool()
                current = int(memory_pool.used_bytes())
                pool = int(memory_pool.total_bytes())
                peak = max(current, pool)
            except Exception:
                current = pool = peak = None
        else:
            current = int(self.ds.memory_estimate()["tracked_array_bytes"])
            pool = current
            peak = current
        self.memory_plan.record_actual(
            current_bytes=current,
            pool_bytes=pool,
            peak_bytes=peak,
            unexplained_growth_bytes=(
                None if peak is None else max(0, int(peak) - int(self.memory_plan.peak_bytes))
            ),
        )
        self.ds.metadata["memory_preflight"] = self.memory_plan.to_dict()

    def execution_metadata(self) -> dict[str, Any]:
        self._refresh_memory_evidence()
        return {
            "simulation_backend": getattr(
                self.plan,
                "simulation_backend",
                "gpu_persistent",
            ),
            "decision_backend": getattr(
                self.plan,
                "decision_backend",
                "raqic_dense_gpu",
            ),
            "backend": self.ds.backend.name,
            "plan_hash": getattr(self.plan, "plan_hash", None),
            "scientific_contract_version": getattr(self.plan, "scientific_contract_version", None),
            "random_contract_version": getattr(self.plan, "random_contract_version", None),
            "graph_scope": getattr(self.plan, "graph_scope", "off"),
            "device_state_instances": 1,
            "checkpoint_count": int(self.checkpoint_count),
            "fallback_count": int(self.fallback_count),
            "graph": self.graph_manager.graph_status(),
            "per_ow_qiskit": self.ds.metadata.get("last_per_ow_qiskit"),
            "qiskit_validation": (
                None if self.qiskit_validation is None else self.qiskit_validation.summary()
            ),
            "cpu_shadow": (None if self.shadow_auditor is None else self.shadow_auditor.summary()),
            "visual": (
                None if self.visual_controller is None else self.visual_controller.summary()
            ),
            "recording": (
                None
                if self.recording_policy is None
                else {
                    "level": self.recording_policy.level,
                    "every": self.recording_policy.every,
                }
            ),
            "memory_preflight": self.ds.metadata.get("memory_preflight"),
            "last_metric": (self.metrics[-1] if self.metrics else None),
            "all_ow_accounted": (
                None if not self.metrics else bool(self.metrics[-1].get("all_ow_accounted", False))
            ),
        }

    def close(self, *, checkpoint: bool = False) -> None:
        if self.closed:
            return
        self._poll_metric_transfers(block=True)
        if checkpoint:
            self.checkpoint()
        self.streams.synchronize_all()
        if self.async_writer:
            self.async_writer.close()
        if self.visual_controller is not None:
            self.visual_controller.close()
        if self.counterfactual_source_observer is not None:
            self.counterfactual_source_observer.close()
        # Counterfactual branches have no recorder, visual controller, host
        # observer, or checkpoint ownership. Their ledgers are merged into the
        # scheduler before close. Suppress branch-local report files so
        # concurrent GPU lanes never race on runs/unidentified/reports and a
        # counterfactual cannot mutate the factual run's report tree.
        if bool(self.ds.metadata.get("counterfactual_suppress_close_reports", False)):
            self.closed = True
            return
        report_dir = (
            self.run_paths.reports
            if self.run_paths is not None
            else Path("runs/unidentified/reports")
        )
        report_dir.mkdir(parents=True, exist_ok=True)
        if bool(
            getattr(
                self.cfg.raqic,
                "full_gpu_enable_numerical_ledger",
                True,
            )
        ):
            self.numerical_ledger.graph_invalidation_count = int(
                self.graph_manager.invalidation_count
            )
            self.numerical_ledger.fallback_count = int(self.fallback_count)
            self.numerical_ledger.write(report_dir / "numerical_ledger.json")
        self.transfer_ledger.write(report_dir / "transfer_ledger.json")
        (report_dir / "graph_status.json").write_text(
            json.dumps(
                self.graph_manager.graph_status(),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (report_dir / "execution_metadata.json").write_text(
            json.dumps(self.execution_metadata(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.closed = True


def run_gpu_full_persistent(
    cfg: SimulationConfig,
    max_steps: int | None = None,
    *,
    plan: Any | None = None,
) -> Any:
    run = PersistentOWLDeviceRun.from_config(cfg, plan=plan)
    try:
        run.run(max_steps=max_steps, checkpoint_final=False)
        state = run.checkpoint()
        return state, run.metrics
    finally:
        run.close(checkpoint=False)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run persistent GPU-full OWL + RAQIC.")
    parser.add_argument("config")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument(
        "--out",
        default="results/gpu_v09_persistent_metrics.json",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    state, metrics = run_gpu_full_persistent(cfg, max_steps=args.steps)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"metrics": metrics, "final_tick": state.tick},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(path)


if __name__ == "__main__":
    main()
