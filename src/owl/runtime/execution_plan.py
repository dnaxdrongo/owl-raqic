from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Literal, cast

from owl.science.contract import current_scientific_contract, sha256_canonical
from owl_raqic.qiskit_backend.qiskit_policy import (
    QiskitExecutionPolicy,
)

from .capabilities import RuntimeCapabilities
from .module_enablement import ModuleEnablement
from .settings import RuntimeSettings, resolve_runtime_settings

SimulationBackend = Literal["cpu", "gpu_stage_once", "gpu_persistent", "gpu_graph", "gpu_multi"]
DecisionBackend = Literal[
    "legacy_cpu",
    "raqic_cpu",
    "raqic_dense_gpu",
    "raqic_qiskit_per_ow",
    "raqic_hybrid_validate",
]
VisualBackend = Literal["none", "pygame", "vispy", "headless_export"]


@dataclass(frozen=True)
class ExecutionPlan:
    simulation_backend: SimulationBackend
    decision_backend: DecisionBackend
    device_ids: tuple[int, ...]
    multi_gpu: bool
    visual_backend: VisualBackend
    recording_level: str
    cpu_shadow_ticks: tuple[int, ...]
    qiskit_policy: QiskitExecutionPolicy
    require_certification: bool
    graph_requirement: str
    strict: bool
    fallback_allowed: bool
    runtime_settings: RuntimeSettings
    enabled_modules: tuple[str, ...]
    graph_scope: Literal["off", "rank_local", "single_gpu_full_tick"]
    scientific_shadow_required: bool
    implementation_shadow_required: bool
    scientific_contract_version: str
    random_contract_version: str
    distributed_protocol_version: str
    memory_plan_version: str

    @staticmethod
    def _normalize(value: Any) -> Any:
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, tuple):
            return [ExecutionPlan._normalize(item) for item in value]
        if isinstance(value, list):
            return [ExecutionPlan._normalize(item) for item in value]
        if isinstance(value, dict):
            return {str(k): ExecutionPlan._normalize(v) for k, v in value.items()}
        return value

    def canonical_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], self._normalize(asdict(self)))

    def sha256(self) -> str:
        return sha256_canonical(self.canonical_dict())

    @property
    def plan_hash(self) -> str:
        return self.sha256()

    def to_dict(self) -> dict[str, Any]:
        out = self.canonical_dict()
        out["plan_hash"] = self.sha256()
        return out


def _shadow_ticks(cfg: Any) -> tuple[int, ...]:
    raw = int(getattr(cfg.raqic, "full_gpu_cpu_shadow_ticks", 0))
    if raw <= 0:
        return ()
    maximum = int(cfg.world.max_steps)
    return tuple(range(raw, maximum + 1, raw))


def _visual_backend(cfg: Any) -> VisualBackend:
    raw = str(getattr(cfg.raqic, "full_gpu_visual_backend", "none"))
    if not bool(getattr(cfg.visualization, "enabled", False)) and raw == "none":
        return "none"
    if raw == "pygame_copy":
        return "pygame"
    if raw == "vispy_gpu":
        return "vispy"
    if raw == "headless_export":
        return "headless_export"
    # Existing CPU visualization config uses pygame.
    return "pygame"


def compile_execution_plan(cfg: Any, runtime: RuntimeCapabilities) -> ExecutionPlan:
    """Compile configuration and runtime capabilities into one execution plan."""

    settings = resolve_runtime_settings(cfg)
    modules = ModuleEnablement.from_config(cfg)
    qiskit = QiskitExecutionPolicy.from_config(cfg)
    raqic_enabled = bool(getattr(cfg.raqic, "enabled", False))
    mode = str(getattr(cfg.raqic, "mode", "cpu_audit"))
    decision_policy = str(getattr(cfg.raqic, "decision_policy", "legacy"))
    legacy_full_mode = bool(getattr(cfg.raqic, "full_gpu_enabled", False))
    full_mode = mode in {"gpu_full", "gpu_full_hybrid_audit"} or legacy_full_mode
    if settings.gpu_backend != "cupy" or settings.full_gpu_backend != "cupy":
        raise RuntimeError("v0.9 GPU execution supports the CuPy backend only")
    strict = bool(getattr(cfg.raqic, "full_gpu_strict", True))
    # Strict execution never falls back. In non-strict modes fallback must be
    # explicitly requested and is always recorded in execution metadata.
    fallback_allowed = bool(getattr(cfg.raqic, "fallback_on_backend_error", False)) and not strict

    devices = tuple(int(x) for x in getattr(cfg.raqic, "full_gpu_devices", ()))
    if not devices and runtime.cuda_device_count:
        devices = (0,)
    multi_gpu = bool(getattr(cfg.raqic, "full_gpu_multi_gpu", False))

    if multi_gpu:
        if len(devices) < 2:
            raise RuntimeError("multi-GPU execution requires at least two selected devices")
        if runtime.cuda_device_count < len(devices):
            raise RuntimeError(
                f"requested {len(devices)} GPU devices but only "
                f"{runtime.cuda_device_count} are available"
            )
        if not runtime.nccl_available:
            raise RuntimeError("multi-GPU execution requires CuPy NCCL support")
        simulation_backend: SimulationBackend = "gpu_multi"
    elif full_mode:
        tier = str(getattr(cfg.raqic, "full_gpu_execution_tier", "reference"))
        transfer = str(getattr(cfg.raqic, "full_gpu_transfer_policy", "stage_once"))
        if tier == "graph":
            simulation_backend = "gpu_graph"
        elif tier == "persistent" or transfer in {"persistent_mirror", "hybrid_shadow"}:
            simulation_backend = "gpu_persistent"
        else:
            simulation_backend = "gpu_stage_once"
    else:
        simulation_backend = "cpu"

    if (
        simulation_backend.startswith("gpu")
        and strict
        and not runtime.has_cuda
        and not fallback_allowed
    ):
        # Explicit fallback is only legal in non-strict smoke configurations.
        raise RuntimeError(
            "GPU execution was requested in strict mode, but no usable "
            "CuPy CUDA device is available"
        )

    if qiskit.per_ow:
        decision_backend: DecisionBackend = "raqic_qiskit_per_ow"
        if qiskit.device.upper() != "GPU" and qiskit.strict_gpu:
            raise RuntimeError("strict per-OW Qiskit requires qiskit_gpu_device='GPU'")
        if qiskit.device.upper() == "GPU" and qiskit.strict_gpu and not runtime.aer_gpu_available:
            raise RuntimeError(
                "per-OW Qiskit GPU execution was requested but Aer reports no GPU device"
            )
        if not qiskit.confirm_expensive:
            raise RuntimeError("per-OW Qiskit execution requires qiskit_confirm_expensive=true")
    elif qiskit.validation_only or bool(getattr(cfg.raqic, "gpu_validate_qiskit", False)):
        decision_backend = "raqic_hybrid_validate"
    elif raqic_enabled and simulation_backend.startswith("gpu"):
        decision_backend = "raqic_dense_gpu"
    elif raqic_enabled or decision_policy == "raqic":
        decision_backend = "raqic_cpu"
    else:
        decision_backend = "legacy_cpu"

    graph_requirement = str(getattr(cfg.raqic, "full_gpu_graph_requirement", "allow_partial"))
    if graph_requirement == "full_tick" and simulation_backend != "gpu_graph":
        raise RuntimeError("full-tick graph requirement requires gpu_graph execution")
    if graph_requirement == "full_tick":
        if qiskit.per_ow:
            raise RuntimeError(
                "per-OW Qiskit is an external simulator boundary and cannot be "
                "captured inside the full-tick CUDA graph"
            )
        if not bool(getattr(cfg.raqic, "full_gpu_fuse_scatter", False)):
            raise RuntimeError("full-tick graph execution requires full_gpu_fuse_scatter=true")
        if str(getattr(cfg.raqic, "full_gpu_phase_mode", "")) != "canonical_device":
            raise RuntimeError(
                "full-tick graph execution requires full_gpu_phase_mode='canonical_device'"
            )
        required_capacity = 3 * int(cfg.world.height) * int(cfg.world.width)
        if int(getattr(cfg.raqic, "full_gpu_sparse_event_capacity", 0)) < required_capacity:
            raise RuntimeError(
                "full-tick graph execution requires full_gpu_sparse_event_capacity "
                f">= 3*H*W ({required_capacity})"
            )

    visual = _visual_backend(cfg)
    if visual == "vispy" and not runtime.vispy_available:
        raise RuntimeError("VisPy visual backend requested but VisPy is unavailable")
    if visual == "pygame" and not runtime.pygame_available:
        raise RuntimeError("Pygame visual backend requested but Pygame is unavailable")

    shadow_ticks = _shadow_ticks(cfg)
    if mode == "gpu_full_hybrid_audit" and not shadow_ticks:
        raise RuntimeError("gpu_full_hybrid_audit requires full_gpu_cpu_shadow_ticks > 0")

    scientific_contract = current_scientific_contract()
    shadow_reference = str(getattr(cfg.raqic, "full_gpu_shadow_reference", "scientific_cpu"))
    if shadow_reference == "legacy_cpu_semantic":
        shadow_reference = "scientific_cpu"
    elif shadow_reference == "dense_numpy_exact":
        shadow_reference = "implementation_numpy"
    scientific_shadow_required = bool(shadow_ticks) and shadow_reference == "scientific_cpu"
    implementation_shadow_required = bool(
        getattr(cfg.raqic, "full_gpu_implementation_shadow_required", False)
    ) or (bool(shadow_ticks) and shadow_reference == "implementation_numpy")
    graph_scope = "off"
    if simulation_backend == "gpu_graph":
        graph_scope = "single_gpu_full_tick" if graph_requirement == "full_tick" else "rank_local"
    elif (
        simulation_backend == "gpu_multi"
        and str(getattr(cfg.raqic, "full_gpu_execution_tier", "persistent")) == "graph"
    ):
        graph_scope = "rank_local"

    return ExecutionPlan(
        simulation_backend=simulation_backend,
        decision_backend=decision_backend,
        device_ids=devices,
        multi_gpu=multi_gpu,
        visual_backend=visual,
        recording_level=str(
            getattr(
                cfg.raqic,
                "full_gpu_recording_level_v07",
                getattr(cfg.raqic, "full_gpu_recording_level", "summary_gpu"),
            )
        ),
        cpu_shadow_ticks=shadow_ticks,
        qiskit_policy=qiskit,
        require_certification=bool(getattr(cfg.raqic, "full_gpu_certification_required", False)),
        graph_requirement=graph_requirement,
        strict=strict,
        fallback_allowed=fallback_allowed,
        runtime_settings=settings,
        enabled_modules=tuple(sorted(modules.enabled)),
        graph_scope=cast(Literal["off", "rank_local", "single_gpu_full_tick"], graph_scope),
        scientific_shadow_required=scientific_shadow_required,
        implementation_shadow_required=implementation_shadow_required,
        scientific_contract_version=scientific_contract.version,
        random_contract_version=scientific_contract.random_contract_version,
        distributed_protocol_version="owl-nccl-protocol-v2",
        memory_plan_version="owl-memory-plan-v2",
    )
