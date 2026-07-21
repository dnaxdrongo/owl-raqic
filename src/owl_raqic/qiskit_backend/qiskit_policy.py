from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class QiskitDecisionMode(StrEnum):
    OFF = "off"
    VALIDATION_SAMPLE = "validation_sample"
    EVERY_OW_STATIC_EXACT = "every_ow_static_exact"
    EVERY_OW_DYNAMIC_SHOTS = "every_ow_dynamic_shots"
    EVERY_OW_CIRCUIT_FAMILY = "every_ow_circuit_family"


class QiskitReadoutPolicy(StrEnum):
    DETERMINISTIC_SAMPLE = "deterministic_sample"
    ARGMAX = "argmax"
    FIRST_SHOT = "first_shot"


@dataclass(frozen=True)
class QiskitExecutionPolicy:
    mode: QiskitDecisionMode = QiskitDecisionMode.OFF
    circuit_families: tuple[str, ...] = ("static",)
    authoritative_family: str = "static"
    method: str = "statevector"
    shots: int = 4096
    chunk_size: int = 64
    strict_gpu: bool = True
    runtime_parameter_binding: bool = False
    runtime_binding_policy: str = "required_native"
    state_preparation_strategy: str = "exact_native_rotation_tree"
    runtime_binding_preflight_required: bool = True
    runtime_binding_preflight_batch_size: int = 8
    runtime_binding_preflight_tolerance: float = 1e-10
    automatic_execution_fallback: bool = False
    batched_shots_gpu: bool = False
    shot_branching: bool = False
    target_gpus: tuple[int, ...] = ()
    device: str = "GPU"
    job_queue_depth: int = 2
    cache_templates: bool = True
    group_by_signature: bool = True
    readout_policy: QiskitReadoutPolicy = QiskitReadoutPolicy.DETERMINISTIC_SAMPLE
    confirm_expensive: bool = False
    interference_mixer_strength: float = 0.0
    interference_trotter_steps: int = 1
    action_names: tuple[str, ...] = ()

    @property
    def per_ow(self) -> bool:
        return self.mode in {
            QiskitDecisionMode.EVERY_OW_STATIC_EXACT,
            QiskitDecisionMode.EVERY_OW_DYNAMIC_SHOTS,
            QiskitDecisionMode.EVERY_OW_CIRCUIT_FAMILY,
        }

    @property
    def validation_only(self) -> bool:
        return self.mode == QiskitDecisionMode.VALIDATION_SAMPLE

    @classmethod
    def from_config(cls, cfg: Any) -> QiskitExecutionPolicy:
        raw_mode = str(getattr(cfg.raqic, "qiskit_decision_mode", "off"))
        if bool(getattr(cfg.raqic, "use_qiskit_for_all", False)) and raw_mode in {
            "off",
            "validation_sample",
        }:
            raw_mode = "every_ow_static_exact"
        mode = QiskitDecisionMode(raw_mode)
        families = tuple(str(x) for x in getattr(cfg.raqic, "qiskit_circuit_families", ("static",)))
        if not families:
            families = ("static",)
        authoritative = str(getattr(cfg.raqic, "qiskit_authoritative_family", families[0]))
        if authoritative not in families:
            raise ValueError(
                "qiskit_authoritative_family must be included in qiskit_circuit_families"
            )
        chunk_size = getattr(cfg.raqic, "gpu_chunk_size", None)
        if chunk_size is None:
            chunk_size = int(getattr(cfg.raqic, "qiskit_chunk_size", 64))
        target = tuple(int(x) for x in getattr(cfg.raqic, "qiskit_target_gpus", ()))
        return cls(
            mode=mode,
            circuit_families=families,
            authoritative_family=authoritative,
            method=str(getattr(cfg.raqic, "qiskit_gpu_method", "statevector")),
            shots=int(
                getattr(cfg.raqic, "qiskit_validation_shots", getattr(cfg.raqic, "shots", 4096))
            ),
            chunk_size=max(1, int(chunk_size)),
            strict_gpu=bool(getattr(cfg.raqic, "full_gpu_qiskit_strict", True)),
            runtime_parameter_binding=bool(
                getattr(cfg.raqic, "qiskit_runtime_parameter_bind_enable", False)
            ),
            runtime_binding_policy=str(
                getattr(cfg.raqic, "qiskit_runtime_binding_policy", "required_native")
            ),
            state_preparation_strategy=str(
                getattr(
                    cfg.raqic,
                    "qiskit_state_preparation_strategy",
                    "exact_native_rotation_tree",
                )
            ),
            runtime_binding_preflight_required=bool(
                getattr(cfg.raqic, "qiskit_preflight_required", True)
            ),
            runtime_binding_preflight_batch_size=int(
                getattr(cfg.raqic, "qiskit_preflight_batch_size", 8)
            ),
            runtime_binding_preflight_tolerance=float(
                getattr(cfg.raqic, "qiskit_preflight_tolerance", 1e-10)
            ),
            automatic_execution_fallback=bool(
                getattr(cfg.raqic, "qiskit_allow_automatic_execution_fallback", False)
            ),
            batched_shots_gpu=bool(getattr(cfg.raqic, "qiskit_batched_shots_gpu", False)),
            shot_branching=bool(getattr(cfg.raqic, "qiskit_shot_branching_enable", False)),
            target_gpus=target,
            device=str(getattr(cfg.raqic, "qiskit_gpu_device", "GPU")),
            job_queue_depth=max(1, int(getattr(cfg.raqic, "qiskit_job_queue_depth", 2))),
            cache_templates=bool(getattr(cfg.raqic, "cache_templates", True)),
            group_by_signature=bool(
                getattr(cfg.raqic, "dense_signature_grouping", False)
                or getattr(cfg.raqic, "batch_by_feature_signature", False)
            ),
            readout_policy=QiskitReadoutPolicy(
                str(getattr(cfg.raqic, "qiskit_readout_policy", "deterministic_sample"))
            ),
            confirm_expensive=bool(getattr(cfg.raqic, "qiskit_confirm_expensive", False)),
            interference_mixer_strength=float(
                getattr(cfg.raqic, "interference_mixer_strength", 0.0)
            ),
            interference_trotter_steps=int(getattr(cfg.raqic, "interference_trotter_steps", 1)),
            action_names=tuple(getattr(cfg.raqic, "action_names", ())),
        )
