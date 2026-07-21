from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RuntimeSettings:
    """Store resolved runtime controls.

    This object is constructed for every run so public configuration fields are
    either consumed here and propagated, or explicitly marked as unsupported.
    """

    gpu_backend: str
    gpu_transfer_policy: str
    full_gpu_backend: str
    profile_enabled: bool
    audit_fraction: float
    validation_subset_fraction: float
    validation_limit: int
    qiskit_device: str
    probability_tolerance: float
    kl_tolerance: float
    physical_modules: tuple[str, ...]
    movement_conflict_policy: str
    reproduction_conflict_policy: str
    fuse_biology: bool
    policy_backend: str
    recording_level: str
    record_every: int
    benchmark_label: str
    qiskit_job_queue_depth: int
    persist_quantum_state: bool
    record_measurement_records: bool
    record_action_probabilities: bool


# These switches accept alternate configuration names but do not define an
# independent execution mechanism.
DEPRECATED_FIELDS: dict[str, tuple[str, str]] = {
    "qiskit_debug_ow_limit": ("gpu_audit_limit", "1.0"),
    "gpu_audit_fraction": ("full_gpu_audit_fraction", "1.0"),
    "full_gpu_enabled": ("raqic.mode='gpu_full' or 'gpu_full_hybrid_audit'", "1.0"),
    "full_gpu_recording_level": ("full_gpu_recording_level_v07", "1.0"),
    "gpu_transfer_policy": ("full_gpu_transfer_policy", "1.0"),
    "full_gpu_fuse_biology": ("scientifically recovered stage kernels", "1.0"),
}


def resolve_runtime_settings(cfg: Any) -> RuntimeSettings:
    r = cfg.raqic
    for name, (replacement, removal) in DEPRECATED_FIELDS.items():
        field = type(r).model_fields[name]
        value = getattr(r, name)
        default = field.default
        if value != default:
            warnings.warn(
                f"raqic.{name} is deprecated; use {replacement}. "
                f"Scheduled for removal in {removal}.",
                DeprecationWarning,
                stacklevel=2,
            )

    audit_fraction = max(
        float(getattr(r, "gpu_audit_fraction", 0.0)),
        float(getattr(r, "full_gpu_audit_fraction", 0.0)),
    )
    validation_fraction = max(
        float(getattr(r, "qiskit_subset_fraction", 0.0)),
        audit_fraction,
    )
    validation_limit = int(getattr(r, "gpu_audit_limit", 0) or 0)
    debug_limit = int(getattr(r, "qiskit_debug_ow_limit", 0) or 0)
    if debug_limit:
        validation_limit = min(validation_limit or debug_limit, debug_limit)

    return RuntimeSettings(
        gpu_backend=str(r.gpu_backend),
        gpu_transfer_policy=str(r.gpu_transfer_policy),
        full_gpu_backend=str(r.full_gpu_backend),
        profile_enabled=bool(
            getattr(r, "gpu_profile", False) or getattr(r, "full_gpu_profile", False)
        ),
        audit_fraction=audit_fraction,
        validation_subset_fraction=validation_fraction,
        validation_limit=validation_limit,
        qiskit_device=str(getattr(r, "qiskit_gpu_device", "GPU")),
        probability_tolerance=float(getattr(r, "gpu_probability_tolerance", 1e-8)),
        kl_tolerance=float(getattr(r, "gpu_kl_tolerance", 1e-7)),
        physical_modules=tuple(str(x) for x in getattr(r, "full_gpu_physical_modules", ())),
        movement_conflict_policy=str(
            getattr(r, "full_gpu_movement_conflict_policy", "sort_priority")
        ),
        reproduction_conflict_policy=str(
            getattr(r, "full_gpu_reproduction_conflict_policy", "sort_priority")
        ),
        fuse_biology=bool(r.full_gpu_fuse_biology),
        policy_backend=str(getattr(r, "full_gpu_policy_backend", "stable")),
        recording_level=str(
            getattr(
                r,
                "full_gpu_recording_level_v07",
                getattr(r, "full_gpu_recording_level", "metrics_plus_events"),
            )
        ),
        record_every=int(getattr(r, "full_gpu_record_every", 1)),
        benchmark_label=str(getattr(r, "full_gpu_benchmark_label", "v0.9")),
        qiskit_job_queue_depth=int(getattr(r, "qiskit_job_queue_depth", 2)),
        persist_quantum_state=bool(getattr(r, "persist_quantum_state", True)),
        record_measurement_records=bool(getattr(r, "record_measurement_records", True)),
        record_action_probabilities=bool(getattr(r, "record_action_probabilities", True)),
    )
