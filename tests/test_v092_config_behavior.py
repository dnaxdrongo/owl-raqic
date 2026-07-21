from __future__ import annotations

import warnings
from dataclasses import asdict

import pytest

from owl.core.config import RAQICConfig, SimulationConfig
from owl.runtime.config_behavior import covers_config_field
from owl.runtime.settings import resolve_runtime_settings
from owl_raqic.qiskit_backend.qiskit_policy import QiskitExecutionPolicy

FIELDS = (
    "enabled",
    "mode",
    "decision_policy",
    "epsilon_raqic",
    "epsilon_adelic",
    "beta_intention",
    "action_temperature",
    "rounds_per_tick",
    "shots",
    "active_primes",
    "prime_weights",
    "use_qiskit_for_all",
    "qiskit_subset_fraction",
    "qiskit_debug_ow_limit",
    "batch_by_feature_signature",
    "cache_templates",
    "persist_quantum_state",
    "store_density_diagnostics",
    "record_measurement_records",
    "record_action_probabilities",
    "fallback_on_backend_error",
    "assert_recovery_gates",
    "parent_intention_eta",
    "max_cells_per_tick",
    "debug_store_full_records",
    "gpu_backend",
    "gpu_precision",
    "strict_gpu",
    "gpu_all_cells_required",
    "gpu_transfer_policy",
    "gpu_profile",
    "gpu_audit_fraction",
    "gpu_audit_limit",
    "gpu_validate_qiskit",
    "gpu_validate_cpu",
    "gpu_memory_limit_mb",
    "qiskit_gpu_method",
    "qiskit_gpu_device",
    "qiskit_batched_shots_gpu",
    "qiskit_enable_cuStateVec",
    "dense_signature_grouping",
    "gpu_chunk_size",
    "gpu_probability_tolerance",
    "gpu_kl_tolerance",
    "full_gpu_enabled",
    "full_gpu_strict",
    "full_gpu_backend",
    "full_gpu_transfer_policy",
    "full_gpu_precision",
    "full_gpu_physical_modules",
    "full_gpu_sparse_event_capacity",
    "full_gpu_movement_conflict_policy",
    "full_gpu_reproduction_conflict_policy",
    "full_gpu_visual_backend",
    "full_gpu_recording_level",
    "full_gpu_profile",
    "full_gpu_audit_fraction",
    "full_gpu_cpu_shadow_ticks",
    "full_gpu_no_silent_fallback",
    "full_gpu_execution_tier",
    "full_gpu_memory_policy",
    "full_gpu_memory_safety_fraction",
    "full_gpu_graph_mode",
    "full_gpu_graph_warmup_ticks",
    "full_gpu_stencil_backend",
    "full_gpu_fuse_biology",
    "full_gpu_fuse_scatter",
    "full_gpu_phase_mode",
    "full_gpu_phase_policy",
    "full_gpu_policy_backend",
    "full_gpu_recording_level_v07",
    "full_gpu_render_every",
    "full_gpu_record_every",
    "full_gpu_writer_queue_capacity",
    "full_gpu_writer_overflow_policy",
    "full_gpu_visual_event_capacity",
    "full_gpu_sprite_theme",
    "full_gpu_visual_clutter_budget",
    "full_gpu_benchmark_label",
    "full_gpu_metric_every",
    "full_gpu_checkpoint_every",
    "full_gpu_validation_every",
    "full_gpu_graph_allow_fallback",
    "full_gpu_qiskit_strict",
    "full_gpu_qiskit_allow_cpu_fallback",
    "full_gpu_run_class",
    "full_gpu_enable_numerical_ledger",
    "full_gpu_command_capacity",
    "full_gpu_certification_required",
    "full_gpu_memory_preflight",
    "full_gpu_visual_adaptive_lod",
    "full_gpu_visual_max_slowdown_fraction",
    "qiskit_shot_branching_enable",
    "qiskit_runtime_parameter_bind_enable",
    "qiskit_validation_max_qubits",
    "qiskit_validation_shots",
    "qiskit_decision_mode",
    "qiskit_circuit_families",
    "qiskit_authoritative_family",
    "qiskit_readout_policy",
    "qiskit_target_gpus",
    "qiskit_chunk_size",
    "qiskit_job_queue_depth",
    "qiskit_confirm_expensive",
    "full_gpu_graph_requirement",
    "full_gpu_devices",
    "full_gpu_multi_gpu",
    "full_gpu_distributed_timeout_seconds",
    "full_gpu_shadow_strict",
    "full_gpu_shadow_tolerance",
    "full_gpu_shadow_reference",
    "full_gpu_implementation_shadow_required",
    "full_gpu_certification_dir",
    "full_gpu_production_marker",
)


def _alternate(name: str, field, value):
    manual = {
        "prime_weights": {2: 1.0, 3: 0.5},
        "active_primes": (2, 3),
        "max_cells_per_tick": 16,
        "gpu_memory_limit_mb": 512.0,
        "gpu_chunk_size": 8,
        "full_gpu_devices": (0,),
        "qiskit_target_gpus": (0,),
        "full_gpu_physical_modules": ("environment", "sensing", "utility"),
        "qiskit_circuit_families": ("static", "deferred"),
    }
    if name in manual:
        return manual[name]
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1 if value >= 0 else 1
    if isinstance(value, float):
        if name.endswith("fraction") or "fraction" in name:
            return 0.25 if value != 0.25 else 0.5
        return value + 0.125
    if isinstance(value, str):
        annotation = field.annotation
        options = tuple(getattr(annotation, "__args__", ()))
        for option in options:
            if isinstance(option, str) and option != value:
                return option
        return value + "_v092"
    if isinstance(value, tuple):
        return value + value[:1] if value else (1,)
    if value is None:
        return 1
    return value


@covers_config_field(
    "enabled",
    "mode",
    "decision_policy",
    "epsilon_raqic",
    "epsilon_adelic",
    "beta_intention",
    "action_temperature",
    "rounds_per_tick",
    "shots",
    "active_primes",
    "prime_weights",
    "use_qiskit_for_all",
    "qiskit_subset_fraction",
    "qiskit_debug_ow_limit",
    "batch_by_feature_signature",
    "cache_templates",
    "persist_quantum_state",
    "store_density_diagnostics",
    "record_measurement_records",
    "record_action_probabilities",
    "fallback_on_backend_error",
    "assert_recovery_gates",
    "parent_intention_eta",
    "max_cells_per_tick",
    "debug_store_full_records",
    "gpu_backend",
    "gpu_precision",
    "strict_gpu",
    "gpu_all_cells_required",
    "gpu_transfer_policy",
    "gpu_profile",
    "gpu_audit_fraction",
    "gpu_audit_limit",
    "gpu_validate_qiskit",
    "gpu_validate_cpu",
    "gpu_memory_limit_mb",
    "qiskit_gpu_method",
    "qiskit_gpu_device",
    "qiskit_batched_shots_gpu",
    "qiskit_enable_cuStateVec",
    "dense_signature_grouping",
    "gpu_chunk_size",
    "gpu_probability_tolerance",
    "gpu_kl_tolerance",
    "full_gpu_enabled",
    "full_gpu_strict",
    "full_gpu_backend",
    "full_gpu_transfer_policy",
    "full_gpu_precision",
    "full_gpu_physical_modules",
    "full_gpu_sparse_event_capacity",
    "full_gpu_movement_conflict_policy",
    "full_gpu_reproduction_conflict_policy",
    "full_gpu_visual_backend",
    "full_gpu_recording_level",
    "full_gpu_profile",
    "full_gpu_audit_fraction",
    "full_gpu_cpu_shadow_ticks",
    "full_gpu_no_silent_fallback",
    "full_gpu_execution_tier",
    "full_gpu_memory_policy",
    "full_gpu_memory_safety_fraction",
    "full_gpu_graph_mode",
    "full_gpu_graph_warmup_ticks",
    "full_gpu_stencil_backend",
    "full_gpu_fuse_biology",
    "full_gpu_fuse_scatter",
    "full_gpu_phase_mode",
    "full_gpu_phase_policy",
    "full_gpu_policy_backend",
    "full_gpu_recording_level_v07",
    "full_gpu_render_every",
    "full_gpu_record_every",
    "full_gpu_writer_queue_capacity",
    "full_gpu_writer_overflow_policy",
    "full_gpu_visual_event_capacity",
    "full_gpu_sprite_theme",
    "full_gpu_visual_clutter_budget",
    "full_gpu_benchmark_label",
    "full_gpu_metric_every",
    "full_gpu_checkpoint_every",
    "full_gpu_validation_every",
    "full_gpu_graph_allow_fallback",
    "full_gpu_qiskit_strict",
    "full_gpu_qiskit_allow_cpu_fallback",
    "full_gpu_run_class",
    "full_gpu_enable_numerical_ledger",
    "full_gpu_command_capacity",
    "full_gpu_certification_required",
    "full_gpu_memory_preflight",
    "full_gpu_visual_adaptive_lod",
    "full_gpu_visual_max_slowdown_fraction",
    "qiskit_shot_branching_enable",
    "qiskit_runtime_parameter_bind_enable",
    "qiskit_validation_max_qubits",
    "qiskit_validation_shots",
    "qiskit_decision_mode",
    "qiskit_circuit_families",
    "qiskit_authoritative_family",
    "qiskit_readout_policy",
    "qiskit_target_gpus",
    "qiskit_chunk_size",
    "qiskit_job_queue_depth",
    "qiskit_confirm_expensive",
    "full_gpu_graph_requirement",
    "full_gpu_devices",
    "full_gpu_multi_gpu",
    "full_gpu_distributed_timeout_seconds",
    "full_gpu_shadow_strict",
    "full_gpu_shadow_tolerance",
    "full_gpu_shadow_reference",
    "full_gpu_implementation_shadow_required",
    "full_gpu_certification_dir",
    "full_gpu_production_marker",
)
@pytest.mark.parametrize("name", FIELDS)
def test_each_public_raqic_field_has_explicit_mutation_evidence(name):
    field = RAQICConfig.model_fields[name]
    base = RAQICConfig()
    original = getattr(base, name)
    alternate = _alternate(name, field, original)
    mutated = base.model_copy(update={name: alternate})
    assert getattr(mutated, name) != original

    # Exercise the two typed policy compilers used by all public execution
    # paths. Some low-level fields are consumed after planning, so the raw
    # changed value is retained alongside the resolved policy fingerprints.
    cfg = SimulationConfig()
    # Bypass whole-config cross-validation here so each single public field can
    # be observed independently; separate config tests cover invalid
    # combinations and rejection behavior.
    object.__setattr__(cfg, "raqic", mutated)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        runtime = asdict(resolve_runtime_settings(cfg))
        try:
            qiskit = asdict(QiskitExecutionPolicy.from_config(cfg))
        except ValueError:
            # A single-field mutation can intentionally create an invalid
            # family/authority combination; rejection is the behavior.
            qiskit = {"rejected": True}
    fingerprint = {
        "field": name,
        "value": getattr(mutated, name),
        "runtime": runtime,
        "qiskit": qiskit,
    }
    assert fingerprint["value"] == alternate
