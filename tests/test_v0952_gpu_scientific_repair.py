from __future__ import annotations

import copy

import numpy as np

from owl.core.actions import Action
from owl.core.config import SimulationConfig, load_config
from owl.core.init import initialize_world
from owl.engine.aggregation import aggregate_patches
from owl.gpu.device_state import OWLDeviceState
from owl.gpu.kernels.stencil_kernels import (
    _fused_local_raw_compatible,
    fused_local_scratch,
)
from owl.gpu.stages.aggregation_gpu import aggregate_patches_gpu
from owl.gpu.stages.raqic_gpu_stage import quiesce_dead_raqic_fields_gpu
from owl.gpu.stages.topdown_gpu import dispatch_parent_context_gpu
from owl.gpu.stencil import neighbor_sum_8
from owl.raqic.state import ensure_raqic_fields, quiesce_dead_raqic_fields
from owl.science.stage_parity import compare_field


def _small_cfg() -> SimulationConfig:
    data = load_config("configs/mvp.yaml").model_dump()
    data["world"]["height"] = 20
    data["world"]["width"] = 20
    data["world"]["patch_size"] = 5
    data["initialization"]["population_density"] = 0.7
    data["initialization"]["food_patch_count"] = 1
    data["visualization"]["enabled"] = False
    data["recording"]["enabled"] = False
    data["raqic"]["enabled"] = True
    data["raqic"]["full_gpu_precision"] = "audit64"
    return SimulationConfig.model_validate(data)


def test_audit64_preserves_physical_float32_and_promotes_raqic_evidence() -> None:
    cfg = _small_cfg()
    state = initialize_world(cfg, np.random.default_rng(11))
    ensure_raqic_fields(state, cfg)
    ds = OWLDeviceState.from_world_state(state, cfg, force_backend="numpy")

    assert ds.health.dtype == np.float32
    assert ds.food.dtype == np.float32
    assert ds.phase.dtype == np.float32
    assert ds.patch_arrays["phase"].dtype == np.float32
    assert ds.raqic_probabilities.dtype == np.float64
    assert ds.raqic_score.dtype == np.float64
    assert ds.raqic_phase.dtype == np.float64
    assert ds.metadata["precision_policy"] == "raqic_audit64_physical_source"


def test_raw_fused_kernel_gate_rejects_dtype_mismatched_outputs() -> None:
    class FakeCuPy:
        __name__ = "cupy"

    xp = FakeCuPy()
    alive = np.ones((4, 4), dtype=bool)
    field64 = np.ones((4, 4), dtype=np.float64)
    outputs32 = tuple(np.zeros((4, 4), dtype=np.float32) for _ in range(5))
    outputs64 = tuple(np.zeros((4, 4), dtype=np.float64) for _ in range(5))

    assert not _fused_local_raw_compatible(
        alive, field64, field64, field64, outputs32, xp, "toroidal"
    )
    assert _fused_local_raw_compatible(alive, field64, field64, field64, outputs64, xp, "toroidal")


def test_fused_local_mixed_dtype_fallback_matches_float32_contract() -> None:
    rng = np.random.default_rng(19)
    alive = rng.random((8, 8)) > 0.3
    food = rng.random((8, 8)).astype(np.float64)
    toxin = rng.random((8, 8)).astype(np.float64)
    phase = (rng.random((8, 8)) * 2.0 * np.pi).astype(np.float64)
    outputs = tuple(np.zeros((8, 8), dtype=np.float32) for _ in range(5))

    result = fused_local_scratch(alive, food, toxin, phase, np, outputs=outputs)

    expected_alive = neighbor_sum_8(alive.astype(np.float32), np, "toroidal") / np.float32(8.0)
    expected_food = neighbor_sum_8(food.astype(np.float32), np, "toroidal") / np.float32(8.0)
    expected_toxin = neighbor_sum_8(toxin.astype(np.float32), np, "toroidal") / np.float32(8.0)
    expected_sin = neighbor_sum_8(np.sin(phase.astype(np.float32)), np, "toroidal")
    expected_cos = neighbor_sum_8(np.cos(phase.astype(np.float32)), np, "toroidal")

    assert np.array_equal(result.local_alive_density, expected_alive.astype(np.float32))
    assert np.array_equal(result.food_mean, expected_food.astype(np.float32))
    assert np.array_equal(result.toxin_mean, expected_toxin.astype(np.float32))
    assert np.array_equal(result.phase_sin_sum, expected_sin.astype(np.float32))
    assert np.array_equal(result.phase_cos_sum, expected_cos.astype(np.float32))


def test_numpy_device_patch_aggregation_matches_cpu_circular_contract() -> None:
    cfg = _small_cfg()
    state = initialize_world(cfg, np.random.default_rng(23))
    # Adversarial values straddling the 0 / 2pi branch cut.
    checker = np.indices(state.phase.shape).sum(axis=0) % 2
    state.phase[...] = np.where(checker == 0, np.float32(1e-6), np.float32(2.0 * np.pi - 1e-6))
    integration = np.linspace(0.05, 0.95, state.integration.size, dtype=np.float32).reshape(
        state.integration.shape
    )
    state.integration[...] = integration

    cpu_state = copy.deepcopy(state)
    expected = aggregate_patches(cpu_state, cfg)

    ds = OWLDeviceState.from_world_state(copy.deepcopy(state), cfg, force_backend="numpy")
    aggregate_patches_gpu(ds, cfg)

    circular = np.abs(
        np.arctan2(
            np.sin(expected.phase.astype(np.float64) - ds.patch_arrays["phase"].astype(np.float64)),
            np.cos(expected.phase.astype(np.float64) - ds.patch_arrays["phase"].astype(np.float64)),
        )
    )
    assert float(np.max(circular)) <= 2e-6

    for name in (
        "activation",
        "memory",
        "integration",
        "resource",
        "health",
        "boundary",
        "synchrony",
        "coherence",
        "cross_scale",
        "possibility",
        "signal_pressure",
    ):
        np.testing.assert_allclose(
            ds.patch_arrays[name], getattr(expected, name), atol=2e-6, rtol=2e-6
        )
        assert ds.patch_arrays[name].dtype == np.float32


def test_topdown_dispatch_does_not_overwrite_raqic_parent_intention() -> None:
    cfg = _small_cfg()
    state = initialize_world(cfg, np.random.default_rng(31))
    ensure_raqic_fields(state, cfg)
    ds = OWLDeviceState.from_world_state(state, cfg, force_backend="numpy")
    parent = np.zeros_like(ds.raqic_parent_intention)
    parent[..., int(Action.SENSE)] = 0.35
    parent[..., int(Action.INTEGRATE)] = 0.65
    ds.arrays["raqic_parent_intention"][...] = parent

    dispatch_parent_context_gpu(ds, cfg, force_global=True)

    assert np.array_equal(ds.raqic_parent_intention, parent)
    assert "pre_parent_bias" in ds.arrays
    assert "_parent_phase" in ds.arrays


def test_stage_parity_compares_phase_on_the_circle() -> None:
    left = np.array([0.0, np.pi, 2.0 * np.pi - 2e-7], dtype=np.float64)
    right = np.array([2.0 * np.pi - 1e-7, -np.pi, 1e-7], dtype=np.float64)

    circular = compare_field("phase", left, right, exact=False, atol=5e-7, rtol=0.0)
    linear = compare_field("resource", left, right, exact=False, atol=5e-7, rtol=0.0)

    assert circular.passed
    assert circular.max_abs is not None and circular.max_abs < 5e-7
    assert not linear.passed


def test_scientific_trace_forces_raqic_phase_and_populates_all_records() -> None:
    from owl.gpu.stages.raqic_gpu_stage import run_raqic_gpu_stage

    cfg = _small_cfg()
    cfg.raqic.full_gpu_phase_policy = "audit_or_visual"
    state = initialize_world(cfg, np.random.default_rng(41))
    ensure_raqic_fields(state, cfg)
    ds = OWLDeviceState.from_world_state(state, cfg, force_backend="numpy")
    ds.metadata["defer_host_metrics"] = True
    ds.metadata["scientific_stage_parity"] = True
    ds.arrays["_authority_bool"] = np.ones_like(ds.possibility, dtype=bool)

    metadata = run_raqic_gpu_stage(ds, cfg)
    live = (ds.health > 0.0) & (~ds.obstacle)

    assert metadata["phase_computed"] is True
    assert np.any(np.abs(ds.raqic_phase[live]) > 0.0)
    assert np.array_equal(ds.raqic_record_action, ds.raqic_readout)
    assert np.array_equal(
        ds.raqic_record_readout[live],
        np.argmax(ds.raqic_probabilities[live], axis=1),
    )
    assert "raqic_trace_error" in ds.arrays
    assert "raqic_min_eigenvalue" in ds.arrays
    assert "raqic_audit_flags" in ds.arrays


def test_stage_parity_event_payloads_are_tolerant_but_identity_is_exact() -> None:
    from owl.science.stage_parity import StateTraceSnapshot, compare_snapshots

    cpu_event = {
        "kind": "ingestion",
        "tick": 1,
        "source": [164, 68],
        "target": [163, 69],
        "payload": {
            "success": True,
            "probability": 0.4413924217224121,
            "resource_transfer": 0.4344214081764221,
        },
    }
    gpu_event = copy.deepcopy(cpu_event)
    gpu_event["payload"]["probability"] = 0.4413924515247345
    cpu = StateTraceSnapshot("collision", "cpu", {}, (cpu_event,))
    gpu = StateTraceSnapshot("collision", "gpu", {}, (gpu_event,))

    close = compare_snapshots(cpu, gpu, input_hash="x", atol=1e-5, rtol=1e-6)
    assert close.event_comparison["passed"] is True

    gpu_bad = copy.deepcopy(gpu_event)
    gpu_bad["target"] = [163, 70]
    bad = compare_snapshots(
        cpu,
        StateTraceSnapshot("collision", "gpu", {}, (gpu_bad,)),
        input_hash="x",
        atol=1e-5,
        rtol=1e-6,
    )
    assert bad.event_comparison["passed"] is False
    assert bad.event_comparison["first_mismatch_index"] == 0


def test_backend_code_is_provenance_not_scientific_state() -> None:
    from owl.science.stage_parity import flatten_state

    cfg = _small_cfg()
    state = initialize_world(cfg, np.random.default_rng(43))
    ensure_raqic_fields(state, cfg)
    assert state.raqic_backend_code is not None
    state.raqic_backend_code.fill(20)

    flattened = flatten_state(state)
    assert "raqic_backend_code" not in flattened


def test_cpu_and_device_clear_cell_share_identity_and_raqic_terminal_contract() -> None:
    from owl.engine.death import clear_cell
    from owl.gpu.stages.death_gpu import clear_cell_gpu

    cfg = _small_cfg()
    state = initialize_world(cfg, np.random.default_rng(47))
    ensure_raqic_fields(state, cfg)
    position = tuple(int(v) for v in np.argwhere((state.health > 0) & (~state.obstacle))[0])
    y, x = position

    cpu = copy.deepcopy(state)
    gpu_source = copy.deepcopy(state)
    clear_cell(cpu, position)

    ds = OWLDeviceState.from_world_state(gpu_source, cfg, force_backend="numpy")
    dead = np.zeros_like(ds.health, dtype=bool)
    dead[y, x] = True
    clear_cell_gpu(ds, dead)
    gpu = copy.deepcopy(gpu_source)
    ds.write_back_to_cpu(gpu)

    for name in (
        "occupancy",
        "parent_id",
        "lineage_id",
        "readout",
        "raqic_readout",
        "raqic_record_action",
        "raqic_record_readout",
        "raqic_backend_code",
    ):
        assert getattr(cpu, name)[y, x] == getattr(gpu, name)[y, x]
    assert cpu.lineage_id[y, x] == -1
    assert np.array_equal(cpu.raqic_parent_intention[y, x], gpu.raqic_parent_intention[y, x])
    assert cpu.raqic_parent_intention[y, x, int(Action.REST)] == 1.0


def test_decision_stage_contract_includes_raqic_record_evidence() -> None:
    from owl.science.stage_contract import STAGE_CONTRACTS

    decision = next(item for item in STAGE_CONTRACTS if item.name == "decision")
    required = {
        "raqic_record_action",
        "raqic_record_readout",
        "raqic_record_confidence",
        "raqic_score",
        "raqic_trace_error",
        "raqic_min_eigenvalue",
        "raqic_audit_flags",
    }
    assert required.issubset(set(decision.writes))


def test_circular_relative_tolerance_uses_half_turn_scale() -> None:
    # The B200 aggregation residual was 2.6226043701171875e-6 rad,
    # corresponding to 8.35e-7 of a half turn. The configured rtol=1e-6
    # should therefore accept it without changing either tolerance value.
    left = np.array([0.0], dtype=np.float64)
    right = np.array([2.6226043701171875e-6], dtype=np.float64)
    accepted = compare_field(
        "patches.phase",
        left,
        right,
        exact=False,
        atol=9.5367431640625e-7,
        rtol=1e-6,
    )
    rejected = compare_field(
        "patches.phase",
        left,
        np.array([4.0e-6], dtype=np.float64),
        exact=False,
        atol=9.5367431640625e-7,
        rtol=1e-6,
    )

    assert accepted.passed
    assert accepted.max_rel is not None and accepted.max_rel < 1e-6
    assert not rejected.passed


def test_topdown_scientific_trace_preserves_parent_intention_exactly() -> None:
    cfg = _small_cfg()
    state = initialize_world(cfg, np.random.default_rng(53))
    ensure_raqic_fields(state, cfg)
    ds = OWLDeviceState.from_world_state(state, cfg, force_backend="numpy")
    parent = np.zeros_like(ds.raqic_parent_intention)
    parent[..., int(Action.REST)] = 0.2
    parent[..., int(Action.INTEGRATE)] = 0.8
    ds.arrays["raqic_parent_intention"][...] = parent
    ds.metadata["scientific_stage_parity"] = True

    dispatch_parent_context_gpu(ds, cfg, force_global=True)

    assert np.array_equal(ds.raqic_parent_intention, parent)


def test_gpu_tick_end_quiescence_matches_cpu_for_all_terminal_cells() -> None:
    cfg = _small_cfg()
    state = initialize_world(cfg, np.random.default_rng(59))
    ensure_raqic_fields(state, cfg)

    live_positions = np.argwhere((state.health > 0.0) & (~state.obstacle))
    assert live_positions.shape[0] >= 2
    vacated_y, vacated_x = (int(value) for value in live_positions[0])
    live_y, live_x = (int(value) for value in live_positions[1])

    # Simulate a movement-vacated source. It is already empty and therefore is
    # not rediscovered by the death detector, but the CPU tick-end contract
    # still forces all RAQIC distributions to REST.
    state.health[vacated_y, vacated_x] = 0.0
    state.occupancy[vacated_y, vacated_x] = -1

    for name in (
        "raqic_probabilities",
        "raqic_parent_intention",
        "raqic_debug_density_diag",
    ):
        arr = getattr(state, name)
        arr[vacated_y, vacated_x, :] = 0.0
        arr[vacated_y, vacated_x, int(Action.INTEGRATE)] = 1.0

    state.raqic_readout[vacated_y, vacated_x] = int(Action.INTEGRATE)
    state.raqic_record_action[vacated_y, vacated_x] = int(Action.INTEGRATE)
    state.raqic_legacy_shadow_readout[vacated_y, vacated_x] = int(Action.INTEGRATE)

    live_parent_before = np.array(
        state.raqic_parent_intention[live_y, live_x],
        copy=True,
    )

    cpu = copy.deepcopy(state)
    quiesce_dead_raqic_fields(cpu)

    gpu_source = copy.deepcopy(state)
    ds = OWLDeviceState.from_world_state(gpu_source, cfg, force_backend="numpy")
    quiesce_dead_raqic_fields_gpu(ds)
    gpu = copy.deepcopy(gpu_source)
    ds.write_back_to_cpu(gpu)

    for name in (
        "raqic_probabilities",
        "raqic_parent_intention",
        "raqic_debug_density_diag",
        "raqic_readout",
        "raqic_record_action",
        "raqic_legacy_shadow_readout",
    ):
        assert np.array_equal(getattr(cpu, name), getattr(gpu, name))

    assert gpu.raqic_parent_intention[vacated_y, vacated_x, int(Action.REST)] == 1.0
    assert np.sum(gpu.raqic_parent_intention[vacated_y, vacated_x]) == 1.0
    assert np.array_equal(gpu.raqic_parent_intention[live_y, live_x], live_parent_before)


def test_postdecision_graph_manifest_includes_raqic_quiescence() -> None:
    from owl.gpu.graph_safety import build_default_graph_safety_manifest

    manifest = build_default_graph_safety_manifest()
    names = {operation.callable_name for operation in manifest.segments["postdecision"].operations}
    assert "quiesce_dead_raqic_fields_gpu" in names


def test_persistent_and_compatibility_paths_quiesce_before_aggregation() -> None:
    import inspect

    from owl.gpu.full_loop import step_gpu_full
    from owl.gpu.run_context import PersistentOWLDeviceRun

    persistent = inspect.getsource(PersistentOWLDeviceRun._segment_postdecision)
    compatibility = inspect.getsource(step_gpu_full)

    for source in (persistent, compatibility):
        clip = source.index("clip_life_fields_gpu")
        quiesce = source.index("quiesce_dead_raqic_fields_gpu")
        aggregate = source.index("aggregate_patches_gpu", quiesce)
        assert clip < quiesce < aggregate
