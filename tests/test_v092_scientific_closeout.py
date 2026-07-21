from __future__ import annotations

import copy
import os

import numpy as np

from owl.core.config import load_config
from owl.core.init import initialize_world
from owl.gpu.run_context import PersistentOWLDeviceRun
from owl.gpu.shadow_audit import CPUShadowAuditor
from owl.raqic.state import ensure_raqic_fields
from owl.runtime.capabilities import RuntimeCapabilities
from owl.runtime.execution_plan import compile_execution_plan
from owl.runtime.production_guard import evaluate_production_readiness
from owl.science.action_contract import reproduction_plan


def _numpy_runtime() -> RuntimeCapabilities:
    return RuntimeCapabilities(
        cupy_available=False,
        cuda_device_count=0,
        qiskit_available=False,
        aer_available=False,
        aer_gpu_available=False,
        pygame_available=False,
        vispy_available=False,
        nccl_available=False,
        details={"test": "forced_numpy"},
    )


def _scientific_cfg():
    cfg = load_config("configs/gpu_v09_hybrid_shadow.yaml")
    cfg.world.max_steps = 10
    cfg.raqic.mode = "gpu_full"
    cfg.raqic.full_gpu_transfer_policy = "persistent_mirror"
    cfg.raqic.full_gpu_execution_tier = "persistent"
    cfg.raqic.full_gpu_graph_mode = "off"
    cfg.raqic.full_gpu_graph_requirement = "allow_partial"
    cfg.raqic.full_gpu_cpu_shadow_ticks = 0
    cfg.raqic.qiskit_decision_mode = "off"
    cfg.raqic.gpu_validate_qiskit = False
    cfg.raqic.full_gpu_validation_every = 0
    cfg.raqic.full_gpu_visual_backend = "none"
    cfg.raqic.full_gpu_strict = False
    cfg.raqic.strict_gpu = False
    cfg.raqic.fallback_on_backend_error = True
    cfg.raqic.full_gpu_no_silent_fallback = False
    cfg.visualization.enabled = False
    cfg.recording.enabled = False
    return cfg


def test_scientific_cpu_and_numpy_pipeline_match_for_ten_ticks():
    cfg = _scientific_cfg()
    base = initialize_world(cfg, np.random.default_rng(cfg.world.seed))
    ensure_raqic_fields(base, cfg)
    cpu_state = copy.deepcopy(base)
    plan = compile_execution_plan(cfg, _numpy_runtime())
    run = PersistentOWLDeviceRun.from_config(
        cfg, initial_state=copy.deepcopy(base), plan=plan, force_backend="numpy"
    )
    auditor = CPUShadowAuditor(
        cfg,
        ticks=tuple(range(1, 11)),
        tolerance=1e-8,
        strict=False,
        reference_mode="scientific_cpu",
    )
    try:
        for tick in range(1, 11):
            auditor.run_cpu_reference(cpu_state, tick=tick - 1)
            run.step()
            array_state = copy.deepcopy(run.checkpoint(count=False))
            parity = auditor.compare(cpu_state, array_state, tick=tick)
            assert parity.passed, parity.to_dict()
            assert all(parity.exact_event_matches.values())
            assert max(parity.field_residuals.values(), default=0.0) <= max(
                parity.field_tolerances.values(), default=1e-6
            )
    finally:
        run.close(checkpoint=False)


def test_reproduction_plan_is_iteration_and_backend_order_invariant():
    shape = (4, 4)
    readout = np.zeros(shape, dtype=np.int32)
    # Two parents compete for the same neighborhood while retaining other
    # possible targets. The plan must be repeatable from the same state.
    from owl.core.actions import Action

    readout[1, 1] = int(Action.REPRODUCE)
    readout[1, 3] = int(Action.REPRODUCE)
    health = np.zeros(shape, dtype=np.float64)
    health[1, 1] = health[1, 3] = 1.0
    resource = np.ones(shape, dtype=np.float64)
    boundary = np.ones(shape, dtype=np.float64)
    integration = np.ones(shape, dtype=np.float64)
    reproduction_rate = np.ones(shape, dtype=np.float64)
    obstacle = np.zeros(shape, dtype=bool)
    occupancy = np.full(shape, -1, dtype=np.int64)
    occupancy[1, 1] = 10
    occupancy[1, 3] = 20
    kwargs = {
        "min_resource": 0.1,
        "min_health": 0.1,
        "min_boundary": 0.1,
        "min_integration": 0.1,
        "boundary_mode": "toroidal",
        "seed": 123,
        "tick": 4,
        "xp": np,
    }
    first = reproduction_plan(
        readout,
        health,
        resource,
        boundary,
        integration,
        reproduction_rate,
        obstacle,
        occupancy,
        **kwargs,
    )
    second = reproduction_plan(
        readout.copy(),
        health.copy(),
        resource.copy(),
        boundary.copy(),
        integration.copy(),
        reproduction_rate.copy(),
        obstacle.copy(),
        occupancy.copy(),
        **kwargs,
    )
    for name in ("parent_y", "parent_x", "target_y", "target_x", "accepted", "priority"):
        assert np.array_equal(getattr(first, name), getattr(second, name))


def test_execution_plan_hash_is_stable_and_behavior_sensitive():
    cfg = _scientific_cfg()
    plan_a = compile_execution_plan(cfg, _numpy_runtime())
    plan_b = compile_execution_plan(cfg.model_copy(deep=True), _numpy_runtime())
    assert plan_a.plan_hash == plan_b.plan_hash
    changed = cfg.model_copy(deep=True)
    changed.raqic.full_gpu_record_every += 1
    plan_c = compile_execution_plan(changed, _numpy_runtime())
    assert plan_c.plan_hash != plan_a.plan_hash


def test_strict_production_guard_fails_on_missing_evidence():
    cfg = _scientific_cfg()
    plan = compile_execution_plan(cfg, _numpy_runtime())
    readiness = evaluate_production_readiness(
        plan=plan,
        execution_metadata={"fallback_count": 0},
        evidence={"all_configs_valid": True},
    )
    assert not readiness.passed
    assert "scientific_cpu_shadow_passed" in readiness.failures
    assert "memory_estimate_and_actual_peak_passed" in readiness.failures


@__import__("pytest").mark.skipif(
    os.environ.get("OWL_RUN_LONG_CERTIFICATION") != "1",
    reason="100-tick trajectory is executed by the v0.9.2 certification ladder",
)
def test_scientific_cpu_and_numpy_pipeline_preserve_discrete_trajectory_for_100_ticks():
    cfg = _scientific_cfg()
    cfg.world.max_steps = 100
    base = initialize_world(cfg, np.random.default_rng(cfg.world.seed))
    ensure_raqic_fields(base, cfg)
    cpu_state = copy.deepcopy(base)
    plan = compile_execution_plan(cfg, _numpy_runtime())
    run = PersistentOWLDeviceRun.from_config(
        cfg, initial_state=copy.deepcopy(base), plan=plan, force_backend="numpy"
    )
    auditor = CPUShadowAuditor(
        cfg,
        ticks=tuple(range(1, 101)),
        tolerance=1e-8,
        strict=False,
        reference_mode="scientific_cpu",
    )
    try:
        for tick in range(1, 101):
            auditor.run_cpu_reference(cpu_state, tick=tick - 1)
            run.step()
            array_state = copy.deepcopy(run.checkpoint(count=False))
            parity = auditor.compare(cpu_state, array_state, tick=tick)
            assert all(parity.exact_event_matches.values()), parity.to_dict()
            assert parity.passed, parity.to_dict()
    finally:
        run.close(checkpoint=False)
