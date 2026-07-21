from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from owl.cadc.config import load_phase4_config
from owl.core.config import load_config
from owl.core.init import initialize_world
from owl.gpu.graph_certification import certify_graph_status
from owl.gpu.shadow_audit import CPUShadowAuditor
from owl.runtime.capabilities import RuntimeCapabilities
from owl.runtime.execution_plan import compile_execution_plan
from owl.viz.event_bus import VisualEvent, VisualEventBuffer, VisualEventType
from owl_raqic.qiskit_backend.result_decode import (
    ActionBitLayout,
    counts_to_action_probabilities,
)
from owl_raqic.qiskit_backend.workload_planner import estimate_qiskit_workload

ROOT = Path(__file__).resolve().parents[1]


def capabilities(
    *,
    cuda: int = 0,
    aer_gpu: bool = False,
    pygame: bool = False,
    vispy: bool = False,
    nccl: bool = False,
):
    return RuntimeCapabilities(
        cupy_available=bool(cuda),
        cuda_device_count=int(cuda),
        qiskit_available=aer_gpu,
        aer_available=aer_gpu,
        aer_gpu_available=aer_gpu,
        pygame_available=pygame,
        vispy_available=vispy,
        nccl_available=nccl,
        details={},
    )


@pytest.mark.parametrize(
    "path",
    sorted((ROOT / "configs").glob("*.yaml")),
    ids=lambda path: path.name,
)
def test_every_shipped_config_loads(path):
    if path.name.startswith("cadc_phase4_"):
        assert load_phase4_config(path) is not None
    else:
        assert load_config(path) is not None


def test_execution_plan_routes_persistent_main_path():
    cfg = load_config(ROOT / "configs/gpu_v09_persistent_small.yaml")
    plan = compile_execution_plan(cfg, capabilities())
    assert plan.simulation_backend == "gpu_persistent"
    assert plan.decision_backend == "raqic_dense_gpu"
    assert plan.fallback_allowed


def test_execution_plan_routes_full_graph():
    cfg = load_config(ROOT / "configs/gpu_v09_full_graph_small.yaml")
    plan = compile_execution_plan(cfg, capabilities(cuda=1))
    assert plan.simulation_backend == "gpu_graph"
    assert plan.graph_requirement == "full_tick"


def test_execution_plan_routes_per_ow_qiskit():
    cfg = load_config(ROOT / "configs/qiskit_per_ow_static_small.yaml")
    plan = compile_execution_plan(cfg, capabilities(cuda=1, aer_gpu=True))
    assert plan.decision_backend == "raqic_qiskit_per_ow"
    assert plan.qiskit_policy.per_ow


def test_execution_plan_routes_multi_gpu():
    cfg = load_config(ROOT / "configs/gpu_v09_multi_gpu_small.yaml")
    plan = compile_execution_plan(cfg, capabilities(cuda=2, nccl=True))
    assert plan.simulation_backend == "gpu_multi"
    assert plan.device_ids == (0, 1)


def test_graph_certificate_requires_capture_and_replay():
    bad = certify_graph_status(
        {
            "coverage": {
                "required_segments": ["a", "b"],
                "captured_segments": ["a"],
                "replay_counts": {"a": 2, "b": 0},
                "uncovered_reasons": {"b": "not captured"},
            },
            "invalidation_count": 0,
            "capture_contract": {
                "pointer_snapshot_enabled": True,
                "captured_segments_pointer_stable": True,
            },
        },
        requirement="full_tick",
    )
    assert not bad.passed
    good = certify_graph_status(
        {
            "coverage": {
                "required_segments": ["a", "b"],
                "captured_segments": ["a", "b"],
                "replay_counts": {"a": 2, "b": 2},
                "uncovered_reasons": {},
            },
            "invalidation_count": 0,
            "capture_contract": {
                "pointer_snapshot_enabled": True,
                "captured_segments_pointer_stable": True,
            },
        },
        requirement="full_tick",
    )
    assert good.passed


def test_critical_visual_event_survives_capacity_pressure():
    bus = VisualEventBuffer(capacity=4)
    for x in range(4):
        bus.add(
            VisualEvent(1, VisualEventType.MOVE, 0, x),
            replace_lower_priority=True,
        )
    bus.add(
        VisualEvent(1, VisualEventType.AUDIT_FAILURE, 1, 1),
        replace_lower_priority=True,
    )
    assert VisualEventType.AUDIT_FAILURE in bus.event_types()


def test_qiskit_count_endianness_maps_basis_states():
    layout = ActionBitLayout((0, 1, 2), little_endian=True)
    for action in range(8):
        # Qiskit count strings are displayed with the highest classical bit
        # on the left. The decoder owns this conversion.
        key = format(action, "03b")
        row = counts_to_action_probabilities({key: 100}, layout, 8)
        assert int(np.argmax(row)) == action


def test_qiskit_workload_estimate_accounts_for_all_rows():
    estimate = estimate_qiskit_workload(
        ow_rows=100,
        action_count=22,
        family_count=3,
        chunk_size=16,
        shots=4096,
        exact_family_count=1,
    )
    assert estimate.ow_rows == 100
    assert estimate.circuit_rows == 300
    assert estimate.simulator_jobs >= 7


def test_dense_numpy_shadow_advances_one_tick():
    cfg = load_config(ROOT / "configs/gpu_v09_hybrid_shadow.yaml")
    rng = np.random.default_rng(cfg.world.seed)
    state = initialize_world(cfg, rng)
    auditor = CPUShadowAuditor(
        cfg,
        ticks=(1,),
        strict=False,
        reference_mode="dense_numpy_exact",
    )
    result = auditor.run_cpu_reference(state, tick=1)
    assert int(result.tick) == 1


def test_main_cli_persistent_allocates_one_device_state(monkeypatch, tmp_path):
    module = importlib.import_module("owl.experiments.run_single")
    cfg = load_config(ROOT / "configs/gpu_v09_persistent_small.yaml")
    cfg.world.max_steps = 1
    cfg.recording.enabled = False
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(cfg.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "detect_runtime_capabilities", lambda: capabilities())

    from owl.gpu.device_state import OWLDeviceState

    original = OWLDeviceState.from_world_state
    created = 0

    def counted(*args, **kwargs):
        nonlocal created
        created += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(OWLDeviceState, "from_world_state", counted)
    output = module.run_single(config_path)
    assert output.exists()
    assert created == 1
