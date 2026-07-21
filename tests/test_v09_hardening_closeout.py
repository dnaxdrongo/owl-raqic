from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from owl.gpu.backend import get_array_backend
from owl.gpu.commands import CommandKind, GPUCommand
from owl.gpu.graph_inputs import DeviceCommandBuffer, apply_device_commands
from owl.gpu.memory_model import build_memory_plan
from owl.record.gpu_async_writer import AsyncGPUWriter
from owl.runtime.production_guard import (
    evaluate_production_readiness,
    write_ready_marker,
)
from owl.viz.controller import VisualController


class _FakeDeviceState:
    def __init__(self):
        self.backend = get_array_backend(force="numpy")
        self.xp = np
        self.is_gpu = False
        self.food = np.zeros((3, 3), dtype=np.float64)
        self.toxin = np.zeros((3, 3), dtype=np.float64)


def test_graph_command_buffer_aggregates_and_applies_numpy():
    ds = _FakeDeviceState()
    buffer = DeviceCommandBuffer.create(ds.backend, capacity=4)
    commands = [
        GPUCommand(
            CommandKind.INJECT_FOOD,
            {"y": 1, "x": 2, "amount": 0.3},
            state_mutating=True,
        ),
        GPUCommand(
            CommandKind.INJECT_FOOD,
            {"y": 1, "x": 2, "amount": 0.4},
            state_mutating=True,
        ),
        GPUCommand(
            CommandKind.INJECT_TOXIN,
            {"y": 0, "x": 0, "amount": 0.2},
            state_mutating=True,
        ),
    ]
    metadata = buffer.encode(commands, ds.backend)
    assert len(metadata) == 2
    apply_device_commands(ds, buffer)
    assert ds.food[1, 2] == pytest.approx(0.7)
    assert ds.toxin[0, 0] == pytest.approx(0.2)
    assert not np.asarray(buffer.active).any()


def test_graph_command_buffer_rejects_nonmutating_and_capacity():
    ds = _FakeDeviceState()
    buffer = DeviceCommandBuffer.create(ds.backend, capacity=1)
    with pytest.raises(PermissionError):
        buffer.encode(
            [GPUCommand(CommandKind.INJECT_FOOD, {"y": 0, "x": 0, "amount": 1.0})],
            ds.backend,
        )
    with pytest.raises(OverflowError):
        buffer.encode(
            [
                GPUCommand(
                    CommandKind.INJECT_FOOD,
                    {"y": 0, "x": 0, "amount": 1.0},
                    state_mutating=True,
                ),
                GPUCommand(
                    CommandKind.INJECT_TOXIN,
                    {"y": 1, "x": 1, "amount": 1.0},
                    state_mutating=True,
                ),
            ],
            ds.backend,
        )


def test_async_writer_bounded_raise_and_flush(tmp_path: Path):
    path = tmp_path / "metrics.jsonl"
    writer = AsyncGPUWriter(path, max_queue=1, overflow_policy="raise")
    writer.write({"tick": 1})
    with pytest.raises(RuntimeError, match="queue is full"):
        writer.write({"tick": 2})
    writer.close()

    running = AsyncGPUWriter(path, max_queue=2, overflow_policy="block").start()
    running.write({"tick": 3})
    running.close()
    assert '"tick": 3' in path.read_text(encoding="utf-8")


def test_memory_plan_deduplicates_numpy_views():
    owner = np.zeros((4, 4), dtype=np.float64)
    ds = SimpleNamespace(
        arrays={"owner": owner, "view": owner[:, :2]},
        patch_arrays={},
        global_arrays={},
        health=owner,
        possibility=np.zeros((4, 4, 3), dtype=np.float64),
    )
    raqic = SimpleNamespace(
        full_gpu_sparse_event_capacity=8,
        full_gpu_visual_event_capacity=8,
        full_gpu_writer_queue_capacity=2,
        full_gpu_recording_level_v07="metrics_only",
        full_gpu_precision="audit64",
    )
    cfg = SimpleNamespace(raqic=raqic)
    plan = build_memory_plan(
        ds,
        cfg,
        scratch_bytes=0,
        slab_layout=None,
        qiskit_policy=None,
        visual_backend="none",
    )
    entries = {item.name: item for item in plan.allocations}
    assert entries["state.owner"].bytes == owner.nbytes
    assert entries["state.view"].bytes == 0
    assert entries["state.view"].shares_storage_with == "state.owner"


def test_visual_selection_updates_without_optional_backend():
    controller = VisualController(
        backend_name="none",
        render_every=1,
        event_capacity=16,
        clutter_budget=8,
        adaptive=True,
        max_slowdown_fraction=0.1,
        theme="owl_dark_neon",
    )
    controller.update_settings(
        {
            "overlay": "raqic",
            "include_events": False,
            "include_glyphs": True,
            "include_debug": True,
        }
    )
    summary = controller.summary()
    assert summary["selection"]["overlay"] == "raqic"
    assert summary["selection"]["include_events"] is False
    controller.close()


def test_production_guard_requires_graph_replay_and_qiskit_accounting(tmp_path: Path):
    plan = SimpleNamespace(
        graph_requirement="full_tick",
        qiskit_policy=SimpleNamespace(per_ow=True),
    )
    metadata = {
        "fallback_count": 0,
        "graph": {
            "coverage": {
                "full_tick": True,
                "required_segments": ["predecision", "decision"],
                "replay_counts": {"predecision": 2, "decision": 2},
            }
        },
        "per_ow_qiskit": {"processed_count": 10, "expected_count": 10},
    }
    readiness = evaluate_production_readiness(
        plan=plan,
        execution_metadata=metadata,
        all_configs_valid=True,
        config_usage_clean=True,
        memory_preflight_passed=True,
        certification_compatible=True,
    )
    assert readiness.passed
    marker = write_ready_marker(readiness, tmp_path / "READY", metadata=metadata)
    assert marker.exists()

    metadata["graph"]["coverage"]["replay_counts"]["decision"] = 0
    failed = evaluate_production_readiness(
        plan=plan,
        execution_metadata=metadata,
        all_configs_valid=True,
        config_usage_clean=True,
        memory_preflight_passed=True,
        certification_compatible=True,
    )
    assert not failed.passed


def test_persistent_slabs_remain_current_after_tick():
    from owl.core.config import load_config
    from owl.gpu.run_context import PersistentOWLDeviceRun

    cfg = load_config("configs/gpu_v09_persistent_small.yaml")
    cfg.world.max_steps = 1
    cfg.recording.enabled = False
    run = PersistentOWLDeviceRun.from_config(cfg, force_backend="numpy")
    try:
        assert run.slab_manager is not None
        run.step()
        run.slab_manager.assert_views_current(run.ds)
        # audit64 applies to RAQIC evidence/workspaces, not the float32
        # executable OWL physical state used by the CPU scientific reference.
        assert run.ds.health.dtype == np.float32
        assert run.ds.phase.dtype == np.float32
        assert run.ds.raqic_probabilities.dtype == np.float64
    finally:
        run.close(checkpoint=False)
