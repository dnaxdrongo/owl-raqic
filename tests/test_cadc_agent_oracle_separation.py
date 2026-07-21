from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from owl.core.config import SimulationConfig, load_config
from owl.gpu.run_context import PersistentOWLDeviceRun
from owl.record.cadc_capture import capture_agent_oracle_context
from owl.record.cadc_device_buffer import CADCDeviceBuffer
from owl.record.cadc_schema import CaptureStageCode


class _Device(SimpleNamespace):
    def __getattr__(self, name: str) -> Any:
        arrays = object.__getattribute__(self, "arrays")
        if name in arrays:
            return arrays[name]
        raise AttributeError(name)


def _device(*, waste: float) -> _Device:
    h, w, channels = 3, 4, 8
    shape = (h, w)
    health = np.ones(shape, dtype=np.float32)
    arrays = {
        "health": health,
        "obstacle": np.zeros(shape, dtype=bool),
        "resource": np.full(shape, 0.6, dtype=np.float32),
        "boundary": np.full(shape, 0.7, dtype=np.float32),
        "integration": np.full(shape, 0.5, dtype=np.float32),
        "cooperation": np.full(shape, 0.4, dtype=np.float32),
        "food": np.full(shape, 0.25, dtype=np.float32),
        "food_mean": np.full(shape, 0.35, dtype=np.float32),
        "toxin": np.full(shape, 0.10, dtype=np.float32),
        "toxin_mean": np.full(shape, 0.20, dtype=np.float32),
        "alive_density": np.full(shape, 0.50, dtype=np.float32),
        "memory": np.full(shape, 0.3, dtype=np.float32),
        "phase": np.full(shape, 0.2, dtype=np.float32),
        "signal": np.full((*shape, channels), 0.15, dtype=np.float32),
        "signal_reception": np.full((*shape, channels), 0.12, dtype=np.float32),
        "signal_memory": np.full((*shape, channels), 0.08, dtype=np.float32),
        "waste": np.full(shape, waste, dtype=np.float32),
        "occupancy": np.arange(h * w, dtype=np.int64).reshape(shape),
    }
    return _Device(xp=np, arrays=arrays, tick=7)


def test_agent_and_oracle_namespaces_are_physically_separate() -> None:
    cfg = SimulationConfig()
    cfg.recording.cadc.enabled = True
    low = _device(waste=0.1)
    high = _device(waste=0.9)
    low_buffer = CADCDeviceBuffer.create(low, cfg)
    high_buffer = CADCDeviceBuffer.create(high, cfg)
    capture_agent_oracle_context(low_buffer, low, cfg)
    capture_agent_oracle_context(high_buffer, high, cfg)

    agent_fields = [name for name in low_buffer.arrays if name.startswith("agent_")]
    for name in agent_fields:
        np.testing.assert_array_equal(low_buffer.arrays[name], high_buffer.arrays[name])
    assert not np.array_equal(
        low_buffer.arrays["oracle_waste"], high_buffer.arrays["oracle_waste"]
    )
    assert low_buffer.stage_code == int(CaptureStageCode.POST_SENSING)
    assert low_buffer.tick == 7


def test_device_buffer_enforces_configured_memory_bound() -> None:
    cfg = SimulationConfig()
    cfg.recording.cadc.enabled = True
    cfg.recording.cadc.max_device_buffer_bytes = 1024 * 1024
    device = _device(waste=0.1)
    buffer = CADCDeviceBuffer.create(device, cfg)
    assert 0 < buffer.nbytes <= cfg.recording.cadc.max_device_buffer_bytes


def test_cadc_capture_is_scientifically_observational_on_numpy(tmp_path: Any) -> None:
    control_cfg = load_config("configs/gpu_v07_persistent_small.yaml")
    control_cfg.debug.assert_invariants = False
    control_cfg.raqic.full_gpu_metric_every = 1
    evidence_cfg = control_cfg.model_copy(deep=True)
    evidence_cfg.recording.cadc.enabled = True

    control = PersistentOWLDeviceRun.from_config(
        control_cfg, force_backend="numpy", output_root=tmp_path / "control"
    )
    evidence = PersistentOWLDeviceRun.from_config(
        evidence_cfg, force_backend="numpy", output_root=tmp_path / "evidence"
    )
    try:
        control.step()
        evidence.step()
        assert evidence.cadc_buffer is not None
        assert evidence.cadc_buffer.stage_code == int(CaptureStageCode.TICK_COMMIT)
        assert evidence.cadc_buffer.arrays["policy_legal"].shape[-1] == 22
        living = evidence.cadc_buffer.arrays["pre_alive"] > 0
        assert np.all(evidence.cadc_buffer.arrays["decision_sequence"][living] >= 0)
        assert np.all(evidence.cadc_buffer.arrays["selected_action"][living] >= 0)
        assert np.all(evidence.cadc_buffer.arrays["execution_reason_code"][living] >= 0)
        start = evidence.cadc_buffer.arrays["tick_start"]
        final = np.stack(
            [
                evidence.ds.arrays[name][
                    np.maximum(evidence.cadc_buffer.arrays["current_y"], 0),
                    np.maximum(evidence.cadc_buffer.arrays["current_x"], 0),
                ]
                if name != "signal_emission"
                else np.sum(
                    evidence.ds.signal_emission[
                        np.maximum(evidence.cadc_buffer.arrays["current_y"], 0),
                        np.maximum(evidence.cadc_buffer.arrays["current_x"], 0),
                        :,
                    ],
                    axis=-1,
                )
                for name in evidence.cadc_buffer.contribution_fields
            ],
            axis=-1,
        )
        summed = evidence.cadc_buffer.arrays["contribution_delta"].sum(axis=0)
        np.testing.assert_allclose((final - start)[living], summed[living], atol=1e-7, rtol=0)
        for name in sorted(set(control.ds.arrays) & set(evidence.ds.arrays)):
            np.testing.assert_array_equal(control.ds.arrays[name], evidence.ds.arrays[name])
    finally:
        control.close(checkpoint=False)
        evidence.close(checkpoint=False)
