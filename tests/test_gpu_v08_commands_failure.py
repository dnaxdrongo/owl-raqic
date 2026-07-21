import pytest

from owl.core.config import load_config
from owl.gpu.commands import CommandKind, GPUCommand, GPUCommandQueue
from owl.gpu.failure_injection import FailureInjection, FailureKind, inject_failure
from owl.gpu.run_context import PersistentOWLDeviceRun


def test_command_queue_capacity():
    q = GPUCommandQueue(1)
    q.put(GPUCommand(CommandKind.PAUSE))
    with pytest.raises(OverflowError):
        q.put(GPUCommand(CommandKind.RESUME))


def test_failure_injection_detected_by_invariants():
    cfg = load_config("configs/gpu_v07_persistent_small.yaml")
    run = PersistentOWLDeviceRun.from_config(cfg)
    inject_failure(run.ds, FailureInjection(FailureKind.NAN_HEALTH))
    from owl.gpu.invariants import assert_gpu_full_invariants

    with pytest.raises(AssertionError):
        assert_gpu_full_invariants(run.ds, cfg)
    run.closed = True
