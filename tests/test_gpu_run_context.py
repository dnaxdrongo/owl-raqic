from owl.core.config import load_config
from owl.gpu.run_context import PersistentOWLDeviceRun


def test_persistent_context_numpy_fallback_runs_small():
    cfg = load_config("configs/gpu_v07_persistent_small.yaml")
    run = PersistentOWLDeviceRun.from_config(cfg)
    assert run.ds.backend.name in {"numpy", "cupy"}
    diag = run.step()
    assert diag["persistent"] is True
    state = run.checkpoint()
    assert state.tick == 1
    assert run.metrics
