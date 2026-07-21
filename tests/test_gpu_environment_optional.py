import numpy as np

from owl.core.config import SimulationConfig
from owl.core.init import initialize_world
from owl.gpu.device_state import OWLDeviceState
from owl.gpu.stages.environment_gpu import update_environment_gpu


def test_environment_gpu_stage_numpy_fallback_finite():
    cfg = SimulationConfig()
    rng = np.random.default_rng(123)
    state = initialize_world(cfg, rng)
    ds = OWLDeviceState.from_world_state(state, cfg, strict=False, allow_fallback=True)
    update_environment_gpu(ds, cfg)
    assert np.isfinite(ds.backend.asnumpy(ds.food)).all()
    assert np.isfinite(ds.backend.asnumpy(ds.toxin)).all()
