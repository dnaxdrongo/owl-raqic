import numpy as np

from owl.core.config import load_config
from owl.core.init import initialize_world
from owl.gpu.device_state import OWLDeviceState
from owl.gpu.kernels.biology_kernels import fused_biology_update


def test_fused_biology_numpy_fallback_updates_bounds():
    cfg = load_config("configs/gpu_v07_persistent_small.yaml")
    rng = np.random.default_rng(cfg.world.seed)
    state = initialize_world(cfg, rng)
    ds = OWLDeviceState.from_world_state(state, cfg, strict=False, allow_fallback=True)
    out = fused_biology_update(ds, cfg)
    assert out["live"] >= 0
    assert float(ds.backend.asnumpy(ds.health).min()) >= 0.0
    assert float(ds.backend.asnumpy(ds.health).max()) <= 1.0
