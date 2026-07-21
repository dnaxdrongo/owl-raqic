import numpy as np

from owl.core.advanced import ensure_advanced_fields
from owl.core.config import load_config
from owl.core.init import initialize_world
from owl.gpu.device_state import OWLDeviceState
from owl.gpu.slabs import FieldSlabManager


def test_persistent_slab_views_roundtrip():
    cfg = load_config("configs/gpu_v07_persistent_small.yaml")
    cfg.raqic.full_gpu_strict = False
    cfg.raqic.fallback_on_backend_error = True
    state = initialize_world(cfg, np.random.default_rng(cfg.world.seed))
    ensure_advanced_fields(state, cfg)
    ds = OWLDeviceState.from_world_state(state, cfg, strict=False, allow_fallback=True)
    before = np.asarray(ds.backend.asnumpy(ds.arrays["health"])).copy()
    manager = FieldSlabManager.attach(ds)
    manager.assert_views_current(ds)
    ds.arrays["health"][0, 0] = 0.123
    manager.assert_views_current(ds)
    assert abs(float(ds.backend.asnumpy(ds.arrays["health"][0, 0])) - 0.123) < 1e-6
    assert before.shape == ds.arrays["health"].shape
