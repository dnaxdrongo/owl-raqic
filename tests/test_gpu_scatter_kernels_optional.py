import numpy as np

from owl.core.config import load_config
from owl.core.init import initialize_world
from owl.gpu.device_state import OWLDeviceState
from owl.gpu.kernels.scatter_kernels import FieldSlabManager


def test_field_slab_pack_unpack_numpy():
    cfg = load_config("configs/gpu_v07_persistent_small.yaml")
    state = initialize_world(cfg, np.random.default_rng(0))
    ds = OWLDeviceState.from_world_state(state, cfg, strict=False, allow_fallback=True)
    mgr = FieldSlabManager(ds)
    names = tuple(n for n in ("health", "resource", "memory") if n in ds.arrays)
    slab = mgr.pack(names)
    assert slab.slab.shape[0] == len(names)
    mgr.unpack(slab)
    assert "health" in ds.arrays
