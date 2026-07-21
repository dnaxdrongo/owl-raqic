import numpy as np

from owl.core.config import load_config
from owl.core.init import initialize_world
from owl.viz.gpu_compositor import compose_frame_cpu


def test_cpu_compositor_frame_shape_and_determinism():
    cfg = load_config("configs/gpu_v07_persistent_small.yaml")
    state = initialize_world(cfg, np.random.default_rng(0))
    a = compose_frame_cpu(state)
    b = compose_frame_cpu(state)
    assert a.shape == (cfg.world.height, cfg.world.width, 4)
    assert a.dtype == np.uint8
    assert np.array_equal(a, b)
