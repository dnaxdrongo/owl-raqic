import numpy as np

from owl.core.config import load_config
from owl.core.init import initialize_world
from owl.gpu.full_loop import step_gpu_full


def test_gpu_full_small_runs_with_numpy_fallback_config():
    cfg = load_config("configs/gpu_full_small.yaml")
    rng = np.random.default_rng(cfg.world.seed)
    state = initialize_world(cfg, rng)
    step_gpu_full(state, cfg, rng)
    assert state.tick == 1
    assert state.raqic_probabilities is not None
    assert np.allclose(state.raqic_probabilities.sum(axis=-1), 1.0)
    assert np.isfinite(state.health).all()


def test_gpu_full_strict_without_backend_raises_or_runs():
    cfg = load_config("configs/gpu_full_small.yaml").model_copy(deep=True)
    cfg.raqic.fallback_on_backend_error = False
    cfg.raqic.full_gpu_strict = True
    cfg.raqic.strict_gpu = True
    rng = np.random.default_rng(cfg.world.seed)
    state = initialize_world(cfg, rng)
    try:
        step_gpu_full(state, cfg, rng)
    except Exception as exc:
        assert "CuPy" in str(exc) or "GPU" in str(exc) or "cupy" in str(exc)
    else:
        assert state.raqic_probabilities is not None
