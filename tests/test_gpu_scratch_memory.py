from owl.core.config import load_config
from owl.gpu.backend import get_array_backend
from owl.gpu.scratch import ScratchManager


def test_scratch_manager_specs_and_allocates_numpy():
    cfg = load_config("configs/gpu_v07_persistent_small.yaml")
    backend = get_array_backend(strict=False, allow_fallback=True)
    sm = ScratchManager.for_config(backend, cfg)
    assert sm.spec_bytes() > 0
    sm.allocate_all()
    assert sm.memory_bytes() > 0
    assert sm.get("rgba_frame").shape[-1] == 4
