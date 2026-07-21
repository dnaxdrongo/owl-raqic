import numpy as np

from owl.core.config import SimulationConfig
from owl.core.init import initialize_world
from owl.gpu.device_state import OWLDeviceState
from owl.gpu.field_registry import FIELD_REGISTRY


def test_device_state_numpy_fallback_roundtrip():
    cfg = SimulationConfig()
    rng = np.random.default_rng(cfg.world.seed)
    state = initialize_world(cfg, rng)
    ds = OWLDeviceState.from_world_state(state, cfg, strict=False, allow_fallback=True)
    assert ds.health.shape == state.health.shape
    ds.arrays["health"] = ds.health * 0 + 0.5
    ds.write_back_to_cpu(state, fields=["health"])
    assert np.allclose(state.health, 0.5)


def test_field_registry_has_core_life_fields():
    for name in [
        "health",
        "resource",
        "memory",
        "phase",
        "readout",
        "food",
        "toxin",
        "signal",
        "possibility",
    ]:
        assert name in FIELD_REGISTRY
