import numpy as np

from owl.core.actions import Action
from owl.core.config import SimulationConfig
from owl.core.init import initialize_world
from owl.gpu.device_state import OWLDeviceState
from owl.gpu.stages.topology_gpu import apply_topology_events_gpu, detect_topology_events_gpu


def _state_cfg():
    cfg = SimulationConfig(
        world={"height": 10, "width": 10, "patch_size": 5},
        raqic={"full_gpu_strict": False, "full_gpu_no_silent_fallback": False},
    )
    rng = np.random.default_rng(4)
    state = initialize_world(cfg, rng)
    state.health[:] = 0.0
    state.resource[:] = 0.0
    state.boundary[:] = 0.0
    state.integration[:] = 0.0
    state.obstacle[:] = False
    state.readout[:] = int(Action.REST)
    state.parent_id[:] = 0
    return cfg, state


def test_topology_gpu_dense_merge_numpy_fallback():
    cfg, state = _state_cfg()
    state.health[4, 4] = 1.0
    state.health[4, 5] = 0.8
    state.resource[4, 4] = 0.8
    state.resource[4, 5] = 0.6
    state.boundary[4, 4] = state.boundary[4, 5] = 1.0
    state.integration[4, 4] = 0.3
    state.integration[4, 5] = 0.9
    state.occupancy[4, 4] = 44
    state.occupancy[4, 5] = 45
    state.readout[4, 4] = int(Action.MERGE)
    ds = OWLDeviceState.from_world_state(state, cfg, strict=False, allow_fallback=True)
    events = detect_topology_events_gpu(ds, cfg)
    out = apply_topology_events_gpu(ds, cfg, events)
    assert out["merged"] == 1
    parent = ds.backend.asnumpy(ds.parent_id)
    health = ds.backend.asnumpy(ds.health)
    resource = ds.backend.asnumpy(ds.resource)
    assert parent[4, 4] == parent[4, 5]
    assert np.isclose(health[4, 4], health[4, 5])
    assert np.isclose(resource[4, 4] + resource[4, 5], 1.0)  # pooled, clipped


def test_topology_gpu_dense_split_numpy_fallback():
    cfg, state = _state_cfg()
    state.health[5, 5] = 1.0
    state.resource[5, 5] = 0.8
    state.boundary[5, 5] = 1.0
    state.integration[5, 5] = 0.7
    state.occupancy[5, 5] = 55
    state.readout[5, 5] = int(Action.SPLIT)
    ds = OWLDeviceState.from_world_state(state, cfg, strict=False, allow_fallback=True)
    out = apply_topology_events_gpu(ds, cfg)
    assert out["split"] == 1
    health = ds.backend.asnumpy(ds.health)
    parent = ds.backend.asnumpy(ds.parent_id)
    assert np.count_nonzero(health > 0) == 2
    assert np.count_nonzero(parent == 55) == 2


def test_topology_gpu_dense_expel_numpy_fallback():
    cfg, state = _state_cfg()
    state.health[3, 3] = 1.0
    state.resource[3, 3] = 0.7
    state.boundary[3, 3] = 1.0
    state.parent_id[3, 3] = 123
    state.readout[3, 3] = int(Action.EXPEL)
    ds = OWLDeviceState.from_world_state(state, cfg, strict=False, allow_fallback=True)
    out = apply_topology_events_gpu(ds, cfg)
    assert out["expelled"] == 1
    assert int(ds.backend.asnumpy(ds.parent_id)[3, 3]) == 0
