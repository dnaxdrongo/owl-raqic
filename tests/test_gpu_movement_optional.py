import numpy as np

from owl.core.actions import Action
from owl.core.config import SimulationConfig
from owl.core.init import initialize_world
from owl.gpu.device_state import OWLDeviceState
from owl.gpu.stages.movement_gpu import apply_movement_gpu


def test_movement_gpu_stage_numpy_fallback_single_move():
    cfg = SimulationConfig(world={"height": 10, "width": 10, "patch_size": 5})
    rng = np.random.default_rng(1)
    state = initialize_world(cfg, rng)
    state.health[:] = 0
    state.obstacle[:] = False
    state.health[5, 5] = 1
    state.resource[5, 5] = 0.9
    state.readout[:] = int(Action.REST)
    state.readout[5, 5] = int(Action.MOVE_E)
    state.possibility[:] = 0
    state.possibility[..., int(Action.REST)] = 1
    state.possibility[5, 5, int(Action.MOVE_E)] = 1
    ds = OWLDeviceState.from_world_state(state, cfg, strict=False, allow_fallback=True)
    out = apply_movement_gpu(ds, cfg)
    assert out["moved"] >= 0
    arr = ds.backend.asnumpy(ds.health)
    assert np.count_nonzero(arr > 0) == 1
