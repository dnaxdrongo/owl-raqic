from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from owl.core.actions import Action
from owl.gpu.backend import get_array_backend
from owl.gpu.metrics_slab import DeviceMetricSlab


def test_metric_slab_exposes_extension_mechanism_diagnostics() -> None:
    backend = get_array_backend(force="numpy")
    shape = (2, 2)
    actions = len(Action)
    live = np.asarray([[True, True], [False, True]])
    health = live.astype(np.float64)
    obstacle = np.zeros(shape, dtype=bool)
    readout = np.full(shape, int(Action.REST), dtype=np.int32)
    shadow = readout.copy()
    shadow[0, 1] = int(Action.FEED)
    arrays = {
        "raqic_probabilities": np.full((*shape, actions), 1.0 / actions),
        "raqic_readout": readout,
        "raqic_shadow_readout": shadow,
        "raqic_utility_innovation": np.ones((*shape, actions), dtype=np.float64) * 0.1,
        "raqic_phase_alignment": np.ones((*shape, actions), dtype=np.float64) * 0.25,
        "raqic_interference_delta_l1": np.full(shape, 0.2),
        "raqic_policy_kl": np.full(shape, 0.03),
        "raqic_utility_projection_fraction": np.full(shape, 0.4),
        "raqic_utility_score_cosine": np.full(shape, -0.1),
        "raqic_interference_norm_error": np.full(shape, 2e-15),
        "raqic_interference_illegal_mass": np.full(shape, 3e-16),
    }
    ds = SimpleNamespace(
        tick=1,
        health=health,
        obstacle=obstacle,
        food=np.zeros(shape),
        toxin=np.zeros(shape),
        resource=np.ones(shape),
        integration=np.ones(shape) * 0.5,
        arrays=arrays,
        scalars={},
    )
    slab = DeviceMetricSlab.create(backend)
    slab.update(ds)
    decoded = slab.decode(np.asarray(slab.slab), backend="numpy")
    assert np.isclose(decoded["mean_raqic_utility_innovation_l1"], actions * 0.1)
    assert np.isclose(decoded["mean_raqic_phase_alignment"], 0.25)
    assert np.isclose(decoded["mean_raqic_interference_delta_l1"], 0.2)
    assert np.isclose(decoded["max_raqic_interference_norm_error"], 2e-15)
    assert np.isclose(decoded["max_raqic_interference_illegal_mass"], 3e-16)
    assert np.isclose(decoded["raqic_shadow_readout_change_fraction"], 1.0 / 3.0)
