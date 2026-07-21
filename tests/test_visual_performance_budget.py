from __future__ import annotations

import numpy as np

from owl.viz.adaptive_lod import adaptive_lod
from owl.viz.camera import CameraState, cull_positions


def test_lod_never_automatically_removes_ow_identity() -> None:
    assert adaptive_lod(1.0, 1.0) == "overview"
    assert adaptive_lod(6.0, 0.5) == "medium"
    assert adaptive_lod(12.0, 0.1) == "detail"
    assert adaptive_lod(30.0, 0.0) == "focus"


def test_camera_culling_is_bounded() -> None:
    camera = CameraState((0, 0, 400, 300), (200, 200), (100.0, 100.0), 10.0, mode="free")
    positions = np.asarray([[100, 100], [0, 0], [101, 101], [199, 199]], dtype=float)
    mask = cull_positions(camera, positions)
    assert mask.tolist() == [True, False, True, False]
