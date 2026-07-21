from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from owl.viz.frame_model import VisualSelection
from owl.viz.visual_snapshot import (
    hash_snapshot_fields,
    snapshot_from_device_state,
    snapshot_from_world_state,
)
from tests.visual_overhaul_helpers import synthetic_world


class _Backend:
    name = "numpy"

    @staticmethod
    def asnumpy(value: object) -> np.ndarray:
        return cast(np.ndarray, np.asarray(value))


def test_world_snapshot_owns_read_only_copies() -> None:
    world = synthetic_world()
    snapshot = snapshot_from_world_state(world)
    before = hash_snapshot_fields(snapshot)
    world.health[1, 1] = 0.0
    assert snapshot.field("health")[1, 1] > 0.0
    assert hash_snapshot_fields(snapshot) == before
    with pytest.raises(ValueError):
        snapshot.field("health")[1, 1] = 0.0


def test_device_snapshot_explicitly_copies_backend_arrays() -> None:
    world = synthetic_world()
    arrays = {name: value for name, value in vars(world).items() if hasattr(value, "shape")}
    ds = SimpleNamespace(
        arrays=arrays,
        backend=_Backend(),
        tick=world.tick,
        metadata={},
        is_gpu=False,
    )
    selection = VisualSelection(overlay="none")
    snapshot = snapshot_from_device_state(ds, selection)
    assert not np.shares_memory(snapshot.field("health"), arrays["health"])
    assert snapshot.field("health").flags.writeable is False


def test_device_snapshot_world_arrays_take_precedence_over_patch_collisions() -> None:
    world_health: np.ndarray = np.ones((4, 4), dtype=np.float32)
    world_occupancy: np.ndarray = np.arange(16, dtype=np.int64).reshape(4, 4)
    patch_health: np.ndarray = np.zeros((2, 2), dtype=np.float32)

    ds = SimpleNamespace(
        arrays={
            "health": world_health,
            "occupancy": world_occupancy,
        },
        patch_arrays={
            "health": patch_health,
            "raqic_patch_action_phase": np.arange(6, dtype=np.float64).reshape(1, 2, 3),
        },
        global_arrays={
            "health": np.zeros((1,), dtype=np.float32),
            "raqic_global_action_phase": np.arange(3, dtype=np.float64),
        },
        backend=_Backend(),
        xp=np,
        is_gpu=False,
        tick=7,
        metadata={},
    )

    snapshot = snapshot_from_device_state(
        ds,
        field_names=(
            "health",
            "occupancy",
            "raqic_patch_action_phase",
            "raqic_global_action_phase",
        ),
    )

    assert snapshot.world_shape == (4, 4)
    assert snapshot.field("health").shape == (4, 4)
    assert np.array_equal(snapshot.field("health"), world_health)
    assert snapshot.field("occupancy").shape == (4, 4)
    assert snapshot.field("raqic_patch_action_phase").shape == (1, 2, 3)
    assert snapshot.field("raqic_global_action_phase").shape == (3,)


def _world_hashes(world: object) -> dict[str, bytes]:
    import hashlib

    hashes: dict[str, bytes] = {}
    for name, value in vars(world).items():
        if isinstance(value, np.ndarray):
            hashes[name] = hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).digest()
    return hashes


def test_scene_and_headless_submit_do_not_mutate_scientific_world(tmp_path: object) -> None:
    from pathlib import Path

    from owl.viz.backends.headless_backend import HeadlessVisualBackend
    from owl.viz.camera import CameraState, fit_world
    from owl.viz.frame_model import VisualFrame
    from owl.viz.scene import build_visual_scene
    from owl.viz.themes import get_theme

    world = synthetic_world()
    before = _world_hashes(world)
    snapshot = snapshot_from_world_state(world)
    camera = CameraState((0, 0, 640, 480), snapshot.world_shape, (3.0, 3.5), 20.0, mode="fit")
    fit_world(camera)
    scene = build_visual_scene(
        snapshot,
        snapshot,
        0.5,
        camera,
        VisualSelection(overlay="none"),
        (),
        theme=get_theme("owl_dark_neon"),
    )
    backend = HeadlessVisualBackend(Path(str(tmp_path)), resolution=(640, 480))
    backend.submit(VisualFrame(None, scene=scene, scientific_tick=snapshot.tick))
    backend.close()
    assert _world_hashes(world) == before


def test_visual_scene_does_not_advance_numpy_global_rng() -> None:
    from owl.viz.camera import CameraState, fit_world
    from owl.viz.scene import build_visual_scene
    from owl.viz.themes import get_theme

    snapshot = snapshot_from_world_state(synthetic_world())
    camera = CameraState((0, 0, 640, 480), snapshot.world_shape, (3.0, 3.5), 20.0, mode="fit")
    fit_world(camera)
    np.random.seed(1701)
    expected = np.random.random(8)
    np.random.seed(1701)
    build_visual_scene(
        snapshot,
        snapshot,
        0.5,
        camera,
        VisualSelection(overlay="none"),
        (),
        theme=get_theme("owl_dark_neon"),
        visual_seed=9941,
    )
    observed = np.random.random(8)
    assert np.array_equal(observed, expected)
