from __future__ import annotations

from pathlib import Path

from owl.viz.backends.headless_backend import HeadlessVisualBackend
from owl.viz.camera import CameraState, fit_world
from owl.viz.frame_model import VisualFrame, VisualSelection
from owl.viz.scene import build_visual_scene
from owl.viz.themes import get_theme
from tests.visual_overhaul_helpers import synthetic_snapshot


def test_headless_backend_renders_scene_and_marks_renderer(tmp_path: Path) -> None:
    snapshot = synthetic_snapshot()
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
    frame = VisualFrame(None, scene=scene, scientific_tick=1, subframe_index=0)
    backend = HeadlessVisualBackend(tmp_path, resolution=(960, 540))
    backend.submit(frame)
    backend.close()
    output = tmp_path / "frame_00000001_000.png"
    assert output.exists()
    assert output.stat().st_size > 1000
    assert frame.metadata["renderer"] in {"pygame_scene", "pillow_fallback"}
