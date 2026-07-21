from __future__ import annotations

from owl.viz.camera import CameraState, fit_world
from owl.viz.environment_sprites import EnvironmentKind
from owl.viz.frame_model import VisualSelection
from owl.viz.scene import build_visual_scene
from owl.viz.themes import get_theme
from tests.visual_overhaul_helpers import synthetic_snapshot


def test_scene_keeps_ow_bodies_and_distinct_environment_categories() -> None:
    snapshot = synthetic_snapshot()
    camera = CameraState((0, 0, 900, 600), snapshot.world_shape, (3.0, 3.5), 20.0, mode="fit")
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
    assert len(scene.sprites) == len(snapshot.id_to_position)
    kinds = {item.kind for item in scene.environment}
    assert EnvironmentKind.FOOD in kinds
    assert EnvironmentKind.TOXIN in kinds
    assert EnvironmentKind.WASTE in kinds
    assert EnvironmentKind.OBSTACLE in kinds
    assert all(item.layer < 30 for item in scene.environment)
    assert all(item.layer == 30 for item in scene.sprites)


def test_environment_and_patch_overlay_can_be_disabled() -> None:
    snapshot = synthetic_snapshot()
    camera = CameraState((0, 0, 900, 600), snapshot.world_shape, (3.0, 3.5), 20.0, mode="fit")
    fit_world(camera)
    scene = build_visual_scene(
        snapshot,
        snapshot,
        0.5,
        camera,
        VisualSelection(overlay="none"),
        (),
        theme=get_theme("owl_dark_neon"),
        show_environment_sprites=False,
        show_patch_overlay=False,
    )
    assert scene.environment == ()
    assert scene.metadata["show_patch_overlay"] is False


def test_new_child_uses_its_own_descriptor_and_scales_in() -> None:
    from owl.viz.visual_snapshot import snapshot_from_world_state
    from tests.visual_overhaul_helpers import synthetic_world

    current_world = synthetic_world()
    previous_world = synthetic_world(tick=current_world.tick - 1)
    child_id = int(current_world.occupancy[1, 1])
    previous_world.health[1, 1] = 0.0
    previous_world.occupancy[1, 1] = -1
    previous = snapshot_from_world_state(previous_world)
    current = snapshot_from_world_state(current_world)
    camera = CameraState((0, 0, 900, 600), current.world_shape, (3.0, 3.5), 20.0, mode="fit")
    fit_world(camera)
    scene = build_visual_scene(
        previous,
        current,
        0.4,
        camera,
        VisualSelection(overlay="none"),
        (),
        theme=get_theme("owl_dark_neon"),
    )
    child = next(item for item in scene.sprites if item.ow_id == child_id)
    assert 0.15 <= child.pose.scale < 1.0
    assert child.descriptor.ow_id == child_id
    assert any(effect.kind == "birth_glow" for effect in child.effects)


def test_dead_shell_reuses_previous_ow_descriptor() -> None:
    from owl.viz.dynamic_sprites import sprite_states_from_snapshot
    from owl.viz.visual_snapshot import snapshot_from_world_state
    from tests.visual_overhaul_helpers import synthetic_world

    previous_world = synthetic_world(tick=2)
    current_world = synthetic_world(tick=3)
    dead_id = int(previous_world.occupancy[1, 1])
    current_world.health[1, 1] = 0.0
    current_world.occupancy[1, 1] = -1
    previous = snapshot_from_world_state(previous_world)
    current = snapshot_from_world_state(current_world)
    previous_state = next(
        state
        for _, state in sprite_states_from_snapshot(previous)
        if state.descriptor.ow_id == dead_id
    )
    camera = CameraState((0, 0, 900, 600), current.world_shape, (3.0, 3.5), 20.0, mode="fit")
    fit_world(camera)
    scene = build_visual_scene(
        previous,
        current,
        0.5,
        camera,
        VisualSelection(overlay="none"),
        (),
        theme=get_theme("owl_dark_neon"),
    )
    ghost = next(item for item in scene.sprites if item.ow_id == dead_id)
    assert ghost.layer == 25
    assert 0.0 < ghost.pose.alpha < 1.0
    assert ghost.descriptor.trait_color.raw_hex == previous_state.descriptor.trait_color.raw_hex
