from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

import numpy as np

from owl.core.actions import Action
from owl.viz.action_animation import resolve_action_context, sample_action_animation
from owl.viz.adaptive_lod import adaptive_lod
from owl.viz.animation import EffectInstance, SpritePose, ease_in_out
from owl.viz.camera import CameraState, cull_positions, world_to_screen
from owl.viz.dynamic_sprites import sprite_states_from_snapshot
from owl.viz.environment_sprites import (
    EnvironmentInstance,
    environment_instances,
)
from owl.viz.event_bus import VisualEvent
from owl.viz.frame_model import VisualSelection
from owl.viz.sprite_state import SpriteDescriptor, SpriteStatus
from owl.viz.themes import Theme
from owl.viz.trait_color import (
    transform_accessibility_color,
    transform_perceptual_color,
)
from owl.viz.visual_snapshot import VisualSnapshot


@dataclass(frozen=True)
class VisualSpriteInstance:
    ow_id: int
    descriptor: SpriteDescriptor
    status: SpriteStatus
    action: Action
    pose: SpritePose
    effects: tuple[EffectInstance, ...]
    screen_position: tuple[float, float]
    cell_pixels: float
    layer: int
    world_position: tuple[float, float]
    communication_channel: int = -1


@dataclass(frozen=True)
class VisualScene:
    tick: int
    subframe_index: int
    subframe_count: int
    camera: CameraState
    background_rgba: tuple[int, int, int, int]
    environment: tuple[EnvironmentInstance, ...]
    sprites: tuple[VisualSpriteInstance, ...]
    effects: tuple[EffectInstance, ...]
    overlays: tuple[Any, ...]
    hud: Any
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))


def _visible_mask(snapshot: VisualSnapshot, camera: CameraState) -> np.ndarray:
    health = np.asarray(snapshot.field("health"), dtype=float)
    obstacle = np.asarray(
        snapshot.arrays.get("obstacle", np.zeros(snapshot.world_shape)), dtype=bool
    )
    mask = (health > 0) & (~obstacle)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.zeros(snapshot.world_shape, dtype=bool)
    visible = cull_positions(camera, coords, margin_cells=2.0)
    output = np.zeros(snapshot.world_shape, dtype=bool)
    chosen = coords[visible]
    if chosen.size:
        output[chosen[:, 0], chosen[:, 1]] = True
    return output


def _priority(effect: EffectInstance) -> tuple[int, float, int]:
    critical = effect.kind in {
        "birth_bridge",
        "ingest_lunge",
        "split_membrane",
        "merge_bridge",
        "movement_recoil",
    }
    return (1 if critical else 0, effect.intensity, -effect.seed)


def build_visual_scene(
    previous: VisualSnapshot | None,
    current: VisualSnapshot,
    progress: float,
    camera: CameraState,
    selection: VisualSelection,
    visual_events: Sequence[VisualEvent],
    *,
    theme: Theme,
    subframe_index: int = 0,
    subframe_count: int = 1,
    max_high_detail_effects: int = 4096,
    visual_seed: int = 0,
    accessibility_mode: str = "standard",
    trait_color_mode: str = "raw_hex",
    show_environment_sprites: bool = True,
    show_patch_overlay: bool = True,
) -> VisualScene:
    visible_mask = _visible_mask(current, camera)
    sprite_entries = sprite_states_from_snapshot(
        current,
        visible_mask=visible_mask,
        selected_id=selection.selected_ow_id,
    )
    instances: list[VisualSpriteInstance] = []
    effects: list[EffectInstance] = []
    lod = adaptive_lod(camera.cell_pixels, 0.0)
    previous_ids = set(previous.id_to_position) if previous is not None else set()
    for (y, x), state in sprite_entries:
        descriptor = state.descriptor
        display_color = descriptor.trait_color
        if trait_color_mode == "perceptual":
            display_color = transform_perceptual_color(display_color)
        elif trait_color_mode != "raw_hex":
            raise ValueError(f"unknown trait color mode: {trait_color_mode}")
        if accessibility_mode != "standard":
            display_color = transform_accessibility_color(
                display_color,
                accessibility_mode,
            )
        if display_color is not descriptor.trait_color:
            descriptor = replace(descriptor, trait_color=display_color)
        ow_id = descriptor.ow_id
        context = resolve_action_context(
            previous,
            current,
            visual_events,
            ow_id,
            state.action,
            visual_seed=visual_seed,
        )
        pose, action_effects = sample_action_animation(state.action, context, progress)
        if previous is not None and ow_id not in previous_ids:
            birth_progress = ease_in_out(progress)
            pose = replace(
                pose,
                scale=max(0.15, birth_progress),
                alpha=birth_progress,
            )
            red, green, blue = descriptor.trait_color.rendered_rgb
            action_effects = action_effects + (
                EffectInstance(
                    kind="birth_glow",
                    source=(float(y), float(x)),
                    target=None,
                    color=(red, green, blue, 225),
                    progress=birth_progress,
                    intensity=1.0,
                    layer=33,
                    seed=(int(ow_id) * 2654435761 + int(current.tick)) & 0x7FFFFFFF,
                ),
            )
        screen = world_to_screen(camera, pose.position[0], pose.position[1])
        instances.append(
            VisualSpriteInstance(
                ow_id=ow_id,
                descriptor=descriptor,
                status=state.status,
                action=state.action,
                pose=pose,
                effects=action_effects,
                screen_position=screen,
                cell_pixels=camera.cell_pixels,
                layer=30,
                world_position=(float(y), float(x)),
                communication_channel=state.communication_channel,
            )
        )
        if selection.include_effects and (lod != "overview" or state.status.selected):
            effects.extend(action_effects)

    if previous is not None:
        vanished_ids = previous_ids.difference(current.id_to_position)
        if vanished_ids:
            ghost_mask = np.zeros(previous.world_shape, dtype=bool)
            for vanished_id in vanished_ids:
                position = previous.position_of(vanished_id)
                if position is not None:
                    ghost_mask[position] = True
            ghost_entries = sprite_states_from_snapshot(
                previous,
                visible_mask=ghost_mask,
                selected_id=selection.selected_ow_id,
            )
            fade = max(0.0, 1.0 - ease_in_out(progress))
            for (y, x), state in ghost_entries:
                descriptor = state.descriptor
                display_color = descriptor.trait_color
                if trait_color_mode == "perceptual":
                    display_color = transform_perceptual_color(display_color)
                if accessibility_mode != "standard":
                    display_color = transform_accessibility_color(
                        display_color,
                        accessibility_mode,
                    )
                if display_color is not descriptor.trait_color:
                    descriptor = replace(descriptor, trait_color=display_color)
                ghost_status = replace(
                    state.status,
                    health_fraction=0.0,
                    resource_fraction=0.0,
                    selected=False,
                )
                pose = SpritePose(
                    position=(float(y), float(x)),
                    scale=max(0.18, fade),
                    alpha=fade * 0.72,
                    squash=(1.0 + progress * 0.25, 1.0 - progress * 0.45),
                )
                instances.append(
                    VisualSpriteInstance(
                        ow_id=descriptor.ow_id,
                        descriptor=descriptor,
                        status=ghost_status,
                        action=Action.REST,
                        pose=pose,
                        effects=(),
                        screen_position=world_to_screen(camera, float(y), float(x)),
                        cell_pixels=camera.cell_pixels,
                        layer=25,
                        world_position=(float(y), float(x)),
                    )
                )

    if len(effects) > max_high_detail_effects:
        effects.sort(key=_priority, reverse=True)
        effects = effects[:max_high_detail_effects]

    environment: list[EnvironmentInstance] = []
    if show_environment_sprites:
        environment.extend(environment_instances(current, camera, theme))
    metadata = MappingProxyType(
        {
            "visible_ow_count": len(instances),
            "effect_count": len(effects),
            "environment_count": len(environment),
            "world_shape": current.world_shape,
            "selected_ow_id": selection.selected_ow_id,
            "snapshot_source": current.metadata.get("source", "unknown"),
            "show_patch_overlay": bool(show_patch_overlay),
            "show_environment_sprites": bool(show_environment_sprites),
            "trait_color_mode": str(trait_color_mode),
            "accessibility_mode": str(accessibility_mode),
            "lod": lod,
        }
    )
    return VisualScene(
        tick=current.tick,
        subframe_index=int(subframe_index),
        subframe_count=max(1, int(subframe_count)),
        camera=camera,
        background_rgba=theme.empty_space,
        environment=tuple(sorted(environment, key=lambda item: item.layer)),
        sprites=tuple(instances),
        effects=tuple(sorted(effects, key=lambda item: item.layer)),
        overlays=(),
        hud=None,
        metadata=metadata,
    )
