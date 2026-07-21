from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from owl.viz.camera import CameraState, cull_positions, world_to_screen
from owl.viz.event_bus import VisualEvent, VisualEventType
from owl.viz.themes import Theme
from owl.viz.visual_snapshot import VisualSnapshot


class EnvironmentKind(StrEnum):
    FOOD = "food"
    TOXIN = "toxin"
    WASTE = "waste"
    OBSTACLE = "obstacle"
    BOUNDARY = "boundary"
    SIGNAL = "signal"
    DEAD_SHELL = "dead_shell"


@dataclass(frozen=True)
class EnvironmentInstance:
    kind: EnvironmentKind
    world_position: tuple[float, float]
    screen_position: tuple[float, float]
    value: float
    variant: int
    color: tuple[int, int, int, int]
    cell_pixels: float
    layer: int
    source_id: int = -1


def _variant(y: int, x: int, salt: int = 0) -> int:
    return int(((y * 73856093) ^ (x * 19349663) ^ salt) & 0x7FFFFFFF) % 8


def _visible_coords(mask: np.ndarray, camera: CameraState) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return coords
    return coords[cull_positions(camera, coords, margin_cells=1.0)]


def _overview_environment_instances(
    *,
    food: np.ndarray,
    toxin: np.ndarray,
    waste: np.ndarray,
    signal_strength: np.ndarray,
    obstacle: np.ndarray,
    camera: CameraState,
    theme: Theme,
    tick: int,
) -> tuple[EnvironmentInstance, ...]:
    """Aggregate dense environment fields into one truthful icon per screen block.

    The strongest normalized environmental category wins each block. This is a
    visual LOD rule only: source arrays are never changed and the chosen value is
    still derived directly from the completed scientific snapshot.
    """
    target_pixels = 14.0
    stride = max(2, int(np.ceil(target_pixels / max(camera.cell_pixels, 0.25))))
    height, width = food.shape
    block_rows = (height + stride - 1) // stride
    block_cols = (width + stride - 1) // stride
    padded_shape = (block_rows * stride, block_cols * stride)

    thresholds = np.asarray((0.12, 0.03, 0.03, 0.06), dtype=np.float32)
    raw_fields = np.stack((food, toxin, waste, signal_strength), axis=0).astype(
        np.float32,
        copy=False,
    )
    valid = (~obstacle)[None, ...]
    normalized = np.where(valid, raw_fields / thresholds[:, None, None], 0.0)
    padded = np.zeros((4, *padded_shape), dtype=np.float32)
    padded[:, :height, :width] = normalized
    block_values = padded.reshape(
        4,
        block_rows,
        stride,
        block_cols,
        stride,
    ).max(axis=(2, 4))
    winner = np.argmax(block_values, axis=0)
    strength = np.max(block_values, axis=0)
    kinds = (
        EnvironmentKind.FOOD,
        EnvironmentKind.TOXIN,
        EnvironmentKind.WASTE,
        EnvironmentKind.SIGNAL,
    )
    colors = (theme.food, theme.toxin, theme.waste, theme.signal)
    layers = (14, 16, 15, 18)
    instances: list[EnvironmentInstance] = []
    for block_y, block_x in np.argwhere(strength > 1.0):
        category = int(winner[block_y, block_x])
        y = min(height - 1, int(block_y * stride + stride // 2))
        x = min(width - 1, int(block_x * stride + stride // 2))
        world_position = (float(y), float(x))
        screen_position = world_to_screen(camera, *world_position)
        viewport = camera.viewport
        if not (
            viewport[0] - target_pixels
            <= screen_position[0]
            <= viewport[0] + viewport[2] + target_pixels
            and viewport[1] - target_pixels
            <= screen_position[1]
            <= viewport[1] + viewport[3] + target_pixels
        ):
            continue
        kind = kinds[category]
        value = float(
            np.clip(
                block_values[category, block_y, block_x] * thresholds[category],
                0.0,
                1.0,
            )
        )
        instances.append(
            EnvironmentInstance(
                kind=kind,
                world_position=world_position,
                screen_position=screen_position,
                value=value,
                variant=_variant(y, x, tick),
                color=colors[category],
                cell_pixels=max(camera.cell_pixels, float(stride) * camera.cell_pixels),
                layer=layers[category],
            )
        )

    obstacle_padded = np.zeros(padded_shape, dtype=bool)
    obstacle_padded[:height, :width] = obstacle
    obstacle_blocks = obstacle_padded.reshape(
        block_rows,
        stride,
        block_cols,
        stride,
    ).any(axis=(1, 3))
    for block_y, block_x in np.argwhere(obstacle_blocks):
        y = min(height - 1, int(block_y * stride + stride // 2))
        x = min(width - 1, int(block_x * stride + stride // 2))
        world_position = (float(y), float(x))
        screen_position = world_to_screen(camera, *world_position)
        instances.append(
            EnvironmentInstance(
                kind=EnvironmentKind.OBSTACLE,
                world_position=world_position,
                screen_position=screen_position,
                value=1.0,
                variant=_variant(y, x, tick),
                color=theme.obstacle,
                cell_pixels=max(camera.cell_pixels, float(stride) * camera.cell_pixels),
                layer=10,
            )
        )
    return tuple(instances)


def environment_instances(
    snapshot: VisualSnapshot,
    camera: CameraState,
    theme: Theme,
) -> tuple[EnvironmentInstance, ...]:
    shape = snapshot.world_shape
    obstacle = np.asarray(snapshot.arrays.get("obstacle", np.zeros(shape, dtype=bool)), dtype=bool)
    health = np.asarray(snapshot.arrays.get("health", np.zeros(shape)), dtype=float)
    empty = health <= 0
    food = np.asarray(snapshot.arrays.get("food", np.zeros(shape)), dtype=float)
    toxin = np.asarray(snapshot.arrays.get("toxin", np.zeros(shape)), dtype=float)
    waste = np.asarray(snapshot.arrays.get("waste", np.zeros(shape)), dtype=float)
    signal = snapshot.arrays.get("signal_emission")
    signal_strength = (
        np.max(np.asarray(signal, dtype=float), axis=-1) if signal is not None else np.zeros(shape)
    )
    if camera.cell_pixels < 8.0:
        return _overview_environment_instances(
            food=food,
            toxin=toxin,
            waste=waste,
            signal_strength=signal_strength,
            obstacle=obstacle,
            camera=camera,
            theme=theme,
            tick=snapshot.tick,
        )

    instances: list[EnvironmentInstance] = []

    definitions = (
        (EnvironmentKind.OBSTACLE, obstacle, 1.0, theme.obstacle, 10),
        (EnvironmentKind.FOOD, (food > 0.12) & (~obstacle), 0.12, theme.food, 14),
        (EnvironmentKind.TOXIN, (toxin > 0.03) & (~obstacle), 0.03, theme.toxin, 16),
        (EnvironmentKind.WASTE, (waste > 0.03) & (~obstacle), 0.03, theme.waste, 15),
        (EnvironmentKind.SIGNAL, (signal_strength > 0.06) & (~obstacle), 0.06, theme.signal, 18),
    )
    values = {
        EnvironmentKind.OBSTACLE: obstacle.astype(float),
        EnvironmentKind.FOOD: food,
        EnvironmentKind.TOXIN: toxin,
        EnvironmentKind.WASTE: waste,
        EnvironmentKind.SIGNAL: signal_strength,
    }
    for kind, mask, _threshold, color, layer in definitions:
        for y, x in _visible_coords(mask, camera):
            value = float(np.clip(values[kind][y, x], 0.0, 1.0))
            if kind == EnvironmentKind.FOOD and not empty[y, x] and camera.cell_pixels < 8:
                continue
            instances.append(
                EnvironmentInstance(
                    kind=kind,
                    world_position=(float(y), float(x)),
                    screen_position=world_to_screen(camera, float(y), float(x)),
                    value=value,
                    variant=_variant(int(y), int(x), snapshot.tick),
                    color=color,
                    cell_pixels=camera.cell_pixels,
                    layer=layer,
                )
            )
    return tuple(instances)


def dead_shell_instances(
    previous: VisualSnapshot | None,
    current: VisualSnapshot,
    events: Sequence[VisualEvent],
    camera: CameraState,
    theme: Theme,
) -> tuple[EnvironmentInstance, ...]:
    if previous is None:
        return ()
    current_ids = set(current.id_to_position)
    shells: list[EnvironmentInstance] = []
    death_by_source = {
        event.source_id: event
        for event in events
        if event.event_type in {VisualEventType.DEATH, VisualEventType.INGEST}
    }
    for ow_id, position in previous.id_to_position.items():
        if ow_id in current_ids:
            continue
        event = death_by_source.get(ow_id)
        y, x = position if event is None else (event.y, event.x)
        shells.append(
            EnvironmentInstance(
                kind=EnvironmentKind.DEAD_SHELL,
                world_position=(float(y), float(x)),
                screen_position=world_to_screen(camera, float(y), float(x)),
                value=1.0,
                variant=int(ow_id) % 8,
                color=theme.dead_shell,
                cell_pixels=camera.cell_pixels,
                layer=22,
                source_id=int(ow_id),
            )
        )
    return tuple(shells)
