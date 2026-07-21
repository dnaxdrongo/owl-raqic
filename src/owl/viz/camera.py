from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from owl.viz.visual_snapshot import VisualSnapshot


@dataclass
class CameraState:
    viewport: tuple[int, int, int, int]
    world_shape: tuple[int, int]
    center: tuple[float, float]
    zoom: float
    follow_ow_id: int | None = None
    mode: str = "fit"
    min_zoom: float = 0.5
    max_zoom: float = 64.0

    @property
    def cell_pixels(self) -> float:
        return float(self.zoom)


def world_to_screen(camera: CameraState, y: float, x: float) -> tuple[float, float]:
    vx, vy, vw, vh = camera.viewport
    px = vx + vw / 2.0 + (float(x) + 0.5 - camera.center[1]) * camera.zoom
    py = vy + vh / 2.0 + (float(y) + 0.5 - camera.center[0]) * camera.zoom
    return px, py


def screen_to_world(camera: CameraState, px: float, py: float) -> tuple[float, float]:
    vx, vy, vw, vh = camera.viewport
    x = camera.center[1] + (float(px) - vx - vw / 2.0) / camera.zoom - 0.5
    y = camera.center[0] + (float(py) - vy - vh / 2.0) / camera.zoom - 0.5
    return y, x


def visible_world_bounds(camera: CameraState) -> tuple[float, float, float, float]:
    vx, vy, vw, vh = camera.viewport
    y0, x0 = screen_to_world(camera, vx, vy)
    y1, x1 = screen_to_world(camera, vx + vw, vy + vh)
    return min(y0, y1), min(x0, x1), max(y0, y1), max(x0, x1)


def cull_positions(
    camera: CameraState,
    positions: np.ndarray,
    margin_cells: float = 1.0,
) -> np.ndarray:
    points = np.asarray(positions, dtype=float)
    if points.size == 0:
        return np.zeros((0,), dtype=bool)
    y0, x0, y1, x1 = visible_world_bounds(camera)
    margin = float(margin_cells)
    return (
        (points[:, 0] >= y0 - margin)
        & (points[:, 0] <= y1 + margin)
        & (points[:, 1] >= x0 - margin)
        & (points[:, 1] <= x1 + margin)
    )


def zoom_at(camera: CameraState, factor: float, screen_position: tuple[float, float]) -> None:
    before = screen_to_world(camera, *screen_position)
    camera.zoom = float(np.clip(camera.zoom * factor, camera.min_zoom, camera.max_zoom))
    after = screen_to_world(camera, *screen_position)
    camera.center = (
        camera.center[0] + before[0] - after[0],
        camera.center[1] + before[1] - after[1],
    )
    camera.mode = "free"


def pan(camera: CameraState, dx_pixels: float, dy_pixels: float) -> None:
    camera.center = (
        camera.center[0] - float(dy_pixels) / camera.zoom,
        camera.center[1] - float(dx_pixels) / camera.zoom,
    )
    camera.mode = "free"


def fit_world(camera: CameraState, padding: float = 18.0) -> None:
    _vx, _vy, vw, vh = camera.viewport
    height, width = camera.world_shape
    camera.zoom = float(
        np.clip(
            min((vw - 2.0 * padding) / max(width, 1), (vh - 2.0 * padding) / max(height, 1)),
            camera.min_zoom,
            camera.max_zoom,
        )
    )
    camera.center = (height / 2.0, width / 2.0)
    camera.mode = "fit"


def update_follow_camera(
    camera: CameraState,
    snapshot: VisualSnapshot,
    dt: float,
) -> CameraState:
    if camera.follow_ow_id is None:
        return camera
    position = snapshot.position_of(camera.follow_ow_id)
    if position is None:
        return camera
    smooth = float(np.clip(dt * 8.0, 0.0, 1.0))
    camera.center = (
        camera.center[0] + (position[0] + 0.5 - camera.center[0]) * smooth,
        camera.center[1] + (position[1] + 0.5 - camera.center[1]) * smooth,
    )
    camera.mode = "follow"
    return camera


def minimap_viewport_rect(
    camera: CameraState,
    minimap_rect: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    mx, my, mw, mh = minimap_rect
    y0, x0, y1, x1 = visible_world_bounds(camera)
    height, width = camera.world_shape
    left = mx + int(np.clip(x0 / max(width, 1), 0.0, 1.0) * mw)
    top = my + int(np.clip(y0 / max(height, 1), 0.0, 1.0) * mh)
    right = mx + int(np.clip(x1 / max(width, 1), 0.0, 1.0) * mw)
    bottom = my + int(np.clip(y1 / max(height, 1), 0.0, 1.0) * mh)
    return left, top, max(1, right - left), max(1, bottom - top)
