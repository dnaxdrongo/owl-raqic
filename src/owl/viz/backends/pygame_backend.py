from __future__ import annotations

from typing import Any

import numpy as np

from owl.viz.camera import pan, zoom_at
from owl.viz.sprite_atlas import SpriteAtlas
from owl.viz.themes import get_theme


class PygameVisualBackend:
    def __init__(
        self,
        title: str = "OWL + RAQIC Interpretability",
        scale: int = 8,
        theme: str = "owl_dark_neon",
        window_size: tuple[int, int] = (1920, 1080),
        resizable: bool = True,
        fps: int = 30,
        atlas_max_entries: int = 8192,
    ) -> None:
        try:
            import pygame
        except Exception as exc:
            raise RuntimeError("Pygame visual backend is unavailable") from exc
        from owl.viz.backends.pygame_scene_renderer import PygameSceneRenderer

        self.pygame = pygame
        pygame.init()
        self.title = title
        self.scale = max(1, int(scale))
        self.window_size = (max(640, int(window_size[0])), max(480, int(window_size[1])))
        self.resizable = bool(resizable)
        self.fps = max(1, int(fps))
        self.screen: Any | None = None
        self.closed = False
        self.theme = get_theme(theme)
        self.atlas = SpriteAtlas(self.theme, max_entries=atlas_max_entries)
        self.renderer = PygameSceneRenderer(theme=self.theme, atlas=self.atlas, headless=False)
        self.clock = pygame.time.Clock()
        self.dragging = False
        self.drag_origin: tuple[int, int] | None = None
        self.last_scene: Any | None = None
        self.last_render_result: Any | None = None
        self.selected_ow_id: int | None = None

    def _ensure_screen(self, frame: Any) -> Any:
        if self.screen is None:
            flags = self.pygame.RESIZABLE if self.resizable else 0
            self.screen = self.pygame.display.set_mode(self.window_size, flags)
            self.pygame.display.set_caption(self.title)
        if frame.scene is not None:
            width, height = self.screen.get_size()
            sidebar = max(0, width - frame.scene.camera.viewport[2])
            frame.scene.camera.viewport = (
                0,
                0,
                max(320, width - sidebar),
                height,
            )
        return self.screen

    def _handle_events(self, frame: Any) -> None:
        if frame.scene is None:
            return
        camera = frame.scene.camera
        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                self.close()
                return
            if event.type == self.pygame.VIDEORESIZE:
                self.window_size = (max(640, event.w), max(480, event.h))
                flags = self.pygame.RESIZABLE if self.resizable else 0
                self.screen = self.pygame.display.set_mode(self.window_size, flags)
            elif event.type == self.pygame.MOUSEWHEEL:
                zoom_at(camera, 1.14 if event.y > 0 else 1.0 / 1.14, self.pygame.mouse.get_pos())
            elif event.type == self.pygame.MOUSEBUTTONDOWN and event.button in (1, 2):
                self.dragging = True
                self.drag_origin = tuple(event.pos)
            elif event.type == self.pygame.MOUSEBUTTONDOWN and event.button == 3:
                if frame.scene is not None and frame.scene.sprites:
                    px, py = event.pos
                    nearest = min(
                        frame.scene.sprites,
                        key=lambda item: (
                            (item.screen_position[0] - px) ** 2
                            + (item.screen_position[1] - py) ** 2
                        ),
                    )
                    distance_sq = (nearest.screen_position[0] - px) ** 2 + (
                        nearest.screen_position[1] - py
                    ) ** 2
                    if distance_sq <= max(144.0, nearest.cell_pixels**2):
                        self.selected_ow_id = nearest.ow_id
                        camera.follow_ow_id = nearest.ow_id
                        camera.mode = "follow"
            elif event.type == self.pygame.MOUSEBUTTONUP and event.button in (1, 2):
                self.dragging = False
                self.drag_origin = None
            elif event.type == self.pygame.MOUSEMOTION and self.dragging and self.drag_origin:
                dx = event.pos[0] - self.drag_origin[0]
                dy = event.pos[1] - self.drag_origin[1]
                pan(camera, dx, dy)
                self.drag_origin = tuple(event.pos)
            elif event.type == self.pygame.KEYDOWN:
                if event.key == self.pygame.K_ESCAPE:
                    self.close()
                elif event.key == self.pygame.K_HOME:
                    from owl.viz.camera import fit_world

                    fit_world(camera)

    def _draw_legacy(self, frame: Any) -> None:
        assert self.screen is not None
        rgb = np.asarray(frame.rgba[..., :3], dtype=np.uint8)
        surface = self.pygame.surfarray.make_surface(np.swapaxes(rgb, 0, 1))
        surface = self.pygame.transform.scale(surface, self.screen.get_size())
        self.screen.blit(surface, (0, 0))

    def submit(self, frame: Any) -> None:
        if self.closed:
            return
        self._ensure_screen(frame)
        self._handle_events(frame)
        if self.closed:
            return
        assert self.screen is not None
        if frame.scene is not None:
            self.last_scene = frame.scene
            self.last_render_result = self.renderer.render(frame.scene, self.screen)
        elif frame.rgba is not None:
            self._draw_legacy(frame)
        else:
            return
        if self.last_render_result is not None and self.last_render_result.dirty_rects:
            self.pygame.display.update(self.last_render_result.dirty_rects)
        else:
            self.pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self) -> None:
        if not self.closed:
            self.renderer.close()
            self.pygame.quit()
            self.closed = True
