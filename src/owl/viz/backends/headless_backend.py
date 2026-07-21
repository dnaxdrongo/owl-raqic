from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from owl.viz.themes import get_theme


class HeadlessVisualBackend:
    """Offscreen export using the shared Pygame scene renderer when available.

    A Pillow scene fallback exists for environments without Pygame so review
    artifacts can still be produced. Every fallback frame is explicitly marked;
    production RunPod exports with Pygame remain the authoritative visual path.
    """

    def __init__(
        self,
        output_dir: str | Path = "results/visual_frames",
        scale: int = 1,
        resolution: tuple[int, int] = (1920, 1080),
        theme: str = "owl_dark_neon",
        atlas_max_entries: int = 8192,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frames_written = 0
        self.scale = max(1, int(scale))
        self.resolution = (max(320, int(resolution[0])), max(240, int(resolution[1])))
        self.theme = get_theme(theme)
        self.last_render_result: Any | None = None
        self.fallback_used = False
        self.atlas: Any | None = None
        self.pygame: Any | None = None
        self.target: Any | None = None
        self.renderer: Any | None = None
        self.pillow_renderer: Any | None = None

        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        try:
            import pygame

            from owl.viz.backends.pygame_scene_renderer import PygameSceneRenderer
            from owl.viz.sprite_atlas import SpriteAtlas

            self.pygame = pygame
            if not pygame.get_init():
                pygame.init()
            self.target = pygame.Surface(self.resolution, pygame.SRCALPHA, 32)
            self.atlas = SpriteAtlas(self.theme, max_entries=atlas_max_entries)
            self.renderer = PygameSceneRenderer(
                theme=self.theme,
                atlas=self.atlas,
                headless=True,
            )
        except Exception:
            from owl.viz.backends.pillow_scene_renderer import PillowSceneRenderer

            self.fallback_used = True
            self.pillow_renderer = PillowSceneRenderer(
                theme=self.theme,
                resolution=self.resolution,
            )

    def _draw_legacy_pygame(self, frame: Any) -> None:
        assert self.pygame is not None and self.target is not None
        rgba = np.asarray(frame.rgba, dtype=np.uint8)
        rgb = rgba[..., :3]
        surface = self.pygame.surfarray.make_surface(np.swapaxes(rgb, 0, 1))
        surface = self.pygame.transform.scale(surface, self.resolution)
        self.target.blit(surface, (0, 0))

    def _draw_legacy_pillow(self, frame: Any, path: Path) -> None:
        from PIL import Image

        rgba = np.asarray(frame.rgba, dtype=np.uint8)
        image = Image.fromarray(rgba, mode="RGBA")
        image = image.resize(self.resolution, Image.Resampling.NEAREST)
        image.save(path)

    def submit(self, frame: Any) -> None:
        tick = int(frame.scientific_tick or frame.metadata.get("tick", 0))
        subframe = int(frame.subframe_index)
        path = self.output_dir / f"frame_{tick:08d}_{subframe:03d}.png"

        if self.renderer is not None and frame.scene is not None:
            assert self.target is not None and self.pygame is not None
            self.last_render_result = self.renderer.render(frame.scene, self.target)
            self.pygame.image.save(self.target, str(path))
        elif self.pillow_renderer is not None and frame.scene is not None:
            image, self.last_render_result = self.pillow_renderer.render(frame.scene)
            image.save(path)
        elif frame.rgba is not None and self.pygame is not None:
            self.fallback_used = True
            self._draw_legacy_pygame(frame)
            assert self.target is not None
            self.pygame.image.save(self.target, str(path))
        elif frame.rgba is not None:
            self.fallback_used = True
            self._draw_legacy_pillow(frame, path)
        else:
            return

        self.frames_written += 1
        frame.metadata["headless_fallback"] = self.fallback_used
        frame.metadata["output_path"] = str(path)
        frame.metadata["renderer"] = "pillow_fallback" if self.fallback_used else "pygame_scene"

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
        if self.atlas is not None:
            self.atlas.clear()
