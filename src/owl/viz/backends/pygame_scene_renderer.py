from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Any

from owl.viz.camera import minimap_viewport_rect, world_to_screen
from owl.viz.environment_sprites import EnvironmentKind
from owl.viz.hud import TextSurfaceCache, build_hud_state
from owl.viz.scene import VisualScene, VisualSpriteInstance
from owl.viz.sprite_atlas import SpriteAtlas, orientation_bucket, size_bucket
from owl.viz.themes import Theme


@dataclass(frozen=True)
class RenderResult:
    dirty_rects: tuple[Any, ...]
    render_ms: float
    body_blits: int
    effect_count: int
    culled_count: int


class PygameSceneRenderer:
    def __init__(self, *, theme: Theme, atlas: SpriteAtlas, headless: bool) -> None:
        import pygame
        import pygame.freetype

        self.pygame = pygame
        self.theme = theme
        self.atlas = atlas
        self.headless = bool(headless)
        if not pygame.get_init():
            pygame.init()
        pygame.freetype.init()
        self.font = pygame.freetype.SysFont("DejaVu Sans", 15)
        self.small_font = pygame.freetype.SysFont("DejaVu Sans", 12)
        self.title_font = pygame.freetype.SysFont("DejaVu Sans", 18, bold=True)
        self.text_cache = TextSurfaceCache(self.font)

    def render(self, scene: VisualScene, target: Any) -> RenderResult:
        started = time.perf_counter()
        self.render_background(scene, target)
        self.render_environment(scene, target)
        body_blits = self.render_bodies(scene, target)
        effect_count = self.render_effects(scene, target)
        self.render_overlays(scene, target)
        self.render_hud(scene, target)
        return RenderResult(
            dirty_rects=(target.get_rect(),),
            render_ms=(time.perf_counter() - started) * 1000.0,
            body_blits=body_blits,
            effect_count=effect_count,
            culled_count=0,
        )

    def render_background(self, scene: VisualScene, target: Any) -> None:
        pygame = self.pygame
        target.fill(scene.background_rgba)
        vx, vy, vw, vh = scene.camera.viewport
        pygame.draw.rect(target, self.theme.background, (vx, vy, vw, vh))
        if scene.overlays and scene.camera.mode == "fit":
            overlay = scene.overlays[0]
            try:
                import numpy as np

                rgba = np.asarray(overlay, dtype=np.uint8)
                surface = pygame.image.frombuffer(
                    rgba.tobytes(),
                    (rgba.shape[1], rgba.shape[0]),
                    "RGBA",
                )
                surface = pygame.transform.smoothscale(surface, (vw, vh))
                target.blit(surface, (vx, vy))
            except Exception:
                pass
        if scene.camera.cell_pixels >= 6.0:
            y0, x0 = scene.camera.center
            del y0, x0
            height, width = scene.camera.world_shape
            step = scene.camera.cell_pixels
            left_world = scene.camera.center[1] - vw / (2.0 * step)
            top_world = scene.camera.center[0] - vh / (2.0 * step)
            first_x = math.floor(left_world)
            first_y = math.floor(top_world)
            for x in range(first_x, min(width + 1, first_x + int(vw / step) + 3)):
                px, _py = world_to_screen(scene.camera, 0.0, float(x) - 0.5)
                pygame.draw.line(target, self.theme.grid, (px, vy), (px, vy + vh), 1)
            for y in range(first_y, min(height + 1, first_y + int(vh / step) + 3)):
                _px, py = world_to_screen(scene.camera, float(y) - 0.5, 0.0)
                pygame.draw.line(target, self.theme.grid, (vx, py), (vx + vw, py), 1)
        pygame.draw.rect(target, self.theme.panel_border, (vx, vy, vw, vh), 1)

    def render_environment(self, scene: VisualScene, target: Any) -> None:
        blits: list[tuple[Any, Any]] = []
        for instance in scene.environment:
            size = max(
                4,
                int(
                    instance.cell_pixels
                    * (0.82 if instance.kind == EnvironmentKind.OBSTACLE else 0.62)
                ),
            )
            surface = self.atlas.get_environment(instance, size=size)
            rect = surface.get_rect(center=instance.screen_position)
            blits.append((surface, rect))
        if blits:
            target.blits(blits, doreturn=False)

    def _draw_status(self, target: Any, instance: VisualSpriteInstance) -> None:
        pygame = self.pygame
        x, y = instance.screen_position
        radius = max(3, int(instance.cell_pixels * 0.43 * instance.pose.scale))
        status = instance.status
        descriptor = instance.descriptor
        outline = descriptor.trait_color.outline_rgba

        if status.health_fraction < 0.999:
            rect = pygame.Rect(x - radius - 2, y - radius - 2, 2 * radius + 4, 2 * radius + 4)
            pygame.draw.arc(
                target,
                self.theme.health_good if status.health_fraction > 0.4 else self.theme.health_bad,
                rect,
                -math.pi / 2.0,
                -math.pi / 2.0 + 2.0 * math.pi * status.health_fraction,
                max(1, radius // 5),
            )
        if instance.cell_pixels >= 10:
            core_radius = max(1, int(radius * 0.27 * status.resource_fraction))
            pygame.draw.circle(target, self.theme.resource, (int(x), int(y)), core_radius)
            angle = status.phase
            end = (x + math.cos(angle) * radius * 0.86, y + math.sin(angle) * radius * 0.86)
            pygame.draw.line(target, outline, (x, y), end, max(1, radius // 6))
        if status.toxin_fraction > 0.25:
            pygame.draw.circle(target, self.theme.toxin, (int(x), int(y)), radius + 2, 1)
        if status.selected:
            pygame.draw.circle(target, self.theme.selected, (int(x), int(y)), radius + 5, 2)
        if status.health_fraction < 0.25:
            pygame.draw.line(
                target,
                self.theme.health_bad,
                (x - radius * 0.5, y - radius * 0.5),
                (x + radius * 0.45, y + radius * 0.55),
                max(1, radius // 5),
            )

    def render_bodies(self, scene: VisualScene, target: Any) -> int:
        blits: list[tuple[Any, Any]] = []
        for instance in scene.sprites:
            size = size_bucket(max(5, int(instance.cell_pixels * 0.92 * instance.pose.scale)))
            body = self.atlas.get_body(
                instance.descriptor,
                size=size,
                orientation_bucket=orientation_bucket(instance.pose.rotation_degrees),
            )
            if instance.pose.alpha < 0.999:
                body = body.copy()
                body.set_alpha(max(0, min(255, int(instance.pose.alpha * 255))))
            rect = body.get_rect(center=instance.screen_position)
            blits.append((body, rect))
        if blits:
            target.blits(blits, doreturn=False)
        for instance in scene.sprites:
            self._draw_status(target, instance)
        return len(blits)

    def _screen_pair(
        self, scene: VisualScene, effect: Any
    ) -> tuple[tuple[float, float], tuple[float, float] | None]:
        source = world_to_screen(scene.camera, effect.source[0], effect.source[1])
        target = (
            world_to_screen(scene.camera, effect.target[0], effect.target[1])
            if effect.target is not None
            else None
        )
        return source, target

    def _draw_particles(
        self,
        target_surface: Any,
        source: tuple[float, float],
        target: tuple[float, float] | None,
        effect: Any,
        count: int,
        inward: bool,
    ) -> None:
        pygame = self.pygame
        rng = random.Random(effect.seed)
        t = effect.progress
        for index in range(count):
            angle = rng.random() * 2.0 * math.pi
            distance = 10.0 + rng.random() * 24.0
            if target is not None:
                start = source if inward else target
                end = target if inward else source
                px = start[0] + (end[0] - start[0]) * t
                py = start[1] + (end[1] - start[1]) * t
            else:
                factor = (1.0 - t) if inward else t
                px = source[0] + math.cos(angle) * distance * factor
                py = source[1] + math.sin(angle) * distance * factor
            radius = 1 + (index + effect.seed) % 3
            pygame.draw.circle(target_surface, effect.color, (int(px), int(py)), radius)

    def render_effects(self, scene: VisualScene, target_surface: Any) -> int:
        pygame = self.pygame
        for effect in scene.effects:
            source, target = self._screen_pair(scene, effect)
            t = effect.progress
            alpha_color = effect.color
            base_radius = max(4.0, scene.camera.cell_pixels * 0.55)
            kind = effect.kind
            if kind in {"breath", "coherence_halo", "birth_glow"}:
                radius = int(base_radius * (0.85 + 0.35 * math.sin(math.pi * t)))
                pygame.draw.circle(target_surface, alpha_color, source, radius, 1)
            elif kind == "scan_arc":
                rect = pygame.Rect(
                    source[0] - base_radius,
                    source[1] - base_radius,
                    2 * base_radius,
                    2 * base_radius,
                )
                start = 2.0 * math.pi * t
                pygame.draw.arc(target_surface, alpha_color, rect, start, start + math.pi * 0.65, 2)
            elif kind == "signal_wave":
                for offset in (0.0, 0.28, 0.56):
                    radius = int(base_radius * (0.35 + ((t + offset) % 1.0) * 1.45))
                    pygame.draw.circle(target_surface, alpha_color, source, radius, 2)
            elif (
                kind
                in {
                    "inhibit_beam",
                    "birth_bridge",
                    "merge_bridge",
                    "ingest_lunge",
                    "resource_transfer",
                }
                and target is not None
            ):
                width = max(1, int(1 + base_radius * 0.12))
                pygame.draw.line(target_surface, alpha_color, source, target, width)
                if kind == "resource_transfer":
                    self._draw_particles(target_surface, source, target, effect, 6, inward=True)
            elif kind in {"inhibit_field", "integration_orbit"}:
                radius = int(base_radius * (0.55 + 0.7 * t))
                pygame.draw.circle(target_surface, alpha_color, source, radius, 2)
                if kind == "integration_orbit":
                    for index in range(4):
                        angle = 2.0 * math.pi * (index / 4.0 + t)
                        node = (
                            source[0] + math.cos(angle) * radius,
                            source[1] + math.sin(angle) * radius,
                        )
                        pygame.draw.circle(target_surface, alpha_color, node, 2)
            elif kind in {"repair_stitch", "repair_sparks"}:
                if kind == "repair_stitch":
                    for index in range(3):
                        dx = (index - 1) * base_radius * 0.28
                        pygame.draw.line(
                            target_surface,
                            alpha_color,
                            (source[0] + dx - 3, source[1] - 4),
                            (source[0] + dx + 3, source[1] + 4),
                            2,
                        )
                else:
                    self._draw_particles(target_surface, source, None, effect, 7, inward=False)
            elif kind in {"nutrient_spiral"}:
                self._draw_particles(target_surface, source, target, effect, 8, inward=True)
            elif kind in {"expel_burst", "split_membrane"}:
                self._draw_particles(target_surface, source, target, effect, 9, inward=False)
                if target is not None:
                    pygame.draw.line(target_surface, alpha_color, source, target, 2)
            elif kind in {"move_trail", "flee_trail", "pursuit_trail"} and target is not None:
                pygame.draw.line(
                    target_surface, alpha_color, source, target, max(1, int(base_radius * 0.18))
                )
            elif kind == "movement_recoil":
                pygame.draw.circle(target_surface, alpha_color, source, int(base_radius * 0.55), 2)
            elif kind == "target_reticle" and target is not None:
                radius = int(base_radius * (0.55 + 0.15 * math.sin(2.0 * math.pi * t)))
                pygame.draw.circle(target_surface, alpha_color, target, radius, 1)
                pygame.draw.line(
                    target_surface,
                    alpha_color,
                    (target[0] - radius, target[1]),
                    (target[0] + radius, target[1]),
                    1,
                )
                pygame.draw.line(
                    target_surface,
                    alpha_color,
                    (target[0], target[1] - radius),
                    (target[0], target[1] + radius),
                    1,
                )
            elif kind == "ingest_bite":
                pygame.draw.arc(
                    target_surface,
                    alpha_color,
                    pygame.Rect(
                        source[0] - base_radius,
                        source[1] - base_radius,
                        2 * base_radius,
                        2 * base_radius,
                    ),
                    0.15 * math.pi,
                    1.85 * math.pi,
                    2,
                )
            elif kind == "birth_pulse":
                pygame.draw.circle(
                    target_surface, alpha_color, source, int(base_radius * (0.5 + t)), 2
                )
        return len(scene.effects)

    def render_overlays(self, scene: VisualScene, target: Any) -> None:
        if not bool(scene.metadata.get("show_patch_overlay", True)):
            return
        pygame = self.pygame
        height, width = scene.camera.world_shape
        patch = max(1, int(scene.metadata.get("patch_size", 5)))
        if scene.camera.cell_pixels >= 5:
            for x in range(0, width + 1, patch):
                px, _ = world_to_screen(scene.camera, 0.0, float(x) - 0.5)
                pygame.draw.line(
                    target,
                    self.theme.patch,
                    (px, scene.camera.viewport[1]),
                    (px, scene.camera.viewport[1] + scene.camera.viewport[3]),
                    1,
                )
            for y in range(0, height + 1, patch):
                _, py = world_to_screen(scene.camera, float(y) - 0.5, 0.0)
                pygame.draw.line(
                    target,
                    self.theme.patch,
                    (scene.camera.viewport[0], py),
                    (scene.camera.viewport[0] + scene.camera.viewport[2], py),
                    1,
                )

    def render_hud(self, scene: VisualScene, target: Any) -> None:
        pygame = self.pygame
        hud = build_hud_state(scene)
        vx, vy, vw, vh = scene.camera.viewport
        screen_w, screen_h = target.get_size()
        sidebar_x = vx + vw
        if sidebar_x < screen_w:
            pygame.draw.rect(
                target, self.theme.panel, (sidebar_x, 0, screen_w - sidebar_x, screen_h)
            )
            pygame.draw.line(
                target, self.theme.panel_border, (sidebar_x, 0), (sidebar_x, screen_h), 2
            )
            y = 18
            title, _ = self.title_font.render("OWL + RAQIC", fgcolor=self.theme.text)
            target.blit(title, (sidebar_x + 18, y))
            y += 34
            for line in hud.lines:
                surface = self.text_cache.render(line or " ", self.theme.text)
                target.blit(surface, (sidebar_x + 18, y))
                y += 21
            if hud.selected is not None:
                preview = self.atlas.get_body(
                    hud.selected.descriptor,
                    size=96,
                    orientation_bucket=orientation_bucket(hud.selected.pose.rotation_degrees),
                )
                target.blit(
                    preview,
                    preview.get_rect(center=(sidebar_x + 255, 105)),
                )
                nibbles = hud.selected.descriptor.trait_color.nibbles
                labels = ("Pred", "Growth", "Coop", "Resil", "Cog", "Signal")
                y += 8
                for label, value in zip(labels, nibbles, strict=True):
                    pygame.draw.rect(
                        target,
                        (35, 48, 70, 255),
                        (sidebar_x + 18, y, 140, 12),
                    )
                    pygame.draw.rect(
                        target,
                        hud.selected.descriptor.trait_color.rendered_rgb,
                        (sidebar_x + 18, y, int(140 * value / 15.0), 12),
                    )
                    text = self.small_font.render(
                        f"{label} {value:X}",
                        fgcolor=self.theme.text,
                    )[0]
                    target.blit(text, (sidebar_x + 165, y - 2))
                    y += 18

        minimap = (
            max(vx + 10, sidebar_x + 180),
            max(10, screen_h - 130),
            min(150, max(80, screen_w - sidebar_x - 195)),
            110,
        )
        if sidebar_x < screen_w and minimap[2] > 0:
            pygame.draw.rect(target, (8, 12, 20, 255), minimap)
            pygame.draw.rect(target, self.theme.panel_border, minimap, 1)
            height, width = scene.camera.world_shape
            for instance in scene.sprites:
                wy, wx = instance.world_position
                mx = minimap[0] + int((wx + 0.5) / max(width, 1) * minimap[2])
                my = minimap[1] + int((wy + 0.5) / max(height, 1) * minimap[3])
                pygame.draw.circle(
                    target,
                    instance.descriptor.trait_color.rendered_rgb,
                    (mx, my),
                    1,
                )
            rect = minimap_viewport_rect(scene.camera, minimap)
            pygame.draw.rect(target, self.theme.selected, rect, 1)

    def close(self) -> None:
        self.atlas.clear()
