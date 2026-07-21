from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, replace
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from owl.viz.camera import minimap_viewport_rect, world_to_screen
from owl.viz.environment_sprites import EnvironmentKind
from owl.viz.hud import build_hud_state
from owl.viz.scene import VisualScene, VisualSpriteInstance
from owl.viz.themes import Theme


@dataclass(frozen=True)
class PillowRenderResult:
    render_ms: float
    body_count: int
    effect_count: int
    environment_count: int


def _font(size: int, *, bold: bool = False) -> Any:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _rgba(value: tuple[int, ...]) -> tuple[int, int, int, int]:
    if len(value) == 4:
        return tuple(int(channel) for channel in value)  # type: ignore[return-value]
    return int(value[0]), int(value[1]), int(value[2]), 255


def _organic_points(instance: VisualSpriteInstance, radius: float) -> list[tuple[float, float]]:
    cx, cy = instance.screen_position
    archetype = instance.descriptor.archetype
    seed = abs(instance.ow_id * 17 + instance.descriptor.lineage_marker * 31)
    count = 18 if archetype != "spined" else 24
    points: list[tuple[float, float]] = []
    rotation = math.radians(instance.pose.rotation_degrees)
    for index in range(count):
        angle = 2.0 * math.pi * index / count + rotation
        if archetype == "lobed":
            wave = 0.84 + 0.18 * math.sin(3.0 * angle + seed * 0.13)
        elif archetype == "spined":
            wave = 1.22 if index % 2 == 0 else 0.72
        elif archetype == "ciliated":
            wave = 0.88 + 0.12 * math.cos(2.0 * angle)
        elif archetype == "radiant":
            wave = 0.96 + 0.08 * math.sin(6.0 * angle)
        else:
            wave = 0.90 + 0.10 * math.sin(5.0 * angle + seed * 0.23)
        points.append((cx + math.cos(angle) * radius * wave, cy + math.sin(angle) * radius * wave))
    return points


class PillowSceneRenderer:
    """Visual-only fallback used when Pygame cannot initialize.

    The production live/headless path remains the shared Pygame renderer.  This
    fallback consumes the same immutable ``VisualScene`` and is marked in output
    metadata so it can never be confused with the Pygame-certified visual path.
    """

    def __init__(self, *, theme: Theme, resolution: tuple[int, int]) -> None:
        self.theme = theme
        self.resolution = resolution
        self.font = _font(15)
        self.small_font = _font(12)
        self.title_font = _font(20, bold=True)

    def _draw_environment(self, draw: ImageDraw.ImageDraw, scene: VisualScene) -> None:
        for item in scene.environment:
            x, y = item.screen_position
            size = max(
                2.0, item.cell_pixels * (0.43 if item.kind == EnvironmentKind.OBSTACLE else 0.30)
            )
            color = _rgba(item.color)
            if item.kind == EnvironmentKind.FOOD:
                for angle in (0.0, 2.1, 4.2):
                    dx = math.cos(angle + item.variant * 0.1) * size * 0.55
                    dy = math.sin(angle + item.variant * 0.1) * size * 0.55
                    draw.ellipse(
                        (
                            x + dx - size * 0.45,
                            y + dy - size * 0.28,
                            x + dx + size * 0.45,
                            y + dy + size * 0.28,
                        ),
                        fill=color,
                    )
                draw.ellipse(
                    (x - size * 0.18, y - size * 0.18, x + size * 0.18, y + size * 0.18),
                    fill=(250, 225, 105, 230),
                )
            elif item.kind == EnvironmentKind.TOXIN:
                for index in range(3):
                    offset = ((item.variant + index) % 3 - 1) * size * 0.35
                    radius = size * (0.75 - index * 0.14)
                    draw.ellipse(
                        (
                            x + offset - radius,
                            y - offset - radius,
                            x + offset + radius,
                            y - offset + radius,
                        ),
                        outline=color,
                        width=max(1, int(size * 0.18)),
                    )
            elif item.kind == EnvironmentKind.WASTE:
                rng = random.Random(item.variant)
                for _ in range(4):
                    px = x + (rng.random() - 0.5) * size * 1.5
                    py = y + (rng.random() - 0.5) * size * 1.5
                    draw.ellipse(
                        (px - size * 0.18, py - size * 0.18, px + size * 0.18, py + size * 0.18),
                        fill=color,
                    )
            elif item.kind == EnvironmentKind.OBSTACLE:
                points = [
                    (x, y - size),
                    (x + size, y - size * 0.25),
                    (x + size * 0.65, y + size),
                    (x - size * 0.7, y + size * 0.8),
                    (x - size, y),
                ]
                draw.polygon(points, fill=color, outline=(205, 220, 240, 255))
                draw.line(
                    (points[0], points[2]),
                    fill=(130, 150, 180, 255),
                    width=max(1, int(size * 0.18)),
                )
            elif item.kind == EnvironmentKind.SIGNAL:
                for factor in (0.38, 0.68, 1.0):
                    radius = size * factor
                    draw.ellipse(
                        (x - radius, y - radius, x + radius, y + radius),
                        outline=color,
                        width=max(1, int(size * 0.16)),
                    )
            elif item.kind == EnvironmentKind.DEAD_SHELL:
                draw.ellipse(
                    (x - size, y - size, x + size, y + size),
                    outline=color,
                    width=max(1, int(size * 0.18)),
                )
                draw.line(
                    (x - size * 0.5, y - size * 0.5, x + size * 0.5, y + size * 0.5),
                    fill=color,
                    width=max(1, int(size * 0.16)),
                )
                draw.line(
                    (x + size * 0.5, y - size * 0.5, x - size * 0.5, y + size * 0.5),
                    fill=color,
                    width=max(1, int(size * 0.16)),
                )
            else:
                draw.rectangle(
                    (x - size, y - size, x + size, y + size),
                    outline=color,
                    width=max(1, int(size * 0.15)),
                )

    def _draw_body(self, draw: ImageDraw.ImageDraw, instance: VisualSpriteInstance) -> None:
        x, y = instance.screen_position
        radius = max(2.0, instance.cell_pixels * 0.40 * instance.pose.scale)
        descriptor = instance.descriptor
        status = instance.status
        outline = _rgba(descriptor.trait_color.outline_rgba)
        rgb = descriptor.trait_color.rendered_rgb
        fill = (rgb[0], rgb[1], rgb[2], max(60, min(255, int(instance.pose.alpha * 245))))

        cilia_count = descriptor.cilia_level * 4
        for index in range(cilia_count):
            angle = 2.0 * math.pi * index / max(1, cilia_count) + math.radians(
                instance.pose.rotation_degrees
            )
            draw.line(
                (
                    x + math.cos(angle) * radius * 0.85,
                    y + math.sin(angle) * radius * 0.85,
                    x + math.cos(angle) * radius * 1.25,
                    y + math.sin(angle) * radius * 1.25,
                ),
                fill=outline,
                width=max(1, int(radius * 0.12)),
            )

        points = _organic_points(instance, radius)
        draw.polygon(points, fill=fill, outline=outline)
        draw.line(
            points + [points[0]], fill=outline, width=max(1, int(radius * 0.15)), joint="curve"
        )

        if descriptor.membrane_pattern == "dotted":
            for index in range(6):
                angle = 2.0 * math.pi * index / 6.0
                px = x + math.cos(angle) * radius * 0.66
                py = y + math.sin(angle) * radius * 0.66
                rr = max(1.0, radius * 0.10)
                draw.ellipse((px - rr, py - rr, px + rr, py + rr), fill=outline)
        elif descriptor.membrane_pattern == "banded":
            draw.arc(
                (x - radius * 0.78, y - radius * 0.78, x + radius * 0.78, y + radius * 0.78),
                210,
                330,
                fill=outline,
                width=max(1, int(radius * 0.15)),
            )

        core = radius * (0.18 + status.resource_fraction * 0.18)
        core_color = (
            (245, 235, 175, 245) if descriptor.nucleus_style == "double" else (220, 235, 255, 240)
        )
        draw.ellipse(
            (x - core, y - core, x + core, y + core),
            fill=core_color,
            outline=(20, 28, 44, 255),
            width=max(1, int(radius * 0.10)),
        )
        if descriptor.nucleus_style == "double":
            rr = core * 0.38
            draw.ellipse((x - rr, y - rr, x + rr, y + rr), fill=(85, 110, 155, 240))

        marker_angle = (descriptor.lineage_marker & 0xF) / 16.0 * 2.0 * math.pi
        draw.line(
            (
                x + math.cos(marker_angle) * radius * 0.45,
                y + math.sin(marker_angle) * radius * 0.45,
                x + math.cos(marker_angle) * radius * 0.80,
                y + math.sin(marker_angle) * radius * 0.80,
            ),
            fill=outline,
            width=max(1, int(radius * 0.12)),
        )

        for index in range(descriptor.spike_level * 2):
            angle = 2.0 * math.pi * index / max(1, descriptor.spike_level * 2)
            draw.line(
                (
                    x + math.cos(angle) * radius * 0.82,
                    y + math.sin(angle) * radius * 0.82,
                    x + math.cos(angle) * radius * 1.25,
                    y + math.sin(angle) * radius * 1.25,
                ),
                fill=outline,
                width=max(1, int(radius * 0.12)),
            )

        # Dynamic status uses rings/marks, never identity fill.
        start = -90
        end = start + int(360 * status.health_fraction)
        draw.arc(
            (x - radius * 1.13, y - radius * 1.13, x + radius * 1.13, y + radius * 1.13),
            start,
            end,
            fill=_rgba(
                self.theme.health_good if status.health_fraction >= 0.45 else self.theme.health_bad
            ),
            width=max(1, int(radius * 0.13)),
        )
        if status.toxin_fraction > 0.35:
            draw.ellipse(
                (x - radius * 1.28, y - radius * 1.28, x + radius * 1.28, y + radius * 1.28),
                outline=_rgba(self.theme.toxin),
                width=max(1, int(radius * 0.12)),
            )
        if status.selected:
            draw.ellipse(
                (x - radius * 1.45, y - radius * 1.45, x + radius * 1.45, y + radius * 1.45),
                outline=_rgba(self.theme.selected),
                width=max(2, int(radius * 0.14)),
            )
        if status.health_fraction < 0.25:
            draw.line(
                (x - radius * 0.5, y - radius * 0.5, x + radius * 0.5, y + radius * 0.5),
                fill=_rgba(self.theme.health_bad),
                width=max(1, int(radius * 0.15)),
            )

    def _draw_effects(self, draw: ImageDraw.ImageDraw, scene: VisualScene) -> None:
        for effect in scene.effects:
            sx, sy = world_to_screen(scene.camera, effect.source[0], effect.source[1])
            target = (
                world_to_screen(scene.camera, effect.target[0], effect.target[1])
                if effect.target is not None
                else None
            )
            color = _rgba(effect.color)
            radius = max(4.0, scene.camera.cell_pixels * 0.55)
            t = effect.progress
            if effect.kind in {
                "breath",
                "coherence_halo",
                "birth_glow",
                "inhibit_field",
                "integration_orbit",
            }:
                rr = radius * (0.45 + 0.85 * t)
                draw.ellipse(
                    (sx - rr, sy - rr, sx + rr, sy + rr),
                    outline=color,
                    width=max(1, int(radius * 0.16)),
                )
                if effect.kind == "integration_orbit":
                    for index in range(4):
                        angle = 2.0 * math.pi * (index / 4.0 + t)
                        px, py = sx + math.cos(angle) * rr, sy + math.sin(angle) * rr
                        draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=color)
            elif effect.kind == "scan_arc":
                draw.arc(
                    (sx - radius, sy - radius, sx + radius, sy + radius),
                    int(360 * t),
                    int(360 * t + 120),
                    fill=color,
                    width=max(1, int(radius * 0.16)),
                )
            elif effect.kind == "signal_wave":
                for offset in (0.0, 0.28, 0.56):
                    rr = radius * (0.3 + ((t + offset) % 1.0) * 1.45)
                    draw.ellipse(
                        (sx - rr, sy - rr, sx + rr, sy + rr),
                        outline=color,
                        width=max(1, int(radius * 0.14)),
                    )
            elif target is not None and effect.kind in {
                "inhibit_beam",
                "birth_bridge",
                "merge_bridge",
                "ingest_lunge",
                "resource_transfer",
                "move_trail",
                "flee_trail",
                "pursuit_trail",
            }:
                draw.line(
                    (sx, sy, target[0], target[1]), fill=color, width=max(1, int(radius * 0.18))
                )
            elif effect.kind in {
                "repair_stitch",
                "repair_sparks",
                "nutrient_spiral",
                "expel_burst",
                "split_membrane",
            }:
                rng = random.Random(effect.seed)
                for _ in range(7):
                    angle = rng.random() * 2.0 * math.pi
                    distance = (
                        radius
                        * (0.25 + rng.random() * 1.1)
                        * (t if effect.kind in {"expel_burst", "split_membrane"} else (1.0 - t))
                    )
                    px, py = sx + math.cos(angle) * distance, sy + math.sin(angle) * distance
                    draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=color)
            elif effect.kind == "movement_recoil":
                draw.ellipse(
                    (
                        sx - radius * 0.55,
                        sy - radius * 0.55,
                        sx + radius * 0.55,
                        sy + radius * 0.55,
                    ),
                    outline=color,
                    width=max(1, int(radius * 0.14)),
                )
            elif effect.kind == "target_reticle" and target is not None:
                rr = radius * 0.7
                draw.ellipse(
                    (target[0] - rr, target[1] - rr, target[0] + rr, target[1] + rr),
                    outline=color,
                    width=1,
                )
                draw.line((target[0] - rr, target[1], target[0] + rr, target[1]), fill=color)
                draw.line((target[0], target[1] - rr, target[0], target[1] + rr), fill=color)

    def _draw_hud(
        self, draw: ImageDraw.ImageDraw, scene: VisualScene, size: tuple[int, int]
    ) -> None:
        hud = build_hud_state(scene)
        vx, _vy, vw, _vh = scene.camera.viewport
        sidebar_x = vx + vw
        if sidebar_x >= size[0]:
            return
        draw.rectangle((sidebar_x, 0, size[0], size[1]), fill=_rgba(self.theme.panel))
        draw.line((sidebar_x, 0, sidebar_x, size[1]), fill=_rgba(self.theme.panel_border), width=2)
        y = 18
        draw.text(
            (sidebar_x + 18, y), "OWL + RAQIC", fill=_rgba(self.theme.text), font=self.title_font
        )
        y += 36
        for line in hud.lines:
            draw.text((sidebar_x + 18, y), line or " ", fill=_rgba(self.theme.text), font=self.font)
            y += 21
        if hud.selected is not None:
            preview = replace(
                hud.selected,
                screen_position=(sidebar_x + 255.0, 105.0),
                cell_pixels=96.0,
            )
            self._draw_body(draw, preview)
            y += 8
            labels = ("Pred", "Growth", "Coop", "Resil", "Cog", "Signal")
            for label, value in zip(
                labels,
                hud.selected.descriptor.trait_color.nibbles,
                strict=True,
            ):
                draw.rectangle(
                    (sidebar_x + 18, y, sidebar_x + 158, y + 12),
                    fill=(35, 48, 70, 255),
                )
                width = int(140 * value / 15.0)
                draw.rectangle(
                    (sidebar_x + 18, y, sidebar_x + 18 + width, y + 12),
                    fill=(*hud.selected.descriptor.trait_color.rendered_rgb, 255),
                )
                draw.text(
                    (sidebar_x + 165, y - 2),
                    f"{label} {value:X}",
                    fill=_rgba(self.theme.text),
                    font=self.small_font,
                )
                y += 18

        minimap = (
            sidebar_x + 180,
            max(10, size[1] - 130),
            min(150, max(80, size[0] - sidebar_x - 195)),
            110,
        )
        draw.rectangle(
            (
                minimap[0],
                minimap[1],
                minimap[0] + minimap[2],
                minimap[1] + minimap[3],
            ),
            fill=(8, 12, 20, 255),
            outline=_rgba(self.theme.panel_border),
        )
        height, width = scene.camera.world_shape
        for instance in scene.sprites:
            wy, wx = instance.world_position
            mx = minimap[0] + int((wx + 0.5) / max(width, 1) * minimap[2])
            my = minimap[1] + int((wy + 0.5) / max(height, 1) * minimap[3])
            color = (*instance.descriptor.trait_color.rendered_rgb, 255)
            draw.ellipse((mx - 1, my - 1, mx + 1, my + 1), fill=color)
        rect = minimap_viewport_rect(scene.camera, minimap)
        draw.rectangle(
            (rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]),
            outline=_rgba(self.theme.selected),
            width=1,
        )

    def render(self, scene: VisualScene) -> tuple[Image.Image, PillowRenderResult]:
        started = time.perf_counter()
        image = Image.new("RGBA", self.resolution, _rgba(scene.background_rgba))
        draw = ImageDraw.Draw(image, "RGBA")
        vx, vy, vw, vh = scene.camera.viewport
        draw.rectangle(
            (vx, vy, vx + vw, vy + vh),
            fill=_rgba(self.theme.background),
            outline=_rgba(self.theme.panel_border),
        )

        # Optional scientific underlay is intentionally omitted in fallback when
        # its projection cannot be guaranteed; sprite/environment semantics remain.
        self._draw_environment(draw, scene)
        for instance in scene.sprites:
            self._draw_body(draw, instance)
        self._draw_effects(draw, scene)
        if bool(scene.metadata.get("show_patch_overlay", True)) and scene.camera.cell_pixels >= 5:
            patch = max(1, int(scene.metadata.get("patch_size", 5)))
            height, width = scene.camera.world_shape
            for x in range(0, width + 1, patch):
                px, _ = world_to_screen(scene.camera, 0.0, float(x) - 0.5)
                draw.line((px, vy, px, vy + vh), fill=_rgba(self.theme.patch), width=1)
            for y in range(0, height + 1, patch):
                _, py = world_to_screen(scene.camera, float(y) - 0.5, 0.0)
                draw.line((vx, py, vx + vw, py), fill=_rgba(self.theme.patch), width=1)
        self._draw_hud(draw, scene, self.resolution)
        result = PillowRenderResult(
            render_ms=(time.perf_counter() - started) * 1000.0,
            body_count=len(scene.sprites),
            effect_count=len(scene.effects),
            environment_count=len(scene.environment),
        )
        return image, result
