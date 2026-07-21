from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from owl.viz.environment_sprites import EnvironmentInstance, EnvironmentKind
from owl.viz.sprite_state import SpriteDescriptor
from owl.viz.themes import Theme


@dataclass(frozen=True)
class SpriteAtlasKey:
    category: str
    identity: str
    size: int
    orientation_bucket: int
    variant: int = 0


@dataclass
class SpriteAtlasMetrics:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    generated: int = 0
    peak_entries: int = 0


class SpriteAtlas:
    def __init__(self, theme: Theme, *, max_entries: int = 8192) -> None:
        self.theme = theme
        self.max_entries = max(128, int(max_entries))
        self._cache: OrderedDict[SpriteAtlasKey, Any] = OrderedDict()
        self.metrics = SpriteAtlasMetrics()

    def _get(self, key: SpriteAtlasKey) -> Any | None:
        value = self._cache.get(key)
        if value is None:
            self.metrics.misses += 1
            return None
        self._cache.move_to_end(key)
        self.metrics.hits += 1
        return value

    def _put(self, key: SpriteAtlasKey, value: Any) -> Any:
        self._cache[key] = value
        self._cache.move_to_end(key)
        self.metrics.generated += 1
        while len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)
            self.metrics.evictions += 1
        self.metrics.peak_entries = max(self.metrics.peak_entries, len(self._cache))
        return value

    def get_body(
        self,
        descriptor: SpriteDescriptor,
        *,
        size: int,
        orientation_bucket: int,
    ) -> Any:
        bucket_size = size_bucket(size)
        orientation = int(orientation_bucket) % 16
        identity = (
            f"{descriptor.trait_color.raw_hex}:{descriptor.archetype}:"
            f"{descriptor.membrane_pattern}:{descriptor.cilia_level}:"
            f"{descriptor.spike_level}:{descriptor.nucleus_style}:"
            f"{descriptor.developmental_stage}:{descriptor.lineage_marker % 16}"
        )
        key = SpriteAtlasKey("body", identity, bucket_size, orientation)
        cached = self._get(key)
        if cached is not None:
            return cached

        base_key = SpriteAtlasKey("body_base", identity, bucket_size, 0)
        base = self._get(base_key)
        if base is None:
            base = render_procedural_body(descriptor, bucket_size, self.theme)
            self._put(base_key, base)
        if orientation:
            import pygame

            angle = -orientation * (360.0 / 16.0)
            image = pygame.transform.rotozoom(base, angle, 1.0)
        else:
            image = base
        return self._put(key, image)

    def get_environment(self, instance: EnvironmentInstance, *, size: int) -> Any:
        bucket_size = size_bucket(size)
        key = SpriteAtlasKey(
            "environment",
            str(instance.kind),
            bucket_size,
            0,
            int(instance.variant),
        )
        cached = self._get(key)
        if cached is not None:
            return cached
        image = render_environment_sprite(instance, bucket_size, self.theme)
        return self._put(key, image)

    def clear(self) -> None:
        self._cache.clear()

    def summary(self) -> dict[str, Any]:
        return {
            "entries": len(self._cache),
            "max_entries": self.max_entries,
            **self.metrics.__dict__,
            "hit_rate": self.metrics.hits / max(1, self.metrics.hits + self.metrics.misses),
        }


def size_bucket(value: float | int) -> int:
    size = max(4, int(round(float(value))))
    for bucket in (6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96):
        if size <= bucket:
            return bucket
    return 128


def orientation_bucket(angle_degrees: float) -> int:
    return int(round((float(angle_degrees) % 360.0) / (360.0 / 16.0))) % 16


def _polygon_points(
    *,
    center: tuple[float, float],
    radius: float,
    count: int,
    archetype: str,
    seed: int,
) -> list[tuple[int, int]]:
    cx, cy = center
    points: list[tuple[int, int]] = []
    for index in range(count):
        angle = 2.0 * math.pi * index / count
        wave = 1.0
        if archetype == "lobed":
            wave = 0.84 + 0.17 * math.sin(3.0 * angle + seed * 0.13)
        elif archetype == "spined":
            wave = 1.20 if index % 2 == 0 else 0.72
        elif archetype == "ciliated":
            wave = 0.88 + 0.12 * math.cos(2.0 * angle)
        elif archetype == "radiant":
            wave = 0.96 + 0.08 * math.sin(6.0 * angle)
        else:
            wave = 0.90 + 0.10 * math.sin(5.0 * angle + seed * 0.23)
        points.append(
            (
                int(round(cx + math.cos(angle) * radius * wave)),
                int(round(cy + math.sin(angle) * radius * wave)),
            )
        )
    return points


def _convert_alpha(surface: Any) -> Any:
    try:
        import pygame

        if pygame.display.get_init() and pygame.display.get_surface() is not None:
            return surface.convert_alpha()
    except Exception:
        pass
    return surface


def render_procedural_body(
    descriptor: SpriteDescriptor,
    size: int,
    theme: Theme,
) -> Any:
    import pygame

    scale = 3
    side = max(6, int(size)) * scale
    surface = pygame.Surface((side, side), pygame.SRCALPHA, 32)
    cx = cy = side / 2.0
    radius = side * 0.34
    seed = abs(descriptor.lineage_marker * 31 + descriptor.ow_id * 17)
    count = 20 if descriptor.archetype != "spined" else 24
    points = _polygon_points(
        center=(cx, cy),
        radius=radius,
        count=count,
        archetype=descriptor.archetype,
        seed=seed,
    )
    fill = (*descriptor.trait_color.rendered_rgb, 245)
    outline = descriptor.trait_color.outline_rgba

    # Cilia are beneath the membrane and remain visible as a silhouette cue.
    cilia_count = descriptor.cilia_level * 4
    if cilia_count:
        cilia_color = (*outline[:3], 155)
        for index in range(cilia_count):
            angle = 2.0 * math.pi * index / cilia_count
            inner = radius * 0.93
            outer = radius * (1.12 + 0.04 * ((index + seed) % 3))
            pygame.draw.line(
                surface,
                cilia_color,
                (cx + math.cos(angle) * inner, cy + math.sin(angle) * inner),
                (cx + math.cos(angle) * outer, cy + math.sin(angle) * outer),
                max(1, scale),
            )

    pygame.draw.polygon(surface, fill, points)
    pygame.draw.aalines(surface, outline, True, points)
    pygame.draw.lines(surface, outline, True, points, max(scale, side // 30))

    # Stable membrane patterns provide non-color identity redundancy.
    if descriptor.membrane_pattern == "dotted":
        for index in range(6):
            angle = 2.0 * math.pi * index / 6.0 + 0.2
            pygame.draw.circle(
                surface,
                (*outline[:3], 170),
                (
                    int(cx + math.cos(angle) * radius * 0.72),
                    int(cy + math.sin(angle) * radius * 0.72),
                ),
                max(1, side // 35),
            )
    elif descriptor.membrane_pattern == "banded":
        pygame.draw.arc(
            surface,
            (*outline[:3], 170),
            pygame.Rect(
                int(cx - radius * 0.78),
                int(cy - radius * 0.78),
                int(radius * 1.56),
                int(radius * 1.56),
            ),
            0.15 * math.pi,
            0.95 * math.pi,
            max(scale, side // 40),
        )

    # Nucleus and lineage rune.
    nucleus_radius = max(2, int(radius * (0.23 + 0.03 * descriptor.developmental_stage)))
    nucleus_color = tuple(min(255, int(channel * 0.65 + 70)) for channel in fill[:3])
    pygame.draw.circle(surface, (*nucleus_color, 235), (int(cx), int(cy)), nucleus_radius)
    pygame.draw.circle(
        surface, (*outline[:3], 210), (int(cx), int(cy)), nucleus_radius, max(1, scale)
    )
    if descriptor.nucleus_style == "double":
        pygame.draw.circle(
            surface,
            (*outline[:3], 190),
            (int(cx + nucleus_radius * 0.55), int(cy - nucleus_radius * 0.35)),
            max(1, nucleus_radius // 3),
        )

    marker = descriptor.lineage_marker & 0xF
    angle = marker / 16.0 * 2.0 * math.pi
    pygame.draw.line(
        surface,
        (*outline[:3], 230),
        (cx + math.cos(angle) * radius * 0.45, cy + math.sin(angle) * radius * 0.45),
        (cx + math.cos(angle) * radius * 0.78, cy + math.sin(angle) * radius * 0.78),
        max(1, scale),
    )

    # Explicit predator spikes beyond the base archetype.
    for index in range(descriptor.spike_level * 2):
        angle = 2.0 * math.pi * index / max(1, descriptor.spike_level * 2)
        inner = (cx + math.cos(angle) * radius * 0.80, cy + math.sin(angle) * radius * 0.80)
        outer = (cx + math.cos(angle) * radius * 1.14, cy + math.sin(angle) * radius * 1.14)
        pygame.draw.line(surface, (*outline[:3], 220), inner, outer, max(1, scale))

    final = pygame.transform.smoothscale(surface, (max(4, int(size)), max(4, int(size))))
    return _convert_alpha(final)


def render_environment_sprite(
    instance: EnvironmentInstance,
    size: int,
    theme: Theme,
) -> Any:
    del theme
    import pygame

    side = max(4, int(size))
    surface = pygame.Surface((side, side), pygame.SRCALPHA, 32)
    center = (side // 2, side // 2)
    color = instance.color
    value = max(0.0, min(1.0, instance.value))
    radius = max(1, int(side * (0.16 + 0.20 * value)))
    variant = int(instance.variant)

    if instance.kind == EnvironmentKind.FOOD:
        # Three nutrient granules/leaf lobes.
        for angle in (0.0, 2.1, 4.2):
            offset = side * 0.16
            cx = int(center[0] + math.cos(angle + variant * 0.1) * offset)
            cy = int(center[1] + math.sin(angle + variant * 0.1) * offset)
            pygame.draw.ellipse(surface, color, (cx - radius, cy - radius // 2, radius * 2, radius))
        pygame.draw.circle(surface, (250, 225, 105, 220), center, max(1, radius // 2))
    elif instance.kind == EnvironmentKind.TOXIN:
        for index in range(3):
            r = max(1, radius - index * max(1, radius // 4))
            cx = center[0] + ((variant + index) % 3 - 1) * max(1, side // 10)
            cy = center[1] + ((variant + index * 2) % 3 - 1) * max(1, side // 10)
            pygame.draw.circle(
                surface,
                (*color[:3], max(65, color[3] - index * 40)),
                (cx, cy),
                r,
                max(1, side // 18),
            )
    elif instance.kind == EnvironmentKind.WASTE:
        for index in range(4):
            x = int(side * (0.22 + ((variant * 3 + index * 5) % 11) / 18.0))
            y = int(side * (0.22 + ((variant * 7 + index * 3) % 11) / 18.0))
            pygame.draw.circle(surface, color, (x, y), max(1, radius // 2))
    elif instance.kind == EnvironmentKind.OBSTACLE:
        points = [
            (side // 2, max(0, side // 10)),
            (side - side // 8, side // 3),
            (side - side // 5, side - side // 8),
            (side // 4, side - side // 10),
            (side // 10, side // 2),
        ]
        pygame.draw.polygon(surface, color, points)
        pygame.draw.lines(surface, (190, 205, 225, 230), True, points, max(1, side // 12))
        pygame.draw.line(surface, (135, 150, 175, 220), points[0], points[2], max(1, side // 18))
    elif instance.kind == EnvironmentKind.SIGNAL:
        for ring in (0.20, 0.34, 0.48):
            r = max(1, int(side * ring))
            pygame.draw.circle(surface, color, center, r, max(1, side // 20))
    elif instance.kind == EnvironmentKind.DEAD_SHELL:
        pygame.draw.circle(surface, color, center, max(2, int(side * 0.34)), max(1, side // 12))
        pygame.draw.line(
            surface,
            color,
            (side // 3, side // 3),
            (2 * side // 3, 2 * side // 3),
            max(1, side // 12),
        )
        pygame.draw.line(
            surface,
            color,
            (2 * side // 3, side // 3),
            (side // 3, 2 * side // 3),
            max(1, side // 12),
        )
    else:
        pygame.draw.rect(surface, color, surface.get_rect(), max(1, side // 10))
    return _convert_alpha(surface)
