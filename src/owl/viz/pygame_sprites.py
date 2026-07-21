from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from owl.viz.sprite_atlas import SpriteAtlas, orientation_bucket
from owl.viz.sprite_state import SpriteState, build_sprite_state
from owl.viz.themes import get_theme


@dataclass
class SpriteCache:
    cache: dict[tuple[Any, ...], Any] = field(default_factory=dict)

    def key(self, action: int, size: int, theme: str) -> tuple[Any, ...]:
        return (int(action), int(size), str(theme))

    def get_or_create(self, action: int, size: int, theme: str, factory: Any) -> Any:
        key = self.key(action, size, theme)
        if key not in self.cache:
            self.cache[key] = factory()
        return self.cache[key]


def lod_for_cell_pixels(cell_px: float) -> str:
    if cell_px < 4:
        return "overview"
    if cell_px < 9:
        return "medium"
    if cell_px < 20:
        return "detail"
    return "focus"


class SpriteRenderer:
    """Compatibility renderer delegated to the stable trait-based atlas."""

    def __init__(
        self,
        theme: str = "owl_dark_neon",
        atlas: SpriteAtlas | None = None,
    ) -> None:
        self.theme = theme
        self.cache = SpriteCache()
        self.atlas = atlas or SpriteAtlas(get_theme(theme))

    def draw_state(self, surface: Any, rect: Any, state: SpriteState) -> None:
        import pygame

        x, y, w, h = map(int, rect)
        size = max(4, min(w, h))
        body = self.atlas.get_body(
            state.descriptor,
            size=size,
            orientation_bucket=orientation_bucket(state.phase_notch * 180.0 / 3.141592653589793),
        )
        target = body.get_rect(center=(x + w // 2, y + h // 2))
        surface.blit(body, target)
        cx, cy = target.center
        radius = max(2, min(w, h) // 2)
        if state.selected:
            pygame.draw.circle(surface, (255, 245, 120, 255), (cx, cy), radius + 2, 1)
        if state.cracked_outline:
            pygame.draw.line(
                surface,
                (255, 80, 80, 255),
                (cx - radius // 2, cy - radius // 2),
                (cx + radius // 2, cy + radius // 2),
                1,
            )
        if state.hazard_outline:
            pygame.draw.circle(surface, (220, 255, 40, 255), (cx, cy), radius + 1, 1)
        if state.debug_marker:
            pygame.draw.line(surface, (255, 0, 255, 255), target.topleft, target.bottomright, 1)
            pygame.draw.line(surface, (255, 0, 255, 255), target.topright, target.bottomleft, 1)

    def draw_cell(
        self,
        surface: Any,
        rect: Any,
        action: int,
        health: float = 1.0,
        confidence: float = 1.0,
        *,
        resource: float = 1.0,
        entropy: float = 0.0,
        coherence: float = 0.0,
        toxin: float = 0.0,
        starvation: float = 0.0,
        communication_channel: int = -1,
        debug_marker: str = "",
        developmental_stage: int = 0,
        lineage_marker: int = -1,
        age_fraction: float = 0.0,
        parent_pressure: float = 0.0,
        phase: float = 0.0,
        reproduction_ready: bool = False,
        selected: bool = False,
    ) -> None:
        state = build_sprite_state(
            action=action,
            health=health,
            resource=resource,
            confidence=confidence,
            entropy=entropy,
            coherence=coherence,
            toxin=toxin,
            starvation=starvation,
            communication_channel=communication_channel,
            debug_marker=debug_marker,
            developmental_stage=developmental_stage,
            lineage_marker=lineage_marker,
            age_fraction=age_fraction,
            parent_pressure=parent_pressure,
            phase=phase,
            reproduction_ready=reproduction_ready,
            selected=selected,
        )
        self.draw_state(surface, rect, state)
