from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from owl.core.actions import Action
from owl.viz.scene import VisualScene, VisualSpriteInstance


@dataclass(frozen=True)
class HUDState:
    tick: int
    visible_ows: int
    environment_items: int
    effect_count: int
    selected: VisualSpriteInstance | None
    lines: tuple[str, ...]


def build_hud_state(scene: VisualScene) -> HUDState:
    selected = next((item for item in scene.sprites if item.status.selected), None)
    lines = [
        f"Tick {scene.tick}",
        f"Visible OWs {len(scene.sprites):,}",
        f"Effects {len(scene.effects):,}",
        f"Environment {len(scene.environment):,}",
        f"Zoom {scene.camera.zoom:.2f}px/cell",
    ]
    if selected is not None:
        color = selected.descriptor.trait_color
        lines.extend(
            (
                "",
                f"OW #{selected.ow_id}",
                f"Action {Action(selected.action).name}",
                f"Trait Hex {color.raw_hex}",
                f"Lineage {selected.descriptor.lineage_marker}",
                f"Health {selected.status.health_fraction:.3f}",
                f"Resource {selected.status.resource_fraction:.3f}",
                f"Integration {selected.status.integration:.3f}",
                f"Confidence {selected.status.confidence:.3f}",
                f"Entropy {selected.status.entropy:.3f}",
            )
        )
    return HUDState(
        tick=scene.tick,
        visible_ows=len(scene.sprites),
        environment_items=len(scene.environment),
        effect_count=len(scene.effects),
        selected=selected,
        lines=tuple(lines),
    )


class TextSurfaceCache:
    def __init__(self, font: Any, *, max_entries: int = 512) -> None:
        self.font = font
        self.max_entries = max(32, int(max_entries))
        self._cache: dict[tuple[str, tuple[int, int, int, int]], Any] = {}

    def render(self, text: str, color: tuple[int, int, int, int]) -> Any:
        key = (str(text), color)
        surface = self._cache.get(key)
        if surface is None:
            surface, _rect = self.font.render(str(text), fgcolor=color)
            if len(self._cache) >= self.max_entries:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = surface
        return surface
