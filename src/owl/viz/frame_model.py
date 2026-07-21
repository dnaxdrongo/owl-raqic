from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from owl.viz.event_bus import VisualEvent
from owl.viz.sprite_state import SpriteState

if TYPE_CHECKING:
    from owl.viz.scene import VisualScene


@dataclass(frozen=True)
class TextLabel:
    position: tuple[float, float]
    text: str
    color: tuple[int, int, int, int] = (255, 255, 255, 255)


@dataclass(frozen=True)
class VisualSelection:
    """Read-only visual selection applied at render boundaries."""

    overlay: str = "health"
    include_events: bool = True
    include_glyphs: bool = True
    include_debug: bool = True
    include_effects: bool = True
    selected_ow_id: int | None = None
    fields: tuple[str, ...] = ()


@dataclass
class VisualFrame:
    """Backend-neutral visual payload.

    Interpretability mode primarily uses ``scene``.  compatibility dense imagery,
    markers, lines, glyphs, and SpriteState tuples remain for compatibility.
    """

    rgba: np.ndarray | None
    scene: VisualScene | None = None
    scientific_tick: int = 0
    subframe_index: int = 0
    subframe_count: int = 1
    markers: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float32))
    marker_colors: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=np.float32))
    marker_sizes: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    lines: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float32))
    line_colors: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=np.float32))
    arrows: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=np.float32))
    sprite_positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float32))
    sprite_states: tuple[SpriteState, ...] = ()
    glyph_lines: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float32))
    glyph_line_colors: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 4), dtype=np.float32)
    )
    texts: tuple[TextLabel, ...] = ()
    events: tuple[VisualEvent, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def estimated_nbytes(self) -> int:
        arrays = (
            self.rgba,
            self.markers,
            self.marker_colors,
            self.marker_sizes,
            self.lines,
            self.line_colors,
            self.arrows,
            self.sprite_positions,
            self.glyph_lines,
            self.glyph_line_colors,
        )
        return sum(int(array.nbytes) for array in arrays if array is not None)
