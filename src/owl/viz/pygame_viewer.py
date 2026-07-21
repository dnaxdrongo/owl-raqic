"""Pygame real-time viewer interface for dense field rendering.

The viewer is a presentation layer only. It reads ``WorldState`` and may keep a
local frame history for replay, but it does not mutate simulation state or own
the engine loop.
"""

from __future__ import annotations

import contextlib
from collections import deque
from dataclasses import dataclass
from types import ModuleType
from typing import Any, cast

import numpy as np

from owl.core.actions import Action
from owl.core.config import SimulationConfig
from owl.core.state import WorldState, field_shape
from owl.viz.overlays import default_overlays
from owl.viz.palettes import (
    action_palette,
    food_palette,
    genome_palette,
    integration_palette,
    patch_palette,
    signal_overlay_palette,
    toxin_palette,
    trust_palette,
    type_palette,
    waste_palette,
)
from owl.viz.pygame_sprites import SpriteRenderer

pygame: ModuleType | None
try:  # pragma: no cover - exercised only when pygame is installed.
    import pygame as pygame_module
except Exception:  # noqa: BLE001 - visualization must import without pygame.
    pygame = None
else:
    pygame = pygame_module


@dataclass(slots=True)
class _FrameRecord:
    """One rendered frame stored for pause/replay."""

    tick: int
    overlay: str
    rgb: np.ndarray
    stats: dict[str, float | int | str]


def _array_or_fallback(
    state: WorldState,
    preferred_name: str,
    fallback: Any,
    *,
    expected_shape: tuple[int, ...],
    dtype: Any | None = None,
) -> np.ndarray:
    """Resolve an optional visualization array without mutating ``state``.

    ``WorldState`` declares several advanced and RAQIC arrays as optional.
    Those attributes therefore exist even when their value is ``None``. The
    normal three-argument ``getattr`` fallback does not handle that case, so
    visual consumers must explicitly select the compatibility or neutral fallback.
    """
    preferred = getattr(state, preferred_name, None)
    using_fallback = preferred is None
    value = fallback if using_fallback else preferred
    array = cast(
        np.ndarray,
        np.asarray(value, dtype=dtype) if dtype is not None else np.asarray(value),
    )
    if tuple(array.shape) != tuple(expected_shape):
        source = "fallback" if using_fallback else preferred_name
        raise ValueError(
            f"{source} resolved for {preferred_name!r} has shape {array.shape}; "
            f"expected {expected_shape}"
        )
    return array


def _state_stats(state: WorldState) -> dict[str, float | int | str]:
    """Return lightweight stats for HUD/replay without mutating state."""
    alive = (state.health > 0.0) & (~state.obstacle)
    return {
        "tick": int(state.tick),
        "alive": int(np.count_nonzero(alive)),
        "mean_integration": float(np.mean(state.integration[alive])) if np.any(alive) else 0.0,
        "mean_health": float(np.mean(state.health[alive])) if np.any(alive) else 0.0,
        "food_total": float(np.sum(state.food, dtype=np.float64)),
        "signal_total": float(np.sum(state.signal, dtype=np.float64)),
    }


class PygameViewer:
    """Real-time Pygame viewer for array fields.

    The viewer supports:
    - view switching;
    - distinct OW type sprites;
    - food/toxin/patch colored regions;
    - pan and zoom;
    - hover stats modal;
    - rendering pause/play independent of backend simulation;
    - replay rewind/fast-forward over frames already passed to ``draw``.

    If Pygame is not installed, the object enters headless mode. Palette and
    replay functions still work; ``draw`` records frames but performs no GUI
    calls.
    """

    def __init__(
        self, height: int, width: int, scale: int = 6, title: str = "Observer-Window Life"
    ):
        if height <= 0 or width <= 0:
            raise ValueError(f"height and width must be positive, got {(height, width)}")
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale!r}")

        self.height = int(height)
        self.width = int(width)
        self.scale = int(scale)
        self.zoom = 1.0
        self.min_zoom = 0.25
        self.max_zoom = 32.0
        self.offset_x = 0
        self.offset_y = 0
        self.title = str(title)
        self.overlay = "integration"
        self.overlays = default_overlays()
        self.running = True
        self.paused_render = False
        self.step_once = False
        self.dragging = False
        self._drag_start: tuple[int, int] | None = None
        self._drag_offset_start: tuple[int, int] | None = None
        self.mouse_pos: tuple[int, int] | None = None
        self.hover_cell: tuple[int, int] | None = None
        self.history: deque[_FrameRecord] = deque(maxlen=5000)
        self.playback_index: int | None = None
        self.last_tick_recorded: int | None = None
        self.last_rgb: np.ndarray | None = None
        self.last_stats: dict[str, float | int | str] = {}
        self.pygame_available = pygame is not None

        self.screen: Any = None
        self.surface: Any = None
        self.clock: Any = None
        self.font: Any = None
        self.small_font: Any = None
        self.sprite_cache: dict[tuple[int, int], Any] = {}
        self.dynamic_sprite_renderer = SpriteRenderer()

        if pygame is not None:  # pragma: no cover - needs pygame runtime.
            pygame.init()
            with contextlib.suppress(Exception):
                pygame.font.init()
            pygame.display.set_caption(self.title)
            self.screen = pygame.display.set_mode(
                (self.width * self.scale, self.height * self.scale)
            )
            self.surface = pygame.Surface((self.width, self.height))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 18) if pygame.font.get_init() else None
            self.small_font = pygame.font.Font(None, 15) if pygame.font.get_init() else None

    def _cell_pixel_size(self) -> int:
        """Return current displayed cell size in pixels."""
        return max(1, int(round(self.scale * self.zoom)))

    def _screen_to_cell(self, pos: tuple[int, int]) -> tuple[int, int] | None:
        """Convert a screen position into ``(y, x)`` cell coordinates."""
        cell_px = self._cell_pixel_size()
        x = int(np.floor((pos[0] - self.offset_x) / cell_px))
        y = int(np.floor((pos[1] - self.offset_y) / cell_px))
        if 0 <= y < self.height and 0 <= x < self.width:
            return y, x
        return None

    def _zoom_at(self, factor: float, mouse_pos: tuple[int, int] | None = None) -> None:
        """Zoom around the current mouse position while preserving world focus."""
        old_zoom = self.zoom
        new_zoom = float(np.clip(self.zoom * factor, self.min_zoom, self.max_zoom))
        if np.isclose(new_zoom, old_zoom):
            return

        if mouse_pos is None:
            mouse_pos = (self.width * self.scale // 2, self.height * self.scale // 2)

        old_px = max(1.0, self.scale * old_zoom)
        world_x = (mouse_pos[0] - self.offset_x) / old_px
        world_y = (mouse_pos[1] - self.offset_y) / old_px

        self.zoom = new_zoom
        new_px = self.scale * self.zoom
        self.offset_x = int(round(mouse_pos[0] - world_x * new_px))
        self.offset_y = int(round(mouse_pos[1] - world_y * new_px))

    def _set_overlay_by_hotkey(self, hotkey: str) -> bool:
        """Select a visualization layer using a single-character hotkey."""
        for spec in self.overlays:
            if spec.hotkey == hotkey:
                self.overlay = spec.name
                return True
        return False

    def _change_playback_index(self, delta: int) -> None:
        """Move replay frame pointer within available history."""
        if not self.history:
            self.playback_index = None
            return
        current = len(self.history) - 1 if self.playback_index is None else self.playback_index
        self.playback_index = int(np.clip(current + delta, 0, len(self.history) - 1))
        self.paused_render = True

    def handle_events(self) -> None:
        """Handle keyboard, mouse, pan/zoom, replay, and window events.

        Mutates only viewer-local state. It never mutates ``WorldState``.
        """
        if pygame is None:  # headless import/test mode
            return

        for event in pygame.event.get():  # pragma: no cover - needs pygame runtime.
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                key = event.key
                if key == pygame.K_ESCAPE:
                    self.running = False
                elif key in (pygame.K_SPACE, pygame.K_p):
                    self.paused_render = not self.paused_render
                    if self.paused_render:
                        self.playback_index = len(self.history) - 1 if self.history else None
                    else:
                        self.playback_index = None
                elif key == pygame.K_PERIOD:
                    self.step_once = True
                    self._change_playback_index(1)
                elif key in (pygame.K_LEFT, pygame.K_COMMA):
                    self._change_playback_index(
                        -10 if (pygame.key.get_mods() & pygame.KMOD_SHIFT) else -1
                    )
                elif key in (pygame.K_RIGHT, pygame.K_SLASH):
                    self._change_playback_index(
                        10 if (pygame.key.get_mods() & pygame.KMOD_SHIFT) else 1
                    )
                elif key in (pygame.K_PLUS, pygame.K_EQUALS):
                    self._zoom_at(1.20, self.mouse_pos)
                elif key == pygame.K_MINUS:
                    self._zoom_at(1.0 / 1.20, self.mouse_pos)
                elif key == pygame.K_HOME:
                    self.offset_x = 0
                    self.offset_y = 0
                    self.zoom = 1.0
                elif pygame.K_0 <= key <= pygame.K_9:
                    self._set_overlay_by_hotkey(str(key - pygame.K_0))
            elif (
                getattr(pygame, "MOUSEWHEEL", None) is not None and event.type == pygame.MOUSEWHEEL
            ):
                self._zoom_at(1.15 if event.y > 0 else 1.0 / 1.15, self.mouse_pos)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.mouse_pos = tuple(event.pos)
                if event.button in (1, 2):
                    self.dragging = True
                    self._drag_start = tuple(event.pos)
                    self._drag_offset_start = (self.offset_x, self.offset_y)
                elif event.button == 4:
                    self._zoom_at(1.15, tuple(event.pos))
                elif event.button == 5:
                    self._zoom_at(1.0 / 1.15, tuple(event.pos))
                elif event.button == 3:
                    self.offset_x = 0
                    self.offset_y = 0
                    self.zoom = 1.0
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button in (1, 2):
                    self.dragging = False
                    self._drag_start = None
                    self._drag_offset_start = None
            elif event.type == pygame.MOUSEMOTION:
                self.mouse_pos = tuple(event.pos)
                self.hover_cell = self._screen_to_cell(self.mouse_pos)
                if (
                    self.dragging
                    and self._drag_start is not None
                    and self._drag_offset_start is not None
                ):
                    dx = event.pos[0] - self._drag_start[0]
                    dy = event.pos[1] - self._drag_start[1]
                    self.offset_x = self._drag_offset_start[0] + dx
                    self.offset_y = self._drag_offset_start[1] + dy

    def field_to_rgb(self, state: WorldState) -> np.ndarray:
        """Convert the selected simulation view into a Pygame RGB array.

        Parameters
        ----------
        state:
            Runtime dense state. This function does not mutate state.

        Returns
        -------
        np.ndarray
            RGB array with shape ``(width, height, 3)`` and dtype ``uint8``.
        """
        h, w = field_shape(state)
        if (h, w) != (self.height, self.width):
            raise ValueError(
                f"viewer size {(self.height, self.width)} does not match state shape {(h, w)}"
            )

        name = self.overlay
        if name == "integration":
            rgb = integration_palette(state)
        elif name == "type":
            rgb = type_palette(state)
        elif name == "action":
            rgb = action_palette(state)
        elif name == "food":
            rgb = food_palette(state)
        elif name == "toxin":
            rgb = toxin_palette(state)
        elif name == "patches":
            rgb = patch_palette(state)
        elif name == "waste":
            rgb = waste_palette(state)
        elif name == "trust":
            rgb = trust_palette(state)
        elif name == "genome":
            rgb = genome_palette(state)
        elif name.startswith("signal:"):
            try:
                channel = int(name.split(":", 1)[1])
                channel = min(max(channel, 0), max(0, state.signal.shape[-1] - 1))
                rgb = signal_overlay_palette(state, channel)
            except Exception:
                rgb = integration_palette(state)
        else:
            rgb = integration_palette(state)

        if rgb.shape != (self.width, self.height, 3):
            raise ValueError(
                f"palette returned shape {rgb.shape}, expected {(self.width, self.height, 3)}"
            )
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return cast(np.ndarray, np.ascontiguousarray(rgb))

    def _append_history(self, state: WorldState, rgb: np.ndarray) -> None:
        """Store the current frame when the tick or visualization mode changes."""
        tick = int(state.tick)
        key_changed = (
            self.last_tick_recorded != tick
            or not self.history
            or self.history[-1].overlay != self.overlay
        )
        if key_changed:
            record = _FrameRecord(
                tick=tick, overlay=self.overlay, rgb=rgb.copy(), stats=_state_stats(state)
            )
            self.history.append(record)
            self.last_tick_recorded = tick
            if not self.paused_render:
                self.playback_index = len(self.history) - 1

    def _selected_frame_rgb(self, current_rgb: np.ndarray) -> np.ndarray:
        """Return current or replay RGB frame."""
        if self.paused_render and self.history:
            index = len(self.history) - 1 if self.playback_index is None else self.playback_index
            index = int(np.clip(index, 0, len(self.history) - 1))
            self.playback_index = index
            return self.history[index].rgb
        return current_rgb

    def _sprite_shape(self, type_id: int) -> str:
        """Return stable sprite shape name for an OW type id."""
        shapes = ("circle", "triangle", "diamond", "square", "cross", "hex", "small_circle", "bar")
        return shapes[int(type_id) % len(shapes)]

    def _draw_type_sprites(self, state: WorldState) -> None:
        """Draw state-dependent action sprites over the raster view.

        The renderer is level-of-detail controlled and reads state only. At
        low zoom the base heatmap remains authoritative; at higher zoom it adds
        action glyphs, confidence/coherence rings, stress outlines, and compact
        health/resource bars.
        """
        if pygame is None or self.screen is None:  # pragma: no cover
            return
        cell_px = self._cell_pixel_size()
        if cell_px < 3:
            return

        visible_y0 = max(0, int(np.floor((-self.offset_y) / cell_px)) - 1)
        visible_x0 = max(0, int(np.floor((-self.offset_x) / cell_px)) - 1)
        sw, sh = self.screen.get_size()
        visible_y1 = min(self.height, int(np.ceil((sh - self.offset_y) / cell_px)) + 1)
        visible_x1 = min(self.width, int(np.ceil((sw - self.offset_x) / cell_px)) + 1)

        living = (state.health > 0.0) & (~state.obstacle)
        grid_shape = (self.height, self.width)
        action_shape = (*grid_shape, int(state.possibility.shape[-1]))
        readout = _array_or_fallback(
            state,
            "raqic_readout",
            state.readout,
            expected_shape=grid_shape,
        )
        probs = _array_or_fallback(
            state,
            "raqic_probabilities",
            state.possibility,
            expected_shape=action_shape,
            dtype=float,
        )
        confidence = np.max(probs, axis=-1)
        entropy = -np.sum(
            np.where(probs > 0, probs * np.log(np.maximum(probs, 1e-12)), 0.0),
            axis=-1,
        )
        coherence = _array_or_fallback(
            state,
            "noetic_C",
            state.integration,
            expected_shape=grid_shape,
        )
        starvation = _array_or_fallback(
            state,
            "starvation_debt",
            np.zeros_like(state.health),
            expected_shape=grid_shape,
        )
        stage = _array_or_fallback(
            state,
            "development_stage",
            np.zeros_like(state.health),
            expected_shape=grid_shape,
        )
        lineage = _array_or_fallback(
            state,
            "lineage_id",
            np.full_like(state.health, -1),
            expected_shape=grid_shape,
        )
        age = _array_or_fallback(
            state,
            "age",
            np.zeros_like(state.health),
            expected_shape=grid_shape,
        )
        phase = _array_or_fallback(
            state,
            "phase",
            np.zeros_like(state.health),
            expected_shape=grid_shape,
        )
        parent = getattr(state, "raqic_parent_intention", None)
        if parent is None:
            parent_pressure = np.zeros_like(state.health)
        else:
            parent_array = np.asarray(parent)
            if tuple(parent_array.shape) != action_shape:
                raise ValueError(
                    "raqic_parent_intention has shape "
                    f"{parent_array.shape}; expected {action_shape}"
                )
            parent_pressure = np.max(parent_array, axis=-1)
        max_age = max(float(np.max(age)), 1.0)
        live_resources = state.resource[living]
        repro_threshold = float(np.quantile(live_resources, 0.75)) if live_resources.size else 1.0

        for y in range(visible_y0, visible_y1):  # pragma: no cover
            for x in range(visible_x0, visible_x1):
                if not living[y, x]:
                    continue
                rect = pygame.Rect(
                    self.offset_x + x * cell_px,
                    self.offset_y + y * cell_px,
                    cell_px,
                    cell_px,
                )
                self.dynamic_sprite_renderer.draw_cell(
                    self.screen,
                    rect,
                    int(readout[y, x]),
                    float(state.health[y, x]),
                    float(confidence[y, x]),
                    resource=float(state.resource[y, x]),
                    entropy=float(entropy[y, x]),
                    coherence=float(coherence[y, x]),
                    toxin=float(state.toxin[y, x]),
                    starvation=float(starvation[y, x]),
                    developmental_stage=int(stage[y, x]),
                    lineage_marker=int(lineage[y, x]),
                    age_fraction=float(age[y, x]) / max_age,
                    parent_pressure=float(parent_pressure[y, x]),
                    phase=float(phase[y, x]),
                    reproduction_ready=bool(
                        state.resource[y, x] >= repro_threshold and state.health[y, x] > 0.5
                    ),
                )
                # Native Pygame primitives remain as a lightweight
                # high-zoom orientation marker and an explicit fallback path.
                # They are derived-only visuals and never mutate simulation state.
                if cell_px >= 12:
                    cx, cy = rect.center
                    dot_radius = max(1, int(cell_px * 0.055))
                    pygame.draw.circle(
                        self.screen,
                        (235, 240, 248),
                        (cx, cy),
                        dot_radius,
                    )
                    theta = float(phase[y, x])
                    tip = (
                        int(cx + np.cos(theta) * cell_px * 0.36),
                        int(cy - np.sin(theta) * cell_px * 0.36),
                    )
                    left = (
                        int(cx + np.cos(theta + 2.55) * cell_px * 0.16),
                        int(cy - np.sin(theta + 2.55) * cell_px * 0.16),
                    )
                    right = (
                        int(cx + np.cos(theta - 2.55) * cell_px * 0.16),
                        int(cy - np.sin(theta - 2.55) * cell_px * 0.16),
                    )
                    pygame.draw.polygon(
                        self.screen,
                        (205, 218, 236),
                        (tip, left, right),
                    )

    def _draw_patch_grid(self) -> None:
        """Draw fixed patch grid lines as a region reference."""
        if pygame is None or self.screen is None:  # pragma: no cover
            return
        # Infer patch size from visible parent-id changes is not robust. Draw
        # Draw a light grid every five cells; moving patch membership is
        # already visible in the patch palette through state.parent_id.
        cell_px = self._cell_pixel_size()
        if cell_px < 4:
            return
        color = (70, 70, 85)
        for x in range(0, self.width + 1, 5):  # pragma: no cover
            sx = self.offset_x + x * cell_px
            pygame.draw.line(
                self.screen,
                color,
                (sx, self.offset_y),
                (sx, self.offset_y + self.height * cell_px),
                1,
            )
        for y in range(0, self.height + 1, 5):  # pragma: no cover
            sy = self.offset_y + y * cell_px
            pygame.draw.line(
                self.screen,
                color,
                (self.offset_x, sy),
                (self.offset_x + self.width * cell_px, sy),
                1,
            )

    def _draw_tooltip(self, state: WorldState) -> None:
        """Draw hover modal with OW stats."""
        if pygame is None or self.screen is None or self.font is None:  # pragma: no cover
            return
        if self.mouse_pos is None:
            return
        cell = self._screen_to_cell(self.mouse_pos)
        self.hover_cell = cell
        if cell is None:
            return
        y, x = cell
        if state.health[y, x] <= 0.0 or state.obstacle[y, x]:
            return

        action_value = int(state.readout[y, x])
        action_label = (
            Action(action_value).name if 0 <= action_value < len(Action) else str(action_value)
        )
        lines = [
            f"OW ({y}, {x}) type={int(state.ow_type[y, x])}",
            f"action={action_label}",
            f"health={state.health[y, x]:.3f}  resource={state.resource[y, x]:.3f}",
            f"boundary={state.boundary[y, x]:.3f} integration={state.integration[y, x]:.3f}",
            f"memory={state.memory[y, x]:.3f} phase={state.phase[y, x]:.2f}",
            f"food={state.food[y, x]:.3f} toxin={state.toxin[y, x]:.3f}",
        ]
        digestion = getattr(state, "digestion", None)
        genome = getattr(state, "genome", None)
        if digestion is not None:
            digestion_array = _array_or_fallback(
                state,
                "digestion",
                np.zeros_like(state.food),
                expected_shape=(self.height, self.width),
            )
            waste = _array_or_fallback(
                state,
                "waste",
                np.zeros_like(state.food),
                expected_shape=(self.height, self.width),
            )
            lines.append(f"digestion={digestion_array[y, x]:.3f} waste={waste[y, x]:.3f}")
        if isinstance(genome, np.ndarray):
            preview = genome[y, x, : min(3, genome.shape[-1])].round(2).tolist()
            lines.append(f"genome[0:3]={preview}")
        rendered = [self.font.render(text, True, (235, 238, 245)) for text in lines]
        width = max(surface.get_width() for surface in rendered) + 14
        height = sum(surface.get_height() for surface in rendered) + 14
        mx, my = self.mouse_pos
        sw, sh = self.screen.get_size()
        left = min(mx + 14, sw - width - 4)
        top = min(my + 14, sh - height - 4)
        rect = pygame.Rect(max(4, left), max(4, top), width, height)
        modal = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        modal.fill((10, 12, 18, 225))
        pygame.draw.rect(modal, (130, 150, 190, 230), modal.get_rect(), 1)
        y_cursor = 7
        for surface in rendered:
            modal.blit(surface, (7, y_cursor))
            y_cursor += surface.get_height()
        self.screen.blit(modal, rect)

    def _draw_hud(self, stats: dict[str, float | int | str]) -> None:
        """Draw the top-left HUD with visualization and replay state."""
        if pygame is None or self.screen is None or self.small_font is None:  # pragma: no cover
            return
        history_note = ""
        if self.paused_render and self.history:
            frame_index = (
                self.playback_index if self.playback_index is not None else len(self.history) - 1
            )
            history_note = f" frame={frame_index}/{len(self.history) - 1}"
        overlay_line = (
            f"overlay={self.overlay} zoom={self.zoom:.2f} paused={self.paused_render}{history_note}"
        )
        mean_integration = float(stats.get("mean_integration", 0.0))
        stats_line = (
            f"tick={stats.get('tick', 0)} alive={stats.get('alive', 0)} int={mean_integration:.3f}"
        )
        lines = [
            overlay_line,
            stats_line,
            "keys: 1-0 overlays, space pause, ←/→ replay, wheel zoom, drag pan",
        ]
        rendered = [self.small_font.render(text, True, (230, 235, 240)) for text in lines]
        width = max(s.get_width() for s in rendered) + 12
        height = sum(s.get_height() for s in rendered) + 10
        panel = pygame.Surface((width, height), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 150))
        y = 5
        for surface in rendered:
            panel.blit(surface, (6, y))
            y += surface.get_height()
        self.screen.blit(panel, (4, 4))

    def draw(self, state: WorldState, fps: int = 30) -> None:
        """Render current state.

        Mutates only viewer-local frame history and display resources. The
        supplied ``WorldState`` is never mutated.
        """
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps!r}")

        self.handle_events()
        current_rgb = self.field_to_rgb(state)
        self._append_history(state, current_rgb)
        rgb = self._selected_frame_rgb(current_rgb)
        self.last_rgb = rgb.copy()
        stats = _state_stats(state)
        self.last_stats = stats

        if pygame is None or self.screen is None or self.surface is None:  # headless mode
            return

        pygame.surfarray.blit_array(self.surface, rgb)  # pragma: no cover - needs pygame runtime.
        cell_px = self._cell_pixel_size()
        scaled_size = (self.width * cell_px, self.height * cell_px)
        scaled = pygame.transform.scale(self.surface, scaled_size)
        self.screen.fill((0, 0, 0))
        self.screen.blit(scaled, (self.offset_x, self.offset_y))
        self._draw_patch_grid()
        self._draw_type_sprites(state)
        self._draw_hud(stats)
        self._draw_tooltip(state)
        pygame.display.flip()
        if self.clock is not None:
            self.clock.tick(fps)

    def close(self) -> None:
        """Close viewer resources.

        Mutates only viewer-local fields and Pygame display state.
        """
        self.running = False
        if pygame is not None:  # pragma: no cover - needs pygame runtime.
            try:
                pygame.display.quit()
                pygame.quit()
            except Exception:
                pass


def create_viewer(cfg: SimulationConfig) -> PygameViewer | None:
    """Construct viewer from config if visualization is enabled.

    Returns ``None`` when visualization is disabled or the backend is ``"none"``.
    If backend is ``"pygame"`` but Pygame is unavailable, a headless
    ``PygameViewer`` is returned so palette/replay code remains testable.
    """
    if not cfg.visualization.enabled or cfg.visualization.backend == "none":
        return None
    return PygameViewer(
        height=cfg.world.height,
        width=cfg.world.width,
        scale=cfg.visualization.scale,
        title="Observer-Window Life",
    )
