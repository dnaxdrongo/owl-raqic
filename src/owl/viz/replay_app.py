"""Interactive, read-only Pygame application for completed OWL replay bundles."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from owl.replay.zarr_source import ZarrReplayDataSource
from owl.viz.backends.pygame_scene_renderer import PygameSceneRenderer
from owl.viz.camera import (
    CameraState,
    fit_world,
    minimap_viewport_rect,
    pan,
    screen_to_world,
    update_follow_camera,
    zoom_at,
)
from owl.viz.frame_model import VisualSelection
from owl.viz.replay_timeline import PLAYBACK_SPEEDS, PlaybackClock
from owl.viz.scene import build_visual_scene
from owl.viz.sprite_atlas import SpriteAtlas
from owl.viz.themes import get_theme


class ReplayApplication:
    """Explore a completed replay bundle without rerunning or mutating science."""

    def __init__(
        self,
        bundle: str | Path,
        *,
        window_size: tuple[int, int] = (1600, 900),
        theme_name: str = "owl_dark_neon",
        headless: bool = False,
        output_dir: str | Path | None = None,
    ) -> None:
        if headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        import pygame
        import pygame.freetype

        self.pygame = pygame
        pygame.init()
        pygame.freetype.init()
        flags = 0 if headless else pygame.RESIZABLE
        self.screen = pygame.display.set_mode(window_size, flags)
        pygame.display.set_caption("OWL/RAQIC Experiment Replay Viewer")
        self.font = pygame.freetype.SysFont("DejaVu Sans", 15)
        self.small_font = pygame.freetype.SysFont("DejaVu Sans", 12)
        self.title_font = pygame.freetype.SysFont("DejaVu Sans", 20, bold=True)
        self.theme = get_theme(theme_name)
        self.source = ZarrReplayDataSource(bundle, cache_entries=16, verify_metadata=True)
        ticks = tuple(self.source.available_ticks())
        if not ticks:
            raise ValueError("replay bundle contains no committed ticks")
        self.timeline = PlaybackClock(ticks=ticks)
        self.current = self.source.load_snapshot(ticks[0])
        self.previous = None
        self.selection = VisualSelection(
            overlay="none",
            include_events=True,
            include_glyphs=True,
            include_debug=True,
            include_effects=True,
            selected_ow_id=None,
        )
        self.selected_ow_id: int | None = None
        self.follow_selected = False
        self.search_active = False
        self.search_text = ""
        self.dragging_world = False
        self.dragging_timeline = False
        self.last_mouse = (0, 0)
        self.status_message = "Loaded replay bundle; metadata checks passed"
        self.running = True
        self.headless = headless
        self.clock = pygame.time.Clock()
        self.atlas = SpriteAtlas(self.theme, max_entries=8192)
        self.renderer = PygameSceneRenderer(theme=self.theme, atlas=self.atlas, headless=headless)
        self.camera = CameraState(
            viewport=(0, 0, 1200, 800),
            world_shape=self.current.world_shape,
            center=(self.current.world_shape[0] / 2.0, self.current.world_shape[1] / 2.0),
            zoom=8.0,
        )
        root = (
            Path(output_dir)
            if output_dir is not None
            else (Path.home() / "OWL_Replay_Exports" / self.source.manifest.run_id)
        )
        self.output_dir = root.expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.inspector_scroll = 0
        self.overlay_modes = ("none", "health", "resource", "integration", "coherence")
        self.bookmarks: list[dict[str, int | None]] = []
        self._history_cache: dict[int, tuple[dict[str, Any], ...]] = {}
        self._lineage_cursor = 0
        self._prefetch_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="owl-replay-prefetch"
        )
        self._prefetch_futures: dict[int, Future[Any]] = {}
        self.button_rects: dict[str, Any] = {}
        self.button_tooltips: dict[str, str] = {}
        self._layout(*window_size)
        fit_world(self.camera)

    def _layout(self, width: int, height: int) -> None:
        self.sidebar_width = min(480, max(350, width // 4))
        self.timeline_height = 132
        self.camera.viewport = (
            0,
            0,
            max(320, width - self.sidebar_width),
            max(240, height - self.timeline_height),
        )
        self.sidebar_rect = self.pygame.Rect(
            width - self.sidebar_width,
            0,
            self.sidebar_width,
            height - self.timeline_height,
        )
        self.timeline_rect = self.pygame.Rect(
            0,
            height - self.timeline_height,
            width,
            self.timeline_height,
        )
        self.minimap_rect = self.pygame.Rect(
            self.sidebar_rect.x + 18,
            self.sidebar_rect.bottom - 190,
            self.sidebar_width - 36,
            155,
        )
        self.inspector_rect = self.pygame.Rect(
            self.sidebar_rect.x + 8,
            0,
            self.sidebar_rect.width - 16,
            max(120, self.minimap_rect.y - 8),
        )
        self._build_buttons()

    def _build_buttons(self) -> None:
        pygame = self.pygame
        labels = (
            ("first", "|<", "First tick"),
            ("back10", "<<", "Back 10 ticks"),
            ("back", "<", "Previous tick"),
            ("play", "Play", "Play or pause"),
            ("next", ">", "Next tick"),
            ("next10", ">>", "Forward 10 ticks"),
            ("last", ">|", "Last tick"),
            ("reverse", "Reverse", "Toggle playback direction"),
            ("speed", f"{self.timeline.speed:g}x", "Cycle playback speed"),
            ("loop_a", "Loop A", "Set loop start"),
            ("loop_b", "Loop B", "Set loop end"),
            ("loop_clear", "Clear", "Clear loop range"),
            ("fit", "Fit", "Fit entire world"),
        )
        self.button_rects.clear()
        self.button_tooltips.clear()
        x = self.timeline_rect.x + 18
        y = self.timeline_rect.y + 12
        for name, label, tooltip in labels:
            width = max(38, 18 + len(label) * 8)
            self.button_rects[name] = pygame.Rect(x, y, width, 27)
            self.button_tooltips[name] = tooltip
            x += width + 6

    def _set_selection(self, ow_id: int | None) -> None:
        self.selected_ow_id = ow_id
        self.selection = VisualSelection(
            overlay=self.selection.overlay,
            include_events=self.selection.include_events,
            include_glyphs=self.selection.include_glyphs,
            include_debug=self.selection.include_debug,
            include_effects=self.selection.include_effects,
            selected_ow_id=ow_id,
            fields=self.selection.fields,
        )
        if self.follow_selected:
            self.camera.follow_ow_id = ow_id
        self._lineage_cursor = 0

    def _set_tick(self, tick: int) -> None:
        if int(tick) == self.current.tick:
            return
        self.previous = self.current
        self.current = self.source.load_snapshot(int(tick))
        if self.follow_selected and self.selected_ow_id is not None:
            self.camera.follow_ow_id = self.selected_ow_id
        self._schedule_prefetch()

    def _select_at(self, position: tuple[int, int]) -> None:
        y, x = screen_to_world(self.camera, *position)
        row, column = int(np.floor(y + 0.5)), int(np.floor(x + 0.5))
        if not (
            0 <= row < self.current.world_shape[0] and 0 <= column < self.current.world_shape[1]
        ):
            return
        occupancy = np.asarray(self.current.arrays.get("occupancy"))
        health = np.asarray(self.current.arrays.get("health"))
        if occupancy.size and health[row, column] > 0 and occupancy[row, column] >= 0:
            self._set_selection(int(occupancy[row, column]))
            self.inspector_scroll = 0
            self.status_message = f"Selected OW {self.selected_ow_id}"

    def _center_from_minimap(self, position: tuple[int, int]) -> None:
        x_fraction = np.clip(
            (position[0] - self.minimap_rect.x) / max(1, self.minimap_rect.width), 0.0, 1.0
        )
        y_fraction = np.clip(
            (position[1] - self.minimap_rect.y) / max(1, self.minimap_rect.height), 0.0, 1.0
        )
        height, width = self.current.world_shape
        self.camera.center = (float(y_fraction * height), float(x_fraction * width))
        self.camera.mode = "free"

    def _timeline_seek(self, x: int) -> None:
        left = self.timeline_rect.x + 20
        right = self.timeline_rect.right - 20
        fraction = np.clip((x - left) / max(1, right - left), 0.0, 1.0)
        index = int(round(fraction * (len(self.timeline.ticks) - 1)))
        self._set_tick(self.timeline.seek_index(index))

    def _search_submit(self) -> None:
        try:
            ow_id = int(self.search_text)
        except ValueError:
            self.status_message = "Stable-ID search requires an integer OW ID"
        else:
            details = self.source.load_ow_details(self.current.tick, ow_id)
            if details is None:
                self.status_message = f"OW {ow_id} is not present at tick {self.current.tick}"
            else:
                self._set_selection(ow_id)
                if details.position is not None:
                    self.camera.center = (details.position[0] + 0.5, details.position[1] + 0.5)
                self.status_message = f"Selected OW {ow_id} by stable ID"
        self.search_active = False

    def _jump_event(self, direction: int) -> None:
        target = self.source.nearest_event_tick(
            self.current.tick,
            direction=direction,
            ow_id=self.selected_ow_id,
        )
        if target is None and self.selected_ow_id is not None:
            target = self.source.nearest_event_tick(self.current.tick, direction=direction)
        if target is None:
            self.status_message = "No matching event in that direction"
            return
        self._set_tick(self.timeline.seek_tick(target))
        self.status_message = f"Jumped to event tick {target}"

    def _schedule_prefetch(self) -> None:
        direction = 1 if self.timeline.direction >= 0 else -1
        for offset in range(1, 5):
            index = self.timeline.index + direction * offset
            if not 0 <= index < len(self.timeline.ticks):
                continue
            tick = int(self.timeline.ticks[index])
            if tick in self._prefetch_futures and not self._prefetch_futures[tick].done():
                continue
            self._prefetch_futures[tick] = self._prefetch_executor.submit(
                self.source.load_snapshot, tick
            )
        self._prefetch_futures = {
            tick: future
            for tick, future in self._prefetch_futures.items()
            if not future.done() or tick == self.current.tick
        }

    def _selected_details(self) -> Any | None:
        if self.selected_ow_id is None:
            return None
        return self.source.load_ow_details(self.current.tick, self.selected_ow_id)

    def _select_parent(self) -> None:
        details = self._selected_details()
        if details is None:
            self.status_message = "No selected OW details at this tick"
            return
        parent_id = int(details.values.get("parent_id", -1))
        if parent_id < 0:
            self.status_message = "Selected OW has no recorded parent"
            return
        parent = self.source.load_ow_details(self.current.tick, parent_id)
        if parent is None:
            self.status_message = f"Parent OW {parent_id} is not present at this tick"
            return
        self._set_selection(parent_id)
        self.status_message = f"Selected parent OW {parent_id}"

    def _select_child(self) -> None:
        if self.selected_ow_id is None:
            return
        children = self.source.find_children(self.current.tick, self.selected_ow_id)
        if not children:
            self.status_message = "No living recorded child at this tick"
            return
        self._set_selection(children[0])
        self.status_message = f"Selected child OW {children[0]}"

    def _cycle_lineage_member(self) -> None:
        details = self._selected_details()
        if details is None:
            return
        lineage_id = int(details.values.get("lineage_id", -1))
        members = self.source.find_lineage_members(self.current.tick, lineage_id)
        if not members:
            self.status_message = "No lineage members at this tick"
            return
        self._lineage_cursor = (self._lineage_cursor + 1) % len(members)
        self._set_selection(members[self._lineage_cursor])
        self.status_message = f"Lineage {lineage_id}: OW {self.selected_ow_id}"

    def _save_bookmark(self) -> None:
        row = {"tick": int(self.current.tick), "ow_id": self.selected_ow_id}
        if row not in self.bookmarks:
            self.bookmarks.append(row)
        target = self.output_dir / "bookmarks.json"
        target.write_text(json.dumps(self.bookmarks, indent=2) + "\n", encoding="utf-8")
        self.status_message = f"Bookmarked tick {self.current.tick}"

    def _selected_history(self) -> tuple[dict[str, Any], ...]:
        if self.selected_ow_id is None:
            return ()
        if self.selected_ow_id not in self._history_cache:
            self._history_cache[self.selected_ow_id] = self.source.load_ow_history(
                self.selected_ow_id,
                start_tick=self.timeline.ticks[0],
                end_tick=self.timeline.ticks[-1],
            )
        return self._history_cache[self.selected_ow_id]

    def _draw_history_plot(self, rect: Any) -> None:
        pygame = self.pygame
        pygame.draw.rect(self.screen, (5, 9, 16, 255), rect)
        pygame.draw.rect(self.screen, self.theme.panel_border, rect, 1)
        history = self._selected_history()
        if len(history) < 2:
            self._draw_text("History unavailable", (rect.x + 8, rect.y + 7), font=self.small_font)
            return
        fields = (
            ("health", self.theme.health_good),
            ("resource", self.theme.resource),
            ("integration", self.theme.selected),
        )
        min_tick = int(history[0].get("tick", 0))
        max_tick = max(min_tick + 1, int(history[-1].get("tick", min_tick + 1)))
        for field, color in fields:
            points: list[tuple[int, int]] = []
            for row in history:
                value = row.get(field)
                if value is None:
                    continue
                fraction_x = (int(row.get("tick", min_tick)) - min_tick) / (max_tick - min_tick)
                fraction_y = float(np.clip(float(value), 0.0, 1.0))
                points.append(
                    (
                        rect.x + 5 + int(fraction_x * (rect.width - 10)),
                        rect.bottom - 5 - int(fraction_y * (rect.height - 10)),
                    )
                )
            if len(points) >= 2:
                pygame.draw.lines(self.screen, color, False, points, 2)

    def _cycle_overlay(self) -> None:
        current = self.overlay_modes.index(self.selection.overlay)
        overlay = self.overlay_modes[(current + 1) % len(self.overlay_modes)]
        self.selection = VisualSelection(
            overlay=overlay,
            include_events=self.selection.include_events,
            include_glyphs=self.selection.include_glyphs,
            include_debug=self.selection.include_debug,
            include_effects=self.selection.include_effects,
            selected_ow_id=self.selected_ow_id,
            fields=self.selection.fields,
        )
        self.status_message = f"Overlay: {overlay}"

    def _activate_button(self, name: str) -> None:
        actions = {
            "first": lambda: self.timeline.seek_index(0),
            "back10": lambda: self.timeline.step(-10),
            "back": lambda: self.timeline.step(-1),
            "next": lambda: self.timeline.step(1),
            "next10": lambda: self.timeline.step(10),
            "last": lambda: self.timeline.seek_index(len(self.timeline.ticks) - 1),
        }
        if name in actions:
            self._set_tick(actions[name]())
        elif name == "play":
            self.timeline.playing = not self.timeline.playing
        elif name == "reverse":
            self.timeline.direction *= -1
        elif name == "speed":
            self.timeline.cycle_speed(1)
        elif name == "loop_a":
            self.timeline.loop_start = self.timeline.index
            if (
                self.timeline.loop_end is not None
                and self.timeline.loop_end < self.timeline.loop_start
            ):
                self.timeline.loop_end = self.timeline.loop_start
        elif name == "loop_b":
            self.timeline.loop_end = self.timeline.index
            if (
                self.timeline.loop_start is not None
                and self.timeline.loop_start > self.timeline.loop_end
            ):
                self.timeline.loop_start = self.timeline.loop_end
        elif name == "loop_clear":
            self.timeline.loop_start = None
            self.timeline.loop_end = None
        elif name == "fit":
            fit_world(self.camera)
        self._build_buttons()

    def _handle_key(self, event: Any) -> None:
        pygame = self.pygame
        if self.search_active:
            if event.key == pygame.K_RETURN:
                self._search_submit()
            elif event.key == pygame.K_ESCAPE:
                self.search_active = False
            elif event.key == pygame.K_BACKSPACE:
                self.search_text = self.search_text[:-1]
            return
        modifiers = pygame.key.get_mods()
        if event.key == pygame.K_SPACE:
            self.timeline.playing = not self.timeline.playing
        elif event.key == pygame.K_LEFT:
            jump = (
                100 if modifiers & pygame.KMOD_CTRL else 10 if modifiers & pygame.KMOD_SHIFT else 1
            )
            self._set_tick(self.timeline.step(-jump))
        elif event.key == pygame.K_RIGHT:
            jump = (
                100 if modifiers & pygame.KMOD_CTRL else 10 if modifiers & pygame.KMOD_SHIFT else 1
            )
            self._set_tick(self.timeline.step(jump))
        elif event.key == pygame.K_HOME:
            self._set_tick(self.timeline.seek_index(0))
        elif event.key == pygame.K_END:
            self._set_tick(self.timeline.seek_index(len(self.timeline.ticks) - 1))
        elif event.key in (pygame.K_LEFTBRACKET, pygame.K_MINUS):
            self.timeline.cycle_speed(-1)
            self._build_buttons()
        elif event.key in (pygame.K_RIGHTBRACKET, pygame.K_EQUALS, pygame.K_PLUS):
            self.timeline.cycle_speed(1)
            self._build_buttons()
        elif event.key == pygame.K_r:
            self.timeline.direction *= -1
        elif event.key == pygame.K_j:
            self._jump_event(-1)
        elif event.key == pygame.K_k:
            self._jump_event(1)
        elif event.key == pygame.K_i:
            self.timeline.loop_start = self.timeline.index
        elif event.key == pygame.K_o:
            self.timeline.loop_end = self.timeline.index
        elif event.key == pygame.K_l:
            self.timeline.loop_start = None
            self.timeline.loop_end = None
        elif event.key == pygame.K_f:
            self.follow_selected = not self.follow_selected
            self.camera.follow_ow_id = self.selected_ow_id if self.follow_selected else None
        elif event.key == pygame.K_p:
            self._select_parent()
        elif event.key == pygame.K_c:
            self._select_child()
        elif event.key == pygame.K_n:
            self._cycle_lineage_member()
        elif event.key == pygame.K_b:
            self._save_bookmark()
        elif event.key == pygame.K_h:
            self._cycle_overlay()
        elif event.key == pygame.K_0:
            fit_world(self.camera)
        elif event.key == pygame.K_SLASH:
            self.search_active = True
            self.search_text = ""
        elif event.key == pygame.K_s:
            target = self.output_dir / f"screenshot_tick_{self.current.tick:08d}.png"
            pygame.image.save(self.screen, target)
            self.status_message = f"Screenshot saved outside bundle: {target.name}"
        elif event.key == pygame.K_e and self.selected_ow_id is not None:
            target = self.output_dir / f"ow_{self.selected_ow_id}_state.csv"
            self.source.export_selection_csv(
                str(target),
                ow_id=self.selected_ow_id,
                start_tick=self.timeline.ticks[0],
                end_tick=self.timeline.ticks[-1],
            )
            math_target = self.output_dir / f"ow_{self.selected_ow_id}_action_math.csv"
            self.source.export_action_math_csv(
                str(math_target),
                ow_id=self.selected_ow_id,
                start_tick=self.timeline.ticks[0],
                end_tick=self.timeline.ticks[-1],
            )
            self.status_message = f"Exported OW data to {self.output_dir}"
        elif event.key == pygame.K_ESCAPE:
            self.running = False

    def handle_events(self) -> None:
        pygame = self.pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.VIDEORESIZE:
                self.screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
                self._layout(*event.size)
            elif event.type == pygame.TEXTINPUT and self.search_active:
                self.search_text += event.text
            elif event.type == pygame.KEYDOWN:
                self._handle_key(event)
            elif event.type == pygame.MOUSEWHEEL:
                mouse = pygame.mouse.get_pos()
                if self.sidebar_rect.collidepoint(mouse) and not self.minimap_rect.collidepoint(
                    mouse
                ):
                    self.inspector_scroll = max(0, self.inspector_scroll - int(event.y) * 42)
                else:
                    zoom_at(self.camera, 1.16 if event.y > 0 else 1 / 1.16, mouse)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    clicked = next(
                        (
                            name
                            for name, rect in self.button_rects.items()
                            if rect.collidepoint(event.pos)
                        ),
                        None,
                    )
                    if clicked is not None:
                        self._activate_button(clicked)
                    elif self.minimap_rect.collidepoint(event.pos):
                        self._center_from_minimap(event.pos)
                    elif self.timeline_rect.collidepoint(event.pos):
                        self.dragging_timeline = True
                        self._timeline_seek(event.pos[0])
                    elif event.pos[0] < self.camera.viewport[2]:
                        self._select_at(event.pos)
                elif event.button in (2, 3) and event.pos[0] < self.camera.viewport[2]:
                    self.dragging_world = True
                    self.last_mouse = event.pos
                elif event.button == 4:
                    zoom_at(self.camera, 1.16, event.pos)
                elif event.button == 5:
                    zoom_at(self.camera, 1 / 1.16, event.pos)
            elif event.type == pygame.MOUSEBUTTONUP:
                self.dragging_world = False
                self.dragging_timeline = False
            elif event.type == pygame.MOUSEMOTION:
                if self.dragging_world:
                    dx = event.pos[0] - self.last_mouse[0]
                    dy = event.pos[1] - self.last_mouse[1]
                    pan(self.camera, dx, dy)
                    self.last_mouse = event.pos
                if self.dragging_timeline:
                    self._timeline_seek(event.pos[0])
                for name, rect in self.button_rects.items():
                    if rect.collidepoint(event.pos):
                        self.status_message = self.button_tooltips[name]
                        break

    def _draw_text(
        self,
        text: str,
        position: tuple[int, int],
        *,
        color: Any = None,
        font: Any = None,
    ) -> int:
        surface, rect = (font or self.font).render(text, color or self.theme.text)
        self.screen.blit(surface, position)
        return int(rect.height)

    def _draw_sidebar(self) -> None:
        pygame = self.pygame
        pygame.draw.rect(self.screen, self.theme.panel, self.sidebar_rect)
        pygame.draw.rect(self.screen, self.theme.panel_border, self.sidebar_rect, 1)
        previous_clip = self.screen.get_clip()
        self.screen.set_clip(self.inspector_rect)
        x = self.sidebar_rect.x + 16
        y = 14 - self.inspector_scroll
        y += self._draw_text("OWL / RAQIC Replay", (x, y), font=self.title_font) + 6
        manifest = self.source.manifest
        loop_text = (
            "none"
            if self.timeline.loop_start is None and self.timeline.loop_end is None
            else f"{self.timeline.loop_start}:{self.timeline.loop_end}"
        )
        for line in (
            f"Run: {manifest.run_id}",
            f"Condition: {manifest.condition}",
            f"Tick: {self.current.tick} / {self.timeline.ticks[-1]}",
            f"Speed: {'-' if self.timeline.direction < 0 else ''}{self.timeline.speed:g}x",
            f"State: {'PLAYING' if self.timeline.playing else 'PAUSED'}",
            f"Loop indexes: {loop_text}",
            f"Overlay: {self.selection.overlay}",
            f"Verification: {self.source.verification_status}",
        ):
            y += self._draw_text(line, (x, y), font=self.small_font) + 3
        y += 8
        if self.search_active:
            y += self._draw_text(f"Find OW ID: {self.search_text}_", (x, y)) + 8
        else:
            y += self._draw_text("Press / to search stable OW ID", (x, y), font=self.small_font) + 8
        if self.selected_ow_id is None:
            y += self._draw_text("Click an OW to inspect it", (x, y))
        else:
            details = self.source.load_ow_details(self.current.tick, self.selected_ow_id)
            y += (
                self._draw_text(f"Selected OW {self.selected_ow_id}", (x, y), font=self.title_font)
                + 5
            )
            if details is None:
                y += self._draw_text("Not living/present at this tick", (x, y))
            else:
                values = details.values
                keys = (
                    "y",
                    "x",
                    "lineage_id",
                    "parent_id",
                    "ow_type",
                    "age",
                    "development_stage",
                    "health",
                    "resource",
                    "toxin",
                    "integration",
                    "starvation_debt",
                    "raqic_readout",
                    "raqic_record_confidence",
                    "raqic_utility_innovation_norm",
                    "raqic_utility_projection_fraction",
                    "raqic_utility_score_cosine",
                    "raqic_utility_orthogonality_residual",
                    "raqic_policy_kl",
                    "raqic_interference_delta_l1",
                    "raqic_interference_norm_error",
                    "raqic_interference_illegal_mass",
                )
                for key in keys:
                    if key in values:
                        y += (
                            self._draw_text(
                                f"{key}: {values[key]}"[:58], (x, y), font=self.small_font
                            )
                            + 2
                        )
                y += 5
                history_rect = pygame.Rect(x, y, self.inspector_rect.width - 32, 76)
                self._draw_history_plot(history_rect)
                y += history_rect.height + 6
                y += (
                    self._draw_text(
                        "History: health / resource / integration", (x, y), font=self.small_font
                    )
                    + 5
                )
                y += self._draw_text("Action mathematics (all actions)", (x, y)) + 3
                headers = "Sel Id Action            Legal   Utility  P(final) Phase"
                y += self._draw_text(headers, (x, y), font=self.small_font) + 2
                for row in details.action_math:
                    probability = row.get(
                        "raqic_probabilities", row.get("last_action_probabilities", "—")
                    )
                    utility = row.get("last_utilities", row.get("pre_utilities", "—"))
                    phase = row.get("raqic_phase", "—")
                    legal = row.get("_authority_bool", row.get("authority", "—"))
                    selected = ">" if row.get("selected") else " "
                    text = (
                        f"{selected} {row['action_index']:02d} {row['action_name'][:15]:15s} "
                        f"{str(legal)[:5]:5s} {str(utility)[:7]:7s} "
                        f"{str(probability)[:7]:7s} {str(phase)[:7]:7s}"
                    )
                    y += self._draw_text(text, (x, y), font=self.small_font) + 1
                if details.recent_events:
                    y += 8
                    y += self._draw_text("Recent events", (x, y)) + 3
                    for event in details.recent_events[-8:]:
                        y += self._draw_text(str(event)[:58], (x, y), font=self.small_font) + 1
        self.screen.set_clip(previous_clip)
        self._draw_minimap()
        self._draw_text(
            self.status_message[:66],
            (self.sidebar_rect.x + 16, self.minimap_rect.y - 23),
            color=(180, 220, 255, 255),
            font=self.small_font,
        )

    def _draw_minimap(self) -> None:
        pygame = self.pygame
        pygame.draw.rect(self.screen, (5, 9, 16, 255), self.minimap_rect)
        pygame.draw.rect(self.screen, self.theme.panel_border, self.minimap_rect, 1)
        health = np.asarray(self.current.arrays["health"])
        occupancy = np.asarray(self.current.arrays.get("occupancy", np.full(health.shape, -1)))
        coords = np.argwhere((health > 0) & (occupancy >= 0))
        h, w = self.current.world_shape
        for y, x in coords[:: max(1, len(coords) // 1500)]:
            px = self.minimap_rect.x + int((x + 0.5) / w * self.minimap_rect.width)
            py = self.minimap_rect.y + int((y + 0.5) / h * self.minimap_rect.height)
            if self.minimap_rect.collidepoint(px, py):
                self.screen.set_at((px, py), self.theme.health_good)
        rect = minimap_viewport_rect(
            self.camera,
            (
                self.minimap_rect.x,
                self.minimap_rect.y,
                self.minimap_rect.width,
                self.minimap_rect.height,
            ),
        )
        pygame.draw.rect(self.screen, self.theme.selected, rect, 2)

    def _draw_timeline(self) -> None:
        pygame = self.pygame
        pygame.draw.rect(self.screen, self.theme.panel, self.timeline_rect)
        pygame.draw.rect(self.screen, self.theme.panel_border, self.timeline_rect, 1)
        for name, rect in self.button_rects.items():
            active = (name == "play" and self.timeline.playing) or (
                name == "reverse" and self.timeline.direction < 0
            )
            color = self.theme.selected if active else self.theme.panel_border
            pygame.draw.rect(self.screen, self.theme.panel, rect, border_radius=4)
            pygame.draw.rect(self.screen, color, rect, 1, border_radius=4)
            labels = {
                "first": "|<",
                "back10": "<<",
                "back": "<",
                "play": "Pause" if self.timeline.playing else "Play",
                "next": ">",
                "next10": ">>",
                "last": ">|",
                "reverse": "Reverse",
                "speed": f"{self.timeline.speed:g}x",
                "loop_a": "Loop A",
                "loop_b": "Loop B",
                "loop_clear": "Clear",
                "fit": "Fit",
            }
            label = labels[name]
            self._draw_text(label, (rect.x + 7, rect.y + 5), font=self.small_font)
        left = self.timeline_rect.x + 20
        right = self.timeline_rect.right - 20
        y = self.timeline_rect.y + 74
        pygame.draw.line(self.screen, self.theme.panel_border, (left, y), (right, y), 5)
        fraction = self.timeline.index / max(1, len(self.timeline.ticks) - 1)
        x = int(left + fraction * (right - left))
        pygame.draw.line(self.screen, self.theme.resource, (left, y), (x, y), 5)
        if self.timeline.loop_start is not None:
            lx = int(
                left
                + self.timeline.loop_start / max(1, len(self.timeline.ticks) - 1) * (right - left)
            )
            pygame.draw.line(self.screen, self.theme.selected, (lx, y - 10), (lx, y + 10), 2)
        if self.timeline.loop_end is not None:
            lx = int(
                left
                + self.timeline.loop_end / max(1, len(self.timeline.ticks) - 1) * (right - left)
            )
            pygame.draw.line(self.screen, self.theme.selected, (lx, y - 10), (lx, y + 10), 2)
        pygame.draw.circle(self.screen, self.theme.selected, (x, y), 9)
        controls = (
            "Keys: Space play  arrows step  Shift/Ctrl jump  J/K events  I/O/L loop  "
            "[/] speed  R reverse  H overlay  wheel zoom  middle/right pan  F follow  / search  "
            "P parent  C child  N lineage  B bookmark"
        )
        self._draw_text(controls, (left, self.timeline_rect.y + 98), font=self.small_font)
        self._draw_text(
            f"Tick {self.current.tick} | {self.timeline.speed:g}x | speeds {PLAYBACK_SPEEDS} | "
            f"cache {self.source._cache.metrics.hits}/{self.source._cache.metrics.misses}",
            (max(left, self.button_rects["fit"].right + 18), self.timeline_rect.y + 18),
            font=self.small_font,
        )

    def render(self, elapsed: float) -> None:
        if self.follow_selected:
            update_follow_camera(self.camera, self.current, elapsed)
        adjacent = (
            self.previous is not None
            and abs(int(self.current.tick) - int(self.previous.tick)) == 1
        )
        progress = self.timeline.interpolation_progress() if adjacent else 1.0
        # At the default 60 Hz display/10 Hz replay rates, render six visual
        # subframes per authoritative scientific snapshot. This metadata and
        # interpolation are read-only and cannot feed back into the replay.
        display_subframes = max(1, int(round(60.0 / self.timeline.tick_rate)))
        subframe_index = min(
            display_subframes - 1,
            max(0, int(progress * display_subframes)),
        )
        scene = build_visual_scene(
            self.previous,
            self.current,
            progress,
            self.camera,
            self.selection,
            self.current.events,
            theme=self.theme,
            subframe_index=subframe_index,
            subframe_count=display_subframes,
            visual_seed=self.source.manifest.seed,
            trait_color_mode="perceptual",
            accessibility_mode="standard",
        )
        self.renderer.render(scene, self.screen)
        self._schedule_prefetch()
        self._draw_sidebar()
        self._draw_timeline()
        self.pygame.display.flip()

    def run(self, *, max_frames: int | None = None) -> int:
        frames = 0
        while self.running:
            elapsed = self.clock.tick(60) / 1000.0
            self.handle_events()
            tick = self.timeline.update(elapsed)
            self._set_tick(tick)
            self.render(elapsed)
            frames += 1
            if max_frames is not None and frames >= max_frames:
                break
        self._prefetch_executor.shutdown(wait=False, cancel_futures=True)
        self.pygame.quit()
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", help="Path to a completed owl.replay.v1 bundle")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--theme", default="owl_dark_neon")
    parser.add_argument("--output-dir")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-frames", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    app = ReplayApplication(
        args.bundle,
        window_size=(args.width, args.height),
        theme_name=args.theme,
        headless=args.headless,
        output_dir=args.output_dir,
    )
    return app.run(max_frames=args.max_frames)


if __name__ == "__main__":
    raise SystemExit(main())
