from __future__ import annotations

from typing import Any

import numpy as np


class VisPyVisualBackend:
    """VisPy adapter for the backend-neutral VisualFrame."""

    def __init__(self, title: str = "OWL + RAQIC") -> None:
        try:
            from vispy import scene
        except Exception as exc:
            raise RuntimeError("VisPy visual backend is unavailable") from exc
        self.scene = scene
        self.canvas = scene.SceneCanvas(keys="interactive", title=title, show=True)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = "panzoom"
        self.image = None
        self.markers = scene.visuals.Markers(parent=self.view.scene)
        self.sprite_markers = scene.visuals.Markers(parent=self.view.scene)
        self.lines = scene.visuals.Line(parent=self.view.scene)
        self.glyphs = scene.visuals.Line(parent=self.view.scene)
        try:
            self.arrows = scene.visuals.Arrow(parent=self.view.scene)
        except Exception:
            self.arrows = None
        self.closed = False

    @staticmethod
    def _sprite_arrays(frame: Any) -> Any:
        if not frame.sprite_states:
            return (
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0, 4), dtype=np.float32),
            )
        pos = np.asarray(frame.sprite_positions, dtype=np.float32)
        face = np.asarray(
            [
                tuple(channel / 255.0 for channel in state.body_color)
                for state in frame.sprite_states
            ],
            dtype=np.float32,
        )
        size = np.asarray(
            [
                4.0 + 4.0 * max(state.health_fraction, state.resource_fraction)
                for state in frame.sprite_states
            ],
            dtype=np.float32,
        )
        edge = np.asarray(
            [
                (1.0, 0.15, 0.15, 1.0)
                if state.cracked_outline or state.hazard_outline
                else (0.7, 0.9, 1.0, max(0.25, state.ring_alpha / 255.0))
                for state in frame.sprite_states
            ],
            dtype=np.float32,
        )
        return pos, face, size, edge

    def submit(self, frame: Any) -> None:
        if self.closed:
            return
        if frame.rgba is not None:
            if self.image is None:
                self.image = self.scene.visuals.Image(frame.rgba, parent=self.view.scene)
            else:
                self.image.set_data(frame.rgba)

        sprite_pos, sprite_face, sprite_size, sprite_edge = self._sprite_arrays(frame)
        if sprite_pos.size:
            self.sprite_markers.set_data(
                pos=sprite_pos,
                face_color=sprite_face,
                edge_color=sprite_edge,
                size=sprite_size,
            )
        else:
            self.sprite_markers.set_data(pos=np.zeros((0, 2), dtype=np.float32))

        if frame.markers.size:
            self.markers.set_data(
                pos=np.asarray(frame.markers, dtype=np.float32),
                face_color=np.asarray(frame.marker_colors, dtype=np.float32),
                size=np.asarray(frame.marker_sizes, dtype=np.float32),
            )
        else:
            self.markers.set_data(pos=np.zeros((0, 2), dtype=np.float32))

        if frame.lines.size:
            self.lines.set_data(
                pos=np.asarray(frame.lines, dtype=np.float32),
                color=np.asarray(frame.line_colors, dtype=np.float32),
                connect="segments",
            )
        else:
            self.lines.set_data(pos=np.zeros((0, 2), dtype=np.float32))

        if frame.glyph_lines.size:
            self.glyphs.set_data(
                pos=np.asarray(frame.glyph_lines, dtype=np.float32),
                color=np.asarray(frame.glyph_line_colors, dtype=np.float32),
                connect="segments",
            )
        else:
            self.glyphs.set_data(pos=np.zeros((0, 2), dtype=np.float32))

        if self.arrows is not None:
            if frame.arrows.size:
                arrow_pos = np.asarray(frame.arrows, dtype=np.float32).reshape(-1, 2)
                try:
                    self.arrows.set_data(
                        pos=arrow_pos,
                        arrows=np.asarray(frame.arrows, dtype=np.float32),
                    )
                except TypeError:
                    self.arrows.set_data(pos=arrow_pos)
            else:
                self.arrows.set_data(pos=np.zeros((0, 2), dtype=np.float32))
        self.canvas.update()

    def close(self) -> None:
        if not self.closed:
            self.canvas.close()
            self.closed = True
