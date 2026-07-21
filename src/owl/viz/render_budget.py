from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RenderBudget:
    target_fps: float = 30.0
    max_slowdown_fraction: float = 0.15
    render_every: int = 1
    glyph_density: float = 1.0
    dropped_frames: int = 0
    mode: str = "live_adaptive"
    decorative_particle_fraction: float = 1.0
    signal_detail_fraction: float = 1.0
    secondary_outline_fraction: float = 1.0
    animation_subframe_fraction: float = 1.0

    def observe(self, *, simulation_ms: float, render_ms: float) -> None:
        if self.mode != "live_adaptive":
            return
        total = max(simulation_ms + render_ms, 1e-12)
        slowdown = render_ms / total
        if slowdown > self.max_slowdown_fraction:
            if self.decorative_particle_fraction > 0.25:
                self.decorative_particle_fraction *= 0.75
            elif self.signal_detail_fraction > 0.25:
                self.signal_detail_fraction *= 0.75
            elif self.secondary_outline_fraction > 0.25:
                self.secondary_outline_fraction *= 0.75
            elif self.animation_subframe_fraction > 0.25:
                self.animation_subframe_fraction *= 0.75
            else:
                self.render_every = min(128, max(2, self.render_every * 2))
        elif slowdown < 0.5 * self.max_slowdown_fraction:
            self.decorative_particle_fraction = min(1.0, self.decorative_particle_fraction * 1.05)
            self.signal_detail_fraction = min(1.0, self.signal_detail_fraction * 1.05)
            self.secondary_outline_fraction = min(1.0, self.secondary_outline_fraction * 1.05)
            self.animation_subframe_fraction = min(1.0, self.animation_subframe_fraction * 1.05)
            if self.render_every > 1:
                self.render_every = max(1, self.render_every // 2)

    def should_render(self, tick: int) -> bool:
        return int(tick) % int(self.render_every) == 0
