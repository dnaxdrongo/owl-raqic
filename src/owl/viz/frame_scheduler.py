from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class VisualScheduleMode(StrEnum):
    LIVE_ADAPTIVE = "live_adaptive"
    RECORD_FIXED = "record_fixed"
    RECORD_KEYFRAMES = "record_keyframes"


@dataclass(frozen=True)
class FrameRequest:
    scientific_tick: int
    subframe_index: int
    subframe_count: int
    progress: float


@dataclass
class VisualFrameScheduler:
    mode: VisualScheduleMode
    frames_per_tick: int
    tick_stride: int
    render_every: int = 1
    fail_on_drop: bool = True

    def requests_for_tick(self, tick: int) -> tuple[FrameRequest, ...]:
        tick_value = int(tick)
        if tick_value % max(1, int(self.tick_stride)) != 0:
            return ()
        if self.mode == VisualScheduleMode.LIVE_ADAPTIVE:
            if tick_value % max(1, int(self.render_every)) != 0:
                return ()
            count = 1
        elif self.mode == VisualScheduleMode.RECORD_KEYFRAMES:
            count = 1
        else:
            count = max(1, int(self.frames_per_tick))
        return tuple(
            FrameRequest(
                scientific_tick=tick_value,
                subframe_index=index,
                subframe_count=count,
                progress=(index + 1) / count,
            )
            for index in range(count)
        )

    def observe_live_cost(self, simulation_ms: float, render_ms: float) -> None:
        if self.mode != VisualScheduleMode.LIVE_ADAPTIVE:
            return
        total = max(float(simulation_ms) + float(render_ms), 1e-12)
        fraction = float(render_ms) / total
        if fraction > 0.35:
            self.render_every = min(128, max(2, self.render_every * 2))
        elif fraction < 0.12 and self.render_every > 1:
            self.render_every = max(1, self.render_every // 2)


def fixed_frame_plan(
    ticks: int,
    tick_stride: int,
    frames_per_tick: int,
) -> tuple[FrameRequest, ...]:
    scheduler = VisualFrameScheduler(
        VisualScheduleMode.RECORD_FIXED,
        frames_per_tick=max(1, int(frames_per_tick)),
        tick_stride=max(1, int(tick_stride)),
    )
    requests: list[FrameRequest] = []
    for tick in range(1, int(ticks) + 1):
        requests.extend(scheduler.requests_for_tick(tick))
    return tuple(requests)
