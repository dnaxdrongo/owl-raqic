from __future__ import annotations

from dataclasses import dataclass

PLAYBACK_SPEEDS = (0.1, 0.25, 0.5, 0.75, 1.0, 1.2, 1.5, 2.0, 4.0, 8.0)


@dataclass
class PlaybackClock:
    ticks: tuple[int, ...]
    tick_rate: float = 10.0
    index: int = 0
    speed: float = 1.0
    playing: bool = False
    direction: int = 1
    loop_start: int | None = None
    loop_end: int | None = None
    accumulator: float = 0.0

    @property
    def current_tick(self) -> int:
        if not self.ticks:
            raise IndexError("timeline contains no ticks")
        return int(self.ticks[self.index])

    def seek_index(self, index: int) -> int:
        self.index = max(0, min(len(self.ticks) - 1, int(index)))
        self.accumulator = 0.0
        return self.current_tick

    def seek_tick(self, tick: int) -> int:
        if not self.ticks:
            raise IndexError("timeline contains no ticks")
        nearest = min(range(len(self.ticks)), key=lambda i: abs(self.ticks[i] - int(tick)))
        return self.seek_index(nearest)

    def step(self, amount: int = 1) -> int:
        return self.seek_index(self.index + int(amount))

    def set_speed(self, speed: float) -> None:
        value = float(speed)
        if value not in PLAYBACK_SPEEDS:
            raise ValueError(f"unsupported playback speed: {value}")
        self.speed = value

    def cycle_speed(self, direction: int) -> float:
        current = min(
            range(len(PLAYBACK_SPEEDS)), key=lambda i: abs(PLAYBACK_SPEEDS[i] - self.speed)
        )
        current = max(0, min(len(PLAYBACK_SPEEDS) - 1, current + int(direction)))
        self.speed = PLAYBACK_SPEEDS[current]
        return self.speed

    def update(self, elapsed_seconds: float) -> int:
        if not self.playing or not self.ticks:
            return self.current_tick
        self.accumulator += max(0.0, float(elapsed_seconds)) * self.tick_rate * self.speed
        whole = int(self.accumulator)
        if whole <= 0:
            return self.current_tick
        self.accumulator -= whole
        for _ in range(whole):
            next_index = self.index + self.direction
            lower = 0 if self.loop_start is None else self.loop_start
            upper = len(self.ticks) - 1 if self.loop_end is None else self.loop_end
            if next_index > upper:
                next_index = lower if self.loop_start is not None else upper
                if self.loop_start is None:
                    self.playing = False
            elif next_index < lower:
                next_index = upper if self.loop_end is not None else lower
                if self.loop_end is None:
                    self.playing = False
            self.index = next_index
            if not self.playing:
                break
        return self.current_tick

    def interpolation_progress(self) -> float:
        """Return presentation-only progress between scientific ticks."""

        if not self.playing:
            return 1.0
        return max(0.0, min(1.0, float(self.accumulator)))
