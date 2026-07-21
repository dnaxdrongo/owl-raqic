"""Bounded recording telemetry and adaptive batch policy.

The policy changes only future batch boundaries. It never changes schema, row
order, field selection, compression, or scientific values.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class AdaptiveBatchTelemetry:
    observations: int = 0
    rows: int = 0
    arrow_bytes: int = 0
    elapsed_seconds: float = 0.0
    ewma_bytes_per_row: float = 0.0
    ewma_rows_per_second: float = 0.0
    current_row_limit: int = 0
    target_batch_bytes: int = 0

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class AdaptiveBatchPolicy:
    """Choose a bounded action-aligned row limit from measured batch costs."""

    def __init__(
        self,
        *,
        action_count: int,
        initial_rows: int,
        min_rows: int,
        max_rows: int,
        target_batch_bytes: int,
        alpha: float = 0.25,
        max_step_ratio: float = 1.5,
    ) -> None:
        if action_count <= 0:
            raise ValueError("action_count must be positive")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if max_step_ratio < 1.0:
            raise ValueError("max_step_ratio must be at least 1")
        self.action_count = int(action_count)
        self.min_rows = self._aligned(max(int(min_rows), self.action_count))
        self.max_rows = self._aligned(max(int(max_rows), self.min_rows))
        self.target_batch_bytes = max(1, int(target_batch_bytes))
        self.alpha = float(alpha)
        self.max_step_ratio = float(max_step_ratio)
        self.current_rows = self._bounded(initial_rows)
        self.telemetry = AdaptiveBatchTelemetry(
            current_row_limit=self.current_rows,
            target_batch_bytes=self.target_batch_bytes,
        )

    def _aligned(self, rows: int) -> int:
        aligned = int(rows) - (int(rows) % self.action_count)
        return max(self.action_count, aligned)

    def _bounded(self, rows: int) -> int:
        return min(self.max_rows, max(self.min_rows, self._aligned(int(rows))))

    def observe(
        self,
        *,
        rows: int,
        arrow_bytes: int,
        elapsed_seconds: float,
        host_headroom_bytes: int | None = None,
    ) -> int:
        """Update EWMA estimates and return the next aligned row limit."""

        rows = max(1, int(rows))
        arrow_bytes = max(0, int(arrow_bytes))
        elapsed_seconds = max(float(elapsed_seconds), 1.0e-9)
        bytes_per_row = arrow_bytes / rows
        rows_per_second = rows / elapsed_seconds
        telemetry = self.telemetry
        if telemetry.observations == 0:
            telemetry.ewma_bytes_per_row = bytes_per_row
            telemetry.ewma_rows_per_second = rows_per_second
        else:
            a = self.alpha
            telemetry.ewma_bytes_per_row = (
                a * bytes_per_row + (1.0 - a) * telemetry.ewma_bytes_per_row
            )
            telemetry.ewma_rows_per_second = (
                a * rows_per_second + (1.0 - a) * telemetry.ewma_rows_per_second
            )
        telemetry.observations += 1
        telemetry.rows += rows
        telemetry.arrow_bytes += arrow_bytes
        telemetry.elapsed_seconds += elapsed_seconds

        effective_target = self.target_batch_bytes
        if host_headroom_bytes is not None:
            effective_target = min(effective_target, max(1, int(host_headroom_bytes) // 8))
        estimated_width = max(telemetry.ewma_bytes_per_row, 1.0)
        desired = self._bounded(int(effective_target / estimated_width))
        lower = self._bounded(int(self.current_rows / self.max_step_ratio))
        upper = self._bounded(int(self.current_rows * self.max_step_ratio))
        self.current_rows = self._bounded(min(upper, max(lower, desired)))
        telemetry.current_row_limit = self.current_rows
        telemetry.target_batch_bytes = effective_target
        return self.current_rows
