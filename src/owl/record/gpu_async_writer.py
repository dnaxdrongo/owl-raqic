from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Thread
from typing import Any


@dataclass
class AsyncGPUWriter:
    """Bounded asynchronous JSONL writer used by persistent GPU runs.

    ``block`` provides lossless backpressure. ``raise`` fails closed when the
    producer outruns the configured queue.  The counters are operational
    evidence consumed by  certification; they are not simulation state.
    """

    path: str | Path
    flush_interval: float = 0.5
    max_queue: int = 1024
    overflow_policy: str = "block"
    write_delay_seconds: float = 0.0
    _queue: Queue[Any] = field(init=False)
    _thread: Thread | None = field(default=None, init=False)
    _closed: bool = field(default=False, init=False)
    records_written: int = field(default=0, init=False)
    overflow_count: int = field(default=0, init=False)
    queue_peak: int = field(default=0, init=False)
    closed_cleanly: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if int(self.max_queue) < 1:
            raise ValueError("max_queue must be positive")
        if self.overflow_policy not in {"block", "raise"}:
            raise ValueError("overflow_policy must be 'block' or 'raise'")
        if float(self.write_delay_seconds) < 0:
            raise ValueError("write_delay_seconds must be nonnegative")
        self._queue = Queue(maxsize=int(self.max_queue))

    def start(self) -> AsyncGPUWriter:
        if self._thread is not None:
            raise RuntimeError("AsyncGPUWriter already started")
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._thread = Thread(target=self._run, daemon=True, name="owl-gpu-writer")
        self._thread.start()
        return self

    def _run(self) -> None:
        with Path(self.path).open("a", encoding="utf-8") as handle:
            while not self._closed or not self._queue.empty():
                try:
                    item = self._queue.get(timeout=self.flush_interval)
                except Empty:
                    handle.flush()
                    continue
                try:
                    if self.write_delay_seconds:
                        time.sleep(float(self.write_delay_seconds))
                    handle.write(json.dumps(item, sort_keys=True, default=str) + "\n")
                    self.records_written += 1
                finally:
                    self._queue.task_done()
            handle.flush()

    def _record_peak(self) -> None:
        self.queue_peak = max(self.queue_peak, int(self._queue.qsize()))

    def write(self, item: dict[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("AsyncGPUWriter is closed")
        payload = dict(item)
        if self.overflow_policy == "block":
            self._queue.put(payload)
            self._record_peak()
            return
        try:
            self._queue.put_nowait(payload)
            self._record_peak()
        except Full as exc:
            self.overflow_count += 1
            raise RuntimeError(f"AsyncGPUWriter queue is full ({self.max_queue})") from exc

    def close(self) -> None:
        if self._closed and self.closed_cleanly:
            return
        self._closed = True
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            if self._thread.is_alive():
                raise RuntimeError("AsyncGPUWriter did not drain within 10 seconds")
        self.closed_cleanly = True

    def summary(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "overflow_policy": self.overflow_policy,
            "max_queue": int(self.max_queue),
            "queue_peak": int(self.queue_peak),
            "records_written": int(self.records_written),
            "overflow_count": int(self.overflow_count),
            "closed_cleanly": bool(self.closed_cleanly),
        }
