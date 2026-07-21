from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


class NullEvent:
    def record(self, stream: Any | None = None) -> None:
        return None

    def synchronize(self) -> None:
        return None

    def query(self) -> bool:
        return True


class NullStream:
    def __enter__(self) -> Any:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        return False

    def synchronize(self) -> None:
        return None

    def use(self) -> Any:
        return self

    def wait_event(self, event: Any) -> None:
        return None


@dataclass
class StreamBundle:
    """Compute/transfer/validation/visual streams with explicit dependencies."""

    backend: Any
    compute: Any = field(default_factory=NullStream)
    transfer: Any = field(default_factory=NullStream)
    validation: Any = field(default_factory=NullStream)
    visual: Any = field(default_factory=NullStream)
    pinned_supported: bool = False

    @classmethod
    def create(cls, backend: Any) -> StreamBundle:
        if getattr(backend, "is_gpu", False):
            cp = backend.xp
            return cls(
                backend=backend,
                compute=cp.cuda.Stream(non_blocking=True),
                transfer=cp.cuda.Stream(non_blocking=True),
                validation=cp.cuda.Stream(non_blocking=True),
                visual=cp.cuda.Stream(non_blocking=True),
                pinned_supported=True,
            )
        return cls(backend=backend)

    def new_event(self) -> Any:
        if getattr(self.backend, "is_gpu", False):
            return self.backend.xp.cuda.Event()
        return NullEvent()

    def record(self, stream: Any | None = None) -> Any:
        event = self.new_event()
        event.record(stream or self.compute)
        return event

    def wait(self, stream: Any, event: Any) -> None:
        if hasattr(stream, "wait_event"):
            stream.wait_event(event)
        else:
            event.synchronize()

    def pinned_array(self, shape: Any, dtype: Any = np.float64) -> Any:
        """Allocate NumPy array backed by page-locked memory on CUDA hosts."""
        dtype = np.dtype(dtype)
        count = int(np.prod(shape, dtype=np.int64))
        if not self.pinned_supported:
            return np.empty(shape, dtype=dtype), None
        cp = self.backend.xp
        mem = cp.cuda.alloc_pinned_memory(count * dtype.itemsize)
        arr = np.frombuffer(mem, dtype=dtype, count=count).reshape(shape)
        return arr, mem  # keep memory owner alive

    def synchronize_all(self) -> None:
        for stream in (self.compute, self.transfer, self.validation, self.visual):
            stream.synchronize()


@dataclass
class TransferTicket:
    host_array: np.ndarray
    event: Any
    owner: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def ready(self) -> bool:
        try:
            return bool(self.event.query())
        except Exception:
            return False

    def result(self, *, block: bool = True) -> np.ndarray | None:
        if block:
            self.event.synchronize()
            return self.host_array
        return self.host_array if self.ready() else None


@dataclass
class HostTransferBuffer:
    """Bounded transfer-ticket ring.

    Slots are never overwritten while a prior asynchronous copy is incomplete.
    In non-blocking mode callers may drop a visual frame instead of stalling the
    simulation. Scientific metric/checkpoint transfers should use ``block=True``.
    """

    max_slots: int = 2
    tickets: list[TransferTicket] = field(default_factory=list)

    def reap(self) -> list[TransferTicket]:
        ready: list[TransferTicket] = []
        pending: list[TransferTicket] = []
        for ticket in self.tickets:
            (ready if ticket.ready() else pending).append(ticket)
        self.tickets = pending
        return ready

    def push(self, ticket: TransferTicket, *, block: bool = False) -> bool:
        self.reap()
        if len(self.tickets) >= self.max_slots:
            if not block:
                return False
            oldest = self.tickets.pop(0)
            oldest.result(block=True)
        self.tickets.append(ticket)
        return True

    def flush(self) -> list[np.ndarray]:
        out = []
        for ticket in self.tickets:
            value = ticket.result(block=True)
            if value is not None:
                out.append(value)
        self.tickets.clear()
        return out
