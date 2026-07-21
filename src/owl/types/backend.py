"""Structural contracts for array backends, streams, events, and memory pools."""

from __future__ import annotations

from typing import Any, Protocol

from .arrays import ArrayAny, DeviceArray


class StreamLike(Protocol):
    ptr: int

    def synchronize(self) -> None: ...


class EventLike(Protocol):
    def record(self, stream: StreamLike | None = None) -> None: ...
    def synchronize(self) -> None: ...


class MemoryPoolLike(Protocol):
    def used_bytes(self) -> int: ...
    def total_bytes(self) -> int: ...


class ArrayBackendProtocol(Protocol):
    name: str
    xp: Any
    is_gpu: bool

    def asarray(self, value: Any, dtype: Any | None = None) -> DeviceArray[Any]: ...
    def asnumpy(self, value: DeviceArray[Any] | Any) -> ArrayAny: ...
    def synchronize(self) -> None: ...
