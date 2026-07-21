"""Distributed transport contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .arrays import DeviceArray
from .backend import StreamLike


class CollectiveTransport(Protocol):
    rank: int
    world_size: int

    def send(self, buffer: DeviceArray[Any], peer: int, *, stream: StreamLike) -> None: ...
    def recv(self, buffer: DeviceArray[Any], peer: int, *, stream: StreamLike) -> None: ...
    def allreduce(
        self, send: DeviceArray[Any], recv: DeviceArray[Any], *, op: str, stream: StreamLike
    ) -> None: ...


@dataclass(frozen=True)
class CollectiveSignature:
    phase: str
    field_group: str
    count: int
    dtype: str
    peer: int | None = None
