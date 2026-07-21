from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class CacheMetrics:
    hits: int = 0
    misses: int = 0
    evictions: int = 0


class LRUCache(Generic[T]):
    def __init__(self, max_entries: int = 8) -> None:
        self.max_entries = max(1, int(max_entries))
        self._items: OrderedDict[int, T] = OrderedDict()
        self.metrics = CacheMetrics()
        self._lock = RLock()

    def get(self, key: int) -> T | None:
        with self._lock:
            value = self._items.get(int(key))
            if value is None:
                self.metrics.misses += 1
                return None
            self._items.move_to_end(int(key))
            self.metrics.hits += 1
            return value

    def put(self, key: int, value: T) -> T:
        with self._lock:
            self._items[int(key)] = value
            self._items.move_to_end(int(key))
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)
                self.metrics.evictions += 1
            return value

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
