from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


def template_key(
    *,
    mode: str,
    action_count: int,
    active_primes: Any = (),
    mask_pattern: Any | None = None,
    layout: Any | None = None,
) -> str:
    payload = {
        "mode": str(mode),
        "action_count": int(action_count),
        "active_primes": tuple(int(p) for p in active_primes),
        "mask_pattern": None if mask_pattern is None else tuple(bool(x) for x in mask_pattern),
        "layout": layout,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


@dataclass
class CircuitTemplateCache:
    max_entries: int = 256
    _items: dict[str, Any] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    def get_or_build(self, key: str, builder: Callable[[], Any]) -> Any:
        if key in self._items:
            self.hits += 1
            return self._items[key]
        self.misses += 1
        value = builder()
        if len(self._items) >= self.max_entries:
            self._items.pop(next(iter(self._items)))
        self._items[key] = value
        return value

    def clear(self) -> None:
        self._items.clear()

    def stats(self) -> dict[str, int]:
        return {"entries": len(self._items), "hits": self.hits, "misses": self.misses}
