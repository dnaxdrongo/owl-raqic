"""Optional NVTX instrumentation with a zero-cost CPU fallback."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def nvtx_range(name: str) -> Iterator[None]:
    try:
        import nvtx
    except ImportError:
        yield
        return
    with nvtx.annotate(str(name)):
        yield
