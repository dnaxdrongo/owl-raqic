"""Declare behavioral coverage for public configuration fields.

Each covered field is tied to an exact runtime attribute access and a named
test. The strict audit rejects metadata-only or comment-only references as
evidence of runtime use.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

COVERAGE_ATTRIBUTE = "__owl_config_fields__"


def covers_config_field(*names: str) -> Any:
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("covers_config_field requires nonempty field names")

    def decorate(func: Callable[..., Any]) -> Any:
        existing = tuple(getattr(func, COVERAGE_ATTRIBUTE, ()))
        setattr(func, COVERAGE_ATTRIBUTE, tuple(dict.fromkeys(existing + tuple(names))))
        return func

    return decorate


def declared_fields(func: Callable[..., Any]) -> tuple[str, ...]:
    return tuple(getattr(func, COVERAGE_ATTRIBUTE, ()))
