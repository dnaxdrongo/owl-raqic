from __future__ import annotations

from owl_raqic.types import DEFAULT_ACTIONS


def expected_action_names() -> tuple[str, ...]:
    return DEFAULT_ACTIONS


def action_index(name: str) -> int:
    return DEFAULT_ACTIONS.index(name)
