"""Named experiment preset lookup."""

from __future__ import annotations

from collections.abc import Callable

from owl.core.config import SimulationConfig
from owl.experiments.conditions import (
    make_baseline_condition,
    make_carnivore_condition,
    make_fragmented_condition,
    make_integrated_condition,
    make_overcoupled_condition,
    make_rivalry_condition,
)

_CONDITIONS: dict[str, Callable[[SimulationConfig], SimulationConfig]] = {
    "baseline": make_baseline_condition,
    "integrated": make_integrated_condition,
    "rivalry": make_rivalry_condition,
    "fragmented": make_fragmented_condition,
    "overcoupled": make_overcoupled_condition,
    "carnivore": make_carnivore_condition,
}


def list_conditions() -> list[str]:
    """Return available experiment condition names in stable order."""
    return sorted(_CONDITIONS)


def get_condition(name: str, cfg: SimulationConfig) -> SimulationConfig:
    """Return ``cfg`` transformed by a named experiment condition.

    Parameters
    ----------
    name:
        Condition name. Matching is case-insensitive and treats hyphens as
        underscores.
    cfg:
        Base simulation configuration. The input object is not mutated.

    Returns
    -------
    SimulationConfig
        Condition-specific deep copy of ``cfg``.
    """
    key = str(name).strip().lower().replace("-", "_")
    if key not in _CONDITIONS:
        raise ValueError(
            f"unknown condition {name!r}; available conditions: {', '.join(list_conditions())}"
        )
    return _CONDITIONS[key](cfg)
