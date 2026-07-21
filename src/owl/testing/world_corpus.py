from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from owl.core.advanced import ensure_advanced_fields
from owl.core.init import initialize_world
from owl.raqic.state import ensure_raqic_fields


@dataclass(frozen=True)
class WorldCase:
    name: str
    mutate: Callable[..., Any]
    description: str


def _all_dead(state: Any, cfg: Any) -> Any:
    state.health[...] = 0.0
    state.resource[...] = 0.0


def _all_live(state: Any, cfg: Any) -> Any:
    state.health[...] = 1.0
    state.resource[...] = 0.75
    if hasattr(state, "obstacle"):
        state.obstacle[...] = False


def _checkerboard(state: Any, cfg: Any) -> Any:
    yy, xx = np.indices(state.health.shape)
    state.obstacle[...] = (yy + xx) % 2 == 0
    state.health[state.obstacle] = 0.0


def _thresholds(state: Any, cfg: Any) -> Any:
    state.health[...] = 0.0
    state.resource[...] = float(cfg.ecology.starvation_threshold)
    state.phase[...] = (
        np.array([0.0, np.pi, 2 * np.pi, 1e-15])
        .take(np.arange(state.phase.size) % 4)
        .reshape(state.phase.shape)
    )


def _extreme_fields(state: Any, cfg: Any) -> Any:
    state.food[...] = 0.0
    state.toxin[...] = 1.0
    state.resource[...] = np.linspace(0, 1, state.resource.size).reshape(state.resource.shape)
    state.health[...] = np.linspace(1, 0, state.health.size).reshape(state.health.shape)


CASES = (
    WorldCase("all_dead", _all_dead, "No eligible units."),
    WorldCase("all_live", _all_live, "All cells live and unobstructed."),
    WorldCase("checkerboard_obstacles", _checkerboard, "Alternating obstacle edge cases."),
    WorldCase("thresholds", _thresholds, "Exact health/resource/phase threshold values."),
    WorldCase("extreme_fields", _extreme_fields, "Food/toxin/resource extremes."),
)


def build_world_case(cfg: Any, name: str, *, seed: int | None = None) -> Any:
    rng = np.random.default_rng(cfg.world.seed if seed is None else seed)
    state = initialize_world(cfg, rng)
    ensure_advanced_fields(state, cfg)
    if getattr(cfg.raqic, "enabled", False):
        ensure_raqic_fields(state, cfg)
    case = next((c for c in CASES if c.name == name), None)
    if case is None:
        raise KeyError(name)
    case.mutate(state, cfg)
    return state


def clone_world(state: Any) -> Any:
    return copy.deepcopy(state)
