from __future__ import annotations

import numpy as np

from owl.core.actions import Action
from owl.core.config import SimulationConfig
from owl.science.action_contract import candidate_target_context
from owl.science.action_transitions import (
    FLEE_FAMILY,
    PURSUE_FAMILY,
    compile_selected_execution_action,
    resolve_action_transition_context,
)


def _cfg() -> SimulationConfig:
    return SimulationConfig.model_validate(
        {
            "world": {"height": 10, "width": 10, "patch_size": 5},
            "action_transitions": {
                "enabled": True,
                "action_contract_version": "owl.action-transitions.v1",
                "legacy_unsupported_action_recovery": False,
                "active_sense_enabled": True,
                "flee_execution_enabled": True,
                "pursue_execution_enabled": True,
            },
        }
    )


def _world() -> dict[str, np.ndarray]:
    shape = (10, 10)
    health = np.zeros(shape, dtype=np.float32)
    health[5, 5] = 1.0
    occupancy = np.full(shape, -1, dtype=np.int64)
    occupancy[5, 5] = 55
    return {
        "health": health,
        "resource": np.ones(shape, dtype=np.float32),
        "obstacle": np.zeros(shape, dtype=bool),
        "occupancy": occupancy,
        "food": np.zeros(shape, dtype=np.float32),
        "toxin": np.zeros(shape, dtype=np.float32),
        "predation": np.zeros(shape, dtype=np.float32),
        "aggression": np.zeros(shape, dtype=np.float32),
        "mobility": np.ones(shape, dtype=np.float32),
    }


class _RejectScalarWhere:
    """NumPy proxy matching CuPy's requirement for an array condition."""

    def __getattr__(self, name: str) -> object:
        return getattr(np, name)

    @staticmethod
    def where(condition: object, x: object, y: object) -> np.ndarray:
        if isinstance(condition, (bool, np.bool_)):
            raise TypeError("where condition must be a backend array")
        return np.where(condition, x, y)


def test_flee_compiles_away_from_visible_western_threat() -> None:
    world = _world()
    world["health"][5, 3] = 1.0
    world["occupancy"][5, 3] = 53
    world["aggression"][5, 3] = 1.0
    context = resolve_action_transition_context(**world, cfg=_cfg(), xp=np)
    assert context.target_x[5, 5, FLEE_FAMILY] == 3
    assert context.flee_compiled_action[5, 5] == int(Action.MOVE_E)
    assert context.flee_executable[5, 5]


def test_pursue_compiles_toward_visible_northeast_target() -> None:
    world = _world()
    world["health"][3, 7] = 1.0
    world["occupancy"][3, 7] = 37
    world["predation"][5, 5] = 1.0
    context = resolve_action_transition_context(**world, cfg=_cfg(), xp=np)
    assert context.target_ow_id[5, 5, PURSUE_FAMILY] == 37
    assert context.pursue_compiled_action[5, 5] == int(Action.MOVE_NE)


def test_blocked_direct_path_uses_best_remaining_deterministic_direction() -> None:
    world = _world()
    world["health"][3, 5] = 1.0
    world["occupancy"][3, 5] = 35
    world["predation"][5, 5] = 1.0
    world["obstacle"][4, 5] = True
    context = resolve_action_transition_context(**world, cfg=_cfg(), xp=np)
    assert context.pursue_executable[5, 5]
    assert context.pursue_compiled_action[5, 5] in {
        int(Action.MOVE_NE), int(Action.MOVE_NW)
    }


def test_no_visible_target_is_inexecutable_and_selected_identity_is_preserved() -> None:
    world = _world()
    context = resolve_action_transition_context(**world, cfg=_cfg(), xp=np)
    assert not context.flee_executable[5, 5]
    assert not context.pursue_executable[5, 5]
    selected = np.full((10, 10), int(Action.FLEE), dtype=np.int16)
    compiled = compile_selected_execution_action(
        selected, context.flee_compiled_action, context.pursue_compiled_action, xp=np
    )
    assert selected[5, 5] == int(Action.FLEE)
    assert compiled[5, 5] == -1


def test_oracle_only_change_outside_sensor_radius_does_not_change_agent_context() -> None:
    world = _world()
    first = resolve_action_transition_context(**world, cfg=_cfg(), xp=np)
    world["health"][0, 0] = 1.0
    world["occupancy"][0, 0] = 999
    world["aggression"][0, 0] = 1.0
    second = resolve_action_transition_context(**world, cfg=_cfg(), xp=np)
    for name in (
        "target_y",
        "target_x",
        "target_ow_id",
        "direction_score",
        "flee_compiled_action",
        "pursue_compiled_action",
    ):
        assert np.array_equal(getattr(first, name)[5, 5], getattr(second, name)[5, 5])


def test_candidate_reason_derivation_never_passes_scalar_bool_to_backend_where() -> None:
    cfg = _cfg()
    world = _world()
    transition = resolve_action_transition_context(**world, cfg=cfg, xp=np)
    physical = {
        name: world[name]
        for name in ("health", "resource", "obstacle", "occupancy", "food", "toxin")
    }
    candidate_target_context(
        **physical,
        parent_id=np.full(world["health"].shape, -1, dtype=np.int64),
        policy_legal=np.ones((*world["health"].shape, len(Action)), dtype=bool),
        boundary_mode=cfg.world.boundary_mode,
        diagonal_movement_enabled=cfg.actions.diagonal_movement_enabled,
        xp=_RejectScalarWhere(),
        action_transition_context=transition,
        action_transition_config=cfg.action_transitions,
        movement_cost=cfg.resources.movement_cost,
    )
