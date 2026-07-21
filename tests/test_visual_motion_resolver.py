from __future__ import annotations

from dataclasses import replace

import numpy as np

from owl.core.actions import Action
from owl.viz.action_animation import resolve_action_context
from owl.viz.visual_snapshot import snapshot_from_world_state
from tests.visual_overhaul_helpers import synthetic_world


def test_stable_id_motion_is_resolved_from_snapshot_positions() -> None:
    world = synthetic_world(tick=1)
    previous = snapshot_from_world_state(world)
    source = (1, 1)
    target = (1, 2)
    ow_id = int(world.occupancy[source])
    world.occupancy[target] = ow_id
    world.occupancy[source] = -1
    world.health[target] = world.health[source]
    world.health[source] = 0.0
    world.tick = 2
    current = snapshot_from_world_state(world)
    context = resolve_action_context(previous, current, (), ow_id, Action.MOVE_E)
    assert context.source == (1.0, 1.0)
    assert context.target == (1.0, 2.0)
    assert context.successful_move is True


def test_no_mutation_of_snapshot_during_resolution() -> None:
    snapshot = snapshot_from_world_state(synthetic_world())
    before = snapshot.field("occupancy").copy()
    ow_id = next(iter(snapshot.id_to_position))
    resolve_action_context(
        snapshot, replace(snapshot, tick=snapshot.tick + 1), (), ow_id, Action.REST
    )
    assert np.array_equal(snapshot.field("occupancy"), before)
