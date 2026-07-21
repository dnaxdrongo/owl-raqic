from __future__ import annotations

from dataclasses import replace

from owl.core.actions import Action
from owl.viz.action_animation import (
    ActionContext,
    resolve_action_context,
    sample_action_animation,
    shortest_toroidal_delta,
    validate_all_actions_covered,
)
from owl.viz.event_bus import VisualEvent, VisualEventType
from tests.visual_overhaul_helpers import synthetic_snapshot


def test_all_twenty_two_actions_are_covered() -> None:
    validate_all_actions_covered()
    context = ActionContext(
        1, 1, (2.0, 2.0), (2.0, 3.0), (0.0, 1.0), True, 3, 1.0, None, "toroidal", (10, 10)
    )
    for action in Action:
        pose, effects = sample_action_animation(action, context, 0.5)
        assert pose.position is not None
        assert effects


def test_toroidal_motion_uses_shortest_path() -> None:
    assert shortest_toroidal_delta((0.0, 0.0), (9.0, 9.0), (10, 10)) == (-1.0, -1.0)


def test_failed_move_recoils_without_translating_to_target() -> None:
    context = ActionContext(
        1,
        1,
        (2.0, 2.0),
        (2.0, 3.0),
        (0.0, 1.0),
        False,
        -1,
        1.0,
        VisualEventType.MOVEMENT_REJECTED,
        "toroidal",
        (10, 10),
    )
    pose, effects = sample_action_animation(Action.MOVE_E, context, 1.0)
    assert pose.position == (2.0, 2.0)
    assert effects[0].kind == "movement_recoil"


def test_event_target_is_used_only_for_matching_source() -> None:
    previous = synthetic_snapshot(tick=1)
    current = replace(previous, tick=2)
    ow_id = next(iter(current.id_to_position))
    position = current.position_of(ow_id)
    assert position is not None
    unrelated = VisualEvent(
        2, VisualEventType.INGEST, 0, 0, 0, 1, action=int(Action.INGEST), source_id=999
    )
    matching = VisualEvent(
        2,
        VisualEventType.INGEST,
        position[0],
        position[1],
        position[0],
        position[1] + 1,
        action=int(Action.INGEST),
        source_id=ow_id,
    )
    context = resolve_action_context(previous, current, (unrelated, matching), ow_id, Action.INGEST)
    assert context.target == (float(position[0]), float(position[1] + 1))


def test_feed_without_scientific_event_is_nondirectional() -> None:
    snapshot = synthetic_snapshot()
    ow_id = next(iter(snapshot.id_to_position))
    context = resolve_action_context(snapshot, snapshot, (), ow_id, Action.FEED)
    assert context.target is None
