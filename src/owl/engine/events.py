"""Sparse event queue management.

Events are Python-level records for rare topology/collision/ingestion events.
Dense cell updates stay in NumPy arrays; this module only manages the queue.
"""

from __future__ import annotations

from collections import defaultdict

from owl.core.state import EventRecord, WorldState


def enqueue_event(state: WorldState, event: EventRecord) -> None:
    """Add a sparse event to ``state.event_queue``.

    Parameters
    ----------
    state:
        World state whose queue will be mutated.
    event:
        Event record to append.
    """
    if not isinstance(event, EventRecord):
        raise TypeError(f"event must be EventRecord, got {type(event).__name__}")
    state.event_queue.append(event)


def dequeue_events(state: WorldState, kind: str | None = None) -> list[EventRecord]:
    """Remove and return queued events, optionally filtering by kind.

    If ``kind`` is ``None``, all events are removed and returned. If ``kind`` is
    supplied, only matching events are removed; nonmatching events remain queued
    in their original relative order.
    """
    if kind is None:
        events = list(state.event_queue)
        state.event_queue.clear()
        return events

    kind_str = str(kind)
    matched: list[EventRecord] = []
    remaining: list[EventRecord] = []
    for event in state.event_queue:
        if str(event.kind) == kind_str:
            matched.append(event)
        else:
            remaining.append(event)
    state.event_queue[:] = remaining
    return matched


def route_events(state: WorldState) -> dict[str, list[EventRecord]]:
    """Return a non-mutating grouping of queued events by event kind."""
    routed: dict[str, list[EventRecord]] = defaultdict(list)
    for event in state.event_queue:
        routed[str(event.kind)].append(event)
    return dict(routed)
