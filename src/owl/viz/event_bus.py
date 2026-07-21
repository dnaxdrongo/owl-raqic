from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from enum import IntEnum
from typing import Any, cast

import numpy as np


class VisualEventType(IntEnum):
    MOVE = 1
    FEED = 2
    COMMUNICATE = 3
    INHIBIT = 4
    REPAIR = 5
    REPRODUCE = 6
    DEATH = 7
    MERGE = 8
    SPLIT = 9
    EXPEL = 10
    RAQIC_UNCERTAINTY = 11
    AUDIT_FAILURE = 12
    INGEST = 13
    INTEGRATE = 14
    MOVEMENT_REJECTED = 15
    QISKIT_MISMATCH = 16
    GPU_FALLBACK = 17
    BUFFER_OVERFLOW = 18


_EVENT_PRIORITY = {
    VisualEventType.AUDIT_FAILURE: 100,
    VisualEventType.QISKIT_MISMATCH: 95,
    VisualEventType.BUFFER_OVERFLOW: 90,
    VisualEventType.DEATH: 80,
    VisualEventType.REPRODUCE: 75,
    VisualEventType.MERGE: 72,
    VisualEventType.SPLIT: 72,
    VisualEventType.EXPEL: 70,
    VisualEventType.INGEST: 65,
    VisualEventType.INHIBIT: 60,
    VisualEventType.REPAIR: 50,
    VisualEventType.COMMUNICATE: 45,
    VisualEventType.FEED: 40,
    VisualEventType.MOVE: 20,
    VisualEventType.RAQIC_UNCERTAINTY: 10,
}


@dataclass(frozen=True)
class VisualEvent:
    tick: int
    event_type: VisualEventType
    y: int
    x: int
    target_y: int = -1
    target_x: int = -1
    action: int = 0
    intensity: float = 1.0
    ttl: int = 3
    source_id: int = -1
    channel: int = -1
    payload0: float = 0.0
    payload1: float = 0.0
    priority: int = 0

    @property
    def effective_priority(self) -> int:
        return int(self.priority or _EVENT_PRIORITY.get(self.event_type, 0))


@dataclass
class VisualEventBuffer:
    capacity: int = 4096
    events: list[VisualEvent] = field(default_factory=list)
    overflow_count: int = 0
    truncated_count: int = 0
    critical_drop_count: int = 0
    dropped_by_type: dict[str, int] = field(default_factory=dict)

    def add(
        self,
        event: VisualEvent,
        *,
        strict: bool = False,
        replace_lower_priority: bool = True,
    ) -> None:
        if len(self.events) >= self.capacity:
            self.overflow_count += 1
            self.truncated_count += 1
            incoming_critical = event.effective_priority >= 70
            replacement = None
            if replace_lower_priority and self.events:
                candidates = list(range(len(self.events)))
                if incoming_critical:
                    noncritical = [i for i in candidates if self.events[i].effective_priority < 70]
                    candidates = noncritical or candidates
                worst_i = min(candidates, key=lambda i: self.events[i].effective_priority)
                if event.effective_priority > self.events[worst_i].effective_priority:
                    replacement = worst_i
            if replacement is not None:
                dropped = self.events[replacement]
                self.dropped_by_type[dropped.event_type.name] = (
                    self.dropped_by_type.get(dropped.event_type.name, 0) + 1
                )
                self.events[replacement] = event
                return
            self.dropped_by_type[event.event_type.name] = (
                self.dropped_by_type.get(event.event_type.name, 0) + 1
            )
            if incoming_critical:
                self.critical_drop_count += 1
            if strict or incoming_critical:
                raise OverflowError(
                    f"visual event capacity {self.capacity} could not retain "
                    f"critical={incoming_critical} event {event.event_type.name}"
                )
            return
        self.events.append(event)

    def sort_for_render(self) -> None:
        self.events.sort(key=lambda e: (-e.effective_priority, e.y, e.x, int(e.event_type)))

    def prune(self) -> None:
        self.events = [replace(e, ttl=e.ttl - 1) for e in self.events if e.ttl > 1]

    def event_types(self) -> tuple[VisualEventType, ...]:
        return tuple(event.event_type for event in self.events)

    def extend(self, other: VisualEventBuffer, *, strict: bool = False) -> None:
        for event in other.events:
            self.add(event, strict=strict, replace_lower_priority=True)

    def to_numpy(self) -> np.ndarray:
        out: np.ndarray[Any, np.dtype[Any]] = np.zeros((len(self.events), 14), dtype=np.float32)
        for i, e in enumerate(self.events):
            out[i] = [
                e.tick,
                int(e.event_type),
                e.y,
                e.x,
                e.target_y,
                e.target_x,
                e.action,
                e.intensity,
                e.ttl,
                e.source_id,
                e.channel,
                e.payload0,
                e.payload1,
                e.effective_priority,
            ]
        return out


def _action_event_type(action: int) -> Any:
    from owl.core.actions import Action

    mapping = {
        int(Action.FEED): VisualEventType.FEED,
        int(Action.COMMUNICATE): VisualEventType.COMMUNICATE,
        int(Action.INHIBIT): VisualEventType.INHIBIT,
        int(Action.REPAIR): VisualEventType.REPAIR,
        int(Action.INTEGRATE): VisualEventType.INTEGRATE,
        int(Action.REPRODUCE): VisualEventType.REPRODUCE,
        int(Action.INGEST): VisualEventType.INGEST,
        int(Action.MERGE): VisualEventType.MERGE,
        int(Action.SPLIT): VisualEventType.SPLIT,
        int(Action.EXPEL): VisualEventType.EXPEL,
        int(Action.MOVE_N): VisualEventType.MOVE,
        int(Action.MOVE_S): VisualEventType.MOVE,
        int(Action.MOVE_E): VisualEventType.MOVE,
        int(Action.MOVE_W): VisualEventType.MOVE,
        int(Action.MOVE_NE): VisualEventType.MOVE,
        int(Action.MOVE_NW): VisualEventType.MOVE,
        int(Action.MOVE_SE): VisualEventType.MOVE,
        int(Action.MOVE_SW): VisualEventType.MOVE,
        int(Action.FLEE): VisualEventType.MOVE,
        int(Action.PURSUE): VisualEventType.MOVE,
    }
    return mapping.get(int(action))


def _action_event_code_array(readout: np.ndarray) -> np.ndarray:
    """Map all action readouts to visual-event codes with fixed-size vector operations."""

    from owl.core.actions import Action

    event_codes: np.ndarray[Any, np.dtype[Any]] = np.zeros(readout.shape, dtype=np.int16)
    mapping = {
        VisualEventType.FEED: (Action.FEED,),
        VisualEventType.COMMUNICATE: (Action.COMMUNICATE,),
        VisualEventType.INHIBIT: (Action.INHIBIT,),
        VisualEventType.REPAIR: (Action.REPAIR,),
        VisualEventType.INTEGRATE: (Action.INTEGRATE,),
        VisualEventType.REPRODUCE: (Action.REPRODUCE,),
        VisualEventType.INGEST: (Action.INGEST,),
        VisualEventType.MERGE: (Action.MERGE,),
        VisualEventType.SPLIT: (Action.SPLIT,),
        VisualEventType.EXPEL: (Action.EXPEL,),
        VisualEventType.MOVE: (
            Action.MOVE_N,
            Action.MOVE_S,
            Action.MOVE_E,
            Action.MOVE_W,
            Action.MOVE_NE,
            Action.MOVE_NW,
            Action.MOVE_SE,
            Action.MOVE_SW,
            Action.FLEE,
            Action.PURSUE,
        ),
    }
    for event_type, actions in mapping.items():
        action_values = np.asarray([int(action) for action in actions], dtype=readout.dtype)
        event_codes[np.isin(readout, action_values)] = int(event_type)
    return event_codes


def events_from_state(
    state: Any,
    *,
    max_events: int = 4096,
    entropy_threshold: float = 1.5,
    strict: bool = False,
) -> VisualEventBuffer:
    """Build truthful, priority-limited visual events from actual state."""
    buf = VisualEventBuffer(capacity=max_events)
    readout = getattr(state, "raqic_readout", None)
    if readout is None:
        readout = state.readout
    readout = np.asarray(readout)
    health = np.asarray(state.health)
    tick = int(state.tick)
    probs = getattr(state, "raqic_probabilities", None)
    entropy = None
    if probs is not None:
        p = np.asarray(probs, dtype=float)
        entropy = -np.sum(np.where(p > 0, p * np.log(np.maximum(p, 1e-12)), 0.0), axis=-1)
    occupancy = np.asarray(getattr(state, "occupancy", np.full_like(health, -1, dtype=int)))
    event_codes = _action_event_code_array(readout)
    living = health > 0
    uncertain = (entropy >= entropy_threshold) if entropy is not None else np.zeros_like(living)
    candidate_flat = np.flatnonzero((living & ((event_codes != 0) | uncertain)).reshape(-1))
    width = int(health.shape[1])
    for flat_index in candidate_flat:
        y, x = divmod(int(flat_index), width)
        action = int(readout[y, x])
        event_code = int(event_codes[y, x])
        if event_code:
            event_type = VisualEventType(event_code)
            intensity = float(np.max(probs[y, x])) if probs is not None else 1.0
            channel = -1
            if event_type == VisualEventType.COMMUNICATE and hasattr(state, "signal_emission"):
                signal = np.asarray(state.signal_emission[y, x])
                channel = int(np.argmax(signal)) if signal.size else -1
            buf.add(
                VisualEvent(
                    tick,
                    event_type,
                    y,
                    x,
                    action=action,
                    intensity=intensity,
                    source_id=int(occupancy[y, x]),
                    channel=channel,
                    ttl=3
                    if event_type
                    in (VisualEventType.REPRODUCE, VisualEventType.MERGE, VisualEventType.SPLIT)
                    else 2,
                ),
                strict=strict,
                replace_lower_priority=True,
            )
        if uncertain[y, x]:
            assert entropy is not None
            buf.add(
                VisualEvent(
                    tick,
                    VisualEventType.RAQIC_UNCERTAINTY,
                    y,
                    x,
                    action=action,
                    intensity=float(entropy[y, x]),
                    source_id=int(occupancy[y, x]),
                ),
                strict=strict,
                replace_lower_priority=True,
            )

    death_mask = getattr(state, "last_death_mask", None)
    if death_mask is not None:
        for y, x in np.argwhere(np.asarray(death_mask, dtype=bool)):
            buf.add(
                VisualEvent(tick, VisualEventType.DEATH, int(y), int(x), ttl=4),
                strict=strict,
                replace_lower_priority=True,
            )

    audit_flags = getattr(state, "raqic_audit_flags", None)
    if audit_flags is not None:
        flags = np.asarray(audit_flags)
        bad = np.any(flags != 0, axis=-1) if flags.ndim == 3 else flags != 0
        for y, x in np.argwhere(bad):
            buf.add(
                VisualEvent(tick, VisualEventType.AUDIT_FAILURE, int(y), int(x), ttl=6),
                strict=strict,
                replace_lower_priority=True,
            )

    buf.sort_for_render()
    return buf


def events_from_topology_buffer(
    buffer: Any,
    backend: Any,
    *,
    tick: int,
    max_events: int = 1024,
) -> VisualEventBuffer:
    """Convert accepted topology events at the scheduled visual boundary."""
    out = VisualEventBuffer(capacity=max_events)
    if buffer is None:
        return out
    active = backend.asnumpy(buffer.active & buffer.accepted).astype(bool)
    indices = np.flatnonzero(active)[:max_events]
    if indices.size == 0:
        return out
    et = backend.asnumpy(buffer.event_type[indices]).astype(int)
    sy = backend.asnumpy(buffer.source_y[indices]).astype(int)
    sx = backend.asnumpy(buffer.source_x[indices]).astype(int)
    ty = backend.asnumpy(buffer.target_y[indices]).astype(int)
    tx = backend.asnumpy(buffer.target_x[indices]).astype(int)
    priority = backend.asnumpy(buffer.priority[indices]).astype(float)
    mapping = {
        1: VisualEventType.MERGE,
        2: VisualEventType.SPLIT,
        3: VisualEventType.EXPEL,
    }
    for i in range(indices.size):
        event_type = mapping.get(int(et[i]), VisualEventType.AUDIT_FAILURE)
        out.add(
            VisualEvent(
                tick=int(tick),
                event_type=event_type,
                y=int(sy[i]),
                x=int(sx[i]),
                target_y=int(ty[i]),
                target_x=int(tx[i]),
                intensity=float(priority[i]),
                ttl=3,
                priority=_EVENT_PRIORITY.get(event_type, 0),
            ),
            replace_lower_priority=True,
        )
    return out


def index_events_by_source(
    events: Any,
) -> dict[int, tuple[VisualEvent, ...]]:
    grouped: dict[int, list[VisualEvent]] = {}
    for event in events:
        grouped.setdefault(int(event.source_id), []).append(event)
    return {
        source_id: tuple(
            sorted(
                source_events,
                key=lambda item: (-item.effective_priority, int(item.event_type)),
            )
        )
        for source_id, source_events in grouped.items()
    }


def match_event_for_action(
    events: Any,
    ow_id: int,
    action: Any,
    *,
    source_positions: tuple[tuple[int, int], ...] = (),
) -> VisualEvent | None:
    """Match only authoritative source identity or source geometry.

    A global action-only fallback would attach an unrelated OW's event and is
    therefore intentionally forbidden.
    """

    expected = _action_event_type(int(action))
    event_values = cast(Iterable[VisualEvent], events)
    source_events = index_events_by_source(event_values).get(int(ow_id), ())
    for event in source_events:
        if expected is None or event.event_type == expected:
            return event
    allowed_positions = {tuple(int(value) for value in position) for position in source_positions}
    if allowed_positions:
        for event in event_values:
            if (int(event.y), int(event.x)) in allowed_positions and (
                expected is None or event.event_type == expected
            ):
                return event
    return None


def events_from_event_records(
    records: Any,
    *,
    tick: int,
    occupancy: np.ndarray | None = None,
    max_events: int = 4096,
) -> VisualEventBuffer:
    """Convert authoritative sparse scientific EventRecord values.

    Source/target coordinates and payloads are copied exactly.  The visual layer
    never performs collision, targeting, or topology inference.
    """

    from owl.core.actions import EventKind

    mapping = {
        str(EventKind.DEATH): VisualEventType.DEATH,
        str(EventKind.REPRODUCTION): VisualEventType.REPRODUCE,
        str(EventKind.INGESTION): VisualEventType.INGEST,
        str(EventKind.EXPULSION): VisualEventType.EXPEL,
        str(EventKind.RELEASE): VisualEventType.EXPEL,
        str(EventKind.MERGE): VisualEventType.MERGE,
        str(EventKind.SPLIT): VisualEventType.SPLIT,
        str(EventKind.COLLISION): VisualEventType.MOVEMENT_REJECTED,
        str(EventKind.SIGNAL_OUTCOME): VisualEventType.COMMUNICATE,
        EventKind.DEATH.value: VisualEventType.DEATH,
        EventKind.REPRODUCTION.value: VisualEventType.REPRODUCE,
        EventKind.INGESTION.value: VisualEventType.INGEST,
        EventKind.EXPULSION.value: VisualEventType.EXPEL,
        EventKind.RELEASE.value: VisualEventType.EXPEL,
        EventKind.MERGE.value: VisualEventType.MERGE,
        EventKind.SPLIT.value: VisualEventType.SPLIT,
        EventKind.COLLISION.value: VisualEventType.MOVEMENT_REJECTED,
        EventKind.SIGNAL_OUTCOME.value: VisualEventType.COMMUNICATE,
    }
    out = VisualEventBuffer(capacity=max_events)
    owner = None if occupancy is None else np.asarray(occupancy)
    for record in list(records or ()):
        source = record.source or (-1, -1)
        target = record.target or (-1, -1)
        event_type = mapping.get(str(record.kind))
        if event_type is None:
            continue
        payload = dict(getattr(record, "payload", {}) or {})
        source_value = payload.get(
            "source_id",
            payload.get("ow_id", payload.get("predator_id", payload.get("parent_id", -1))),
        )
        source_id = -1 if source_value is None else int(source_value)
        if source_id < 0 and owner is not None and source[0] >= 0 and source[1] >= 0:
            source_id = int(owner[source[0], source[1]])
        if source_id < 0 and owner is not None and target[0] >= 0 and target[1] >= 0:
            source_id = int(owner[target[0], target[1]])
        channel_value = payload.get("channel", -1)
        channel = -1 if channel_value is None else int(channel_value)
        probability_value = payload.get("probability", payload.get("success", 1.0))
        probability = 1.0 if probability_value is None else float(probability_value)
        transfer_value = payload.get(
            "resource_transfer",
            payload.get("damage", payload.get("priority", 0.0)),
        )
        transfer = 0.0 if transfer_value is None else float(transfer_value)
        out.add(
            VisualEvent(
                tick=int(getattr(record, "tick", tick)),
                event_type=event_type,
                y=int(source[0]),
                x=int(source[1]),
                target_y=int(target[0]),
                target_x=int(target[1]),
                action=int(payload.get("action", 0)),
                intensity=max(0.0, probability),
                ttl=4 if event_type in {VisualEventType.DEATH, VisualEventType.REPRODUCE} else 3,
                source_id=source_id,
                channel=channel,
                payload0=probability,
                payload1=transfer,
            ),
            replace_lower_priority=True,
        )
    out.sort_for_render()
    return out
