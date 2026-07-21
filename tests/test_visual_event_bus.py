from owl.viz.event_bus import VisualEvent, VisualEventBuffer, VisualEventType


def test_visual_event_buffer_preserves_critical_events_and_ttl():
    buf = VisualEventBuffer(capacity=1)
    buf.add(VisualEvent(1, VisualEventType.FEED, 2, 3, ttl=2))
    buf.add(VisualEvent(1, VisualEventType.DEATH, 4, 5, ttl=3))
    assert buf.overflow_count == 1
    assert buf.critical_drop_count == 0
    assert buf.event_types() == (VisualEventType.DEATH,)
    assert buf.dropped_by_type == {"FEED": 1}
    buf.prune()
    assert len(buf.events) == 1
    buf.prune()
    assert len(buf.events) == 1
    buf.prune()
    assert len(buf.events) == 0


def test_visual_event_buffer_fails_closed_when_only_higher_priority_critical_remains():
    buf = VisualEventBuffer(capacity=1)
    buf.add(VisualEvent(1, VisualEventType.AUDIT_FAILURE, 0, 0))
    try:
        buf.add(VisualEvent(1, VisualEventType.DEATH, 1, 1))
    except OverflowError:
        pass
    else:
        raise AssertionError("critical event loss must fail closed")
    assert buf.critical_drop_count == 1
    assert buf.event_types() == (VisualEventType.AUDIT_FAILURE,)
