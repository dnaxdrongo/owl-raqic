from __future__ import annotations

import numpy as np
import pytest
from scripts._run_cadc_phase4_corpus_unit import _bounded_table_packets

from owl.counterfactual.staging import TablePacket


def _reassemble(parts: list[TablePacket], name: str) -> object:
    values = [part.columns[name] for part in parts]
    if isinstance(values[0], list):
        return [item for value in values for item in value]
    return np.concatenate(values)


def test_oversized_packet_is_split_without_row_or_order_changes() -> None:
    rows = 2_000
    packet = TablePacket(
        "branch_events",
        {
            "tick": np.arange(rows, dtype=np.int64),
            "event_code": np.arange(rows, dtype=np.int16) % 22,
            "event_id": [f"event-{index:06d}-" + ("x" * (index % 37)) for index in range(rows)],
        },
    )
    parts = list(
        _bounded_table_packets(
            [packet],
            max_packet_bytes=4_096,
            max_pending_bytes=8_192,
        )
    )
    assert len(parts) > 1
    assert all(part.table_name == "branch_events" for part in parts)
    assert all(0 < part.nbytes <= 4_096 for part in parts)
    assert sum(part.rows for part in parts) == rows
    np.testing.assert_array_equal(_reassemble(parts, "tick"), packet.columns["tick"])
    np.testing.assert_array_equal(
        _reassemble(parts, "event_code"), packet.columns["event_code"]
    )
    assert _reassemble(parts, "event_id") == packet.columns["event_id"]


def test_pending_limit_is_the_effective_bound() -> None:
    packet = TablePacket(
        "branch_contributions",
        {
            "delta": np.arange(1_000, dtype=np.float64),
            "field": ["health"] * 1_000,
        },
    )
    parts = list(
        _bounded_table_packets(
            [packet],
            max_packet_bytes=32_768,
            max_pending_bytes=2_048,
        )
    )
    assert len(parts) > 1
    assert all(part.nbytes <= 2_048 for part in parts)
    np.testing.assert_array_equal(_reassemble(parts, "delta"), packet.columns["delta"])
    assert _reassemble(parts, "field") == packet.columns["field"]


def test_packet_with_one_unrepresentable_row_fails_closed() -> None:
    packet = TablePacket("branch_events", {"event_id": ["x" * 2_048]})
    with pytest.raises(MemoryError, match="row 0"):
        list(
            _bounded_table_packets(
                [packet],
                max_packet_bytes=1_024,
                max_pending_bytes=1_024,
            )
        )


def test_malformed_column_lengths_fail_closed() -> None:
    packet = TablePacket(
        "branch_events",
        {
            "tick": np.arange(100, dtype=np.int64),
            "event_id": ["x" * 100] * 99,
        },
    )
    with pytest.raises(ValueError, match="has 99 rows; expected 100"):
        list(
            _bounded_table_packets(
                [packet],
                max_packet_bytes=1_024,
                max_pending_bytes=1_024,
            )
        )
