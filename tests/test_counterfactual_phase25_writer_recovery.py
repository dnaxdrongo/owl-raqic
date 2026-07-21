from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from owl.counterfactual.staging import TablePacket
from owl.counterfactual.writer import CounterfactualWriter
from owl.record.cadc_schema import CADC_ACTION_TRANSITION_SCHEMA_DIGEST


def _writer(root: Path, *, resume: bool = False) -> CounterfactualWriter:
    return CounterfactualWriter(
        root,
        source_sha256="a" * 64,
        phase25_certificate_sha256="b" * 64,
        factual_v2_digest=CADC_ACTION_TRANSITION_SCHEMA_DIGEST,
        max_packet_bytes=1_000_000,
        max_pending_bytes=1_000_000,
        row_group_rows=128,
        resume=resume,
    )


def test_writer_recovers_valid_part_and_rejects_corruption(tmp_path) -> None:
    writer = _writer(tmp_path)
    receipt = writer.write_packet(TablePacket("source_states", {"tick": np.array([1])}))
    assert _writer(tmp_path, resume=True).receipts == [receipt]
    part = tmp_path / "source_states" / receipt.path
    part.write_bytes(part.read_bytes() + b"corruption")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        _writer(tmp_path, resume=True)
