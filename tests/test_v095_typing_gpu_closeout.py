from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from owl.core.config import load_config
from owl.core.init import initialize_world
from owl.gpu.full_loop import step_gpu_full
from owl.gpu.qiskit_transfer import pack_qiskit_rows, unpack_qiskit_rows
from owl.gpu.transfer_ledger import TransferKind, TransferLedger
from owl.viz.frame_model import VisualFrame

ROOT = Path(__file__).resolve().parents[1]


def test_transfer_ledger_rejects_unscheduled_boundary(tmp_path: Path) -> None:
    ledger = TransferLedger()
    ledger.record_d2h(
        32,
        kind=TransferKind.METRIC,
        tick=3,
        synchronization="event",
        scheduled=True,
        graph_compatible=False,
        reason="scheduled metric slab",
    )
    ledger.assert_production_safe()
    ledger.record_d2h(
        8,
        kind=TransferKind.COMPATIBILITY,
        tick=4,
        synchronization="device",
        scheduled=False,
        graph_compatible=False,
        reason="injected unscheduled transfer",
    )
    with pytest.raises(RuntimeError, match="unscheduled"):
        ledger.assert_production_safe()
    path = ledger.write(tmp_path / "transfer.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["unscheduled_count"] == 1
    assert payload["metric_bytes"] == 32


def test_qiskit_packed_slab_preserves_identity_and_values() -> None:
    probabilities = np.array([[0.25, 0.75], [1.0, 0.0]], dtype=np.float64)
    phases = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    authority = np.array([[True, True], [True, False]], dtype=np.bool_)
    parent = np.array([[0.6, 0.4], [0.5, 0.5]], dtype=np.float64)
    ow_ids = np.array([2**53 + 11, 2**53 + 17], dtype=np.int64)
    flat_indices = np.array([7, 29], dtype=np.int64)

    slab, layout = pack_qiskit_rows(
        np,
        probabilities=probabilities,
        phases=phases,
        authority=authority,
        parent=parent,
        ow_ids=ow_ids,
        flat_indices=flat_indices,
    )
    restored = unpack_qiskit_rows(slab, layout)

    np.testing.assert_array_equal(restored.ow_ids, ow_ids)
    np.testing.assert_array_equal(restored.flat_indices, flat_indices)
    assert restored.authority is not None
    assert restored.phases is not None
    assert restored.parent is not None
    np.testing.assert_array_equal(restored.authority, authority)
    np.testing.assert_allclose(restored.probabilities, probabilities, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(restored.phases, phases, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(restored.parent, parent, rtol=0.0, atol=0.0)
    assert slab.nbytes == layout.total_bytes


def test_visual_frame_reports_bounded_payload_bytes() -> None:
    rgba = np.zeros((4, 5, 4), dtype=np.uint8)
    markers = np.zeros((3, 2), dtype=np.float32)
    frame = VisualFrame(rgba=rgba, markers=markers)
    expected = sum(
        array.nbytes
        for array in (
            frame.rgba,
            frame.markers,
            frame.marker_colors,
            frame.marker_sizes,
            frame.lines,
            frame.line_colors,
            frame.arrows,
            frame.sprite_positions,
            frame.glyph_lines,
            frame.glyph_line_colors,
        )
        if array is not None
    )
    assert frame.estimated_nbytes() == expected


def test_stage_once_writeback_is_rejected_for_persistent_tier() -> None:
    cfg = load_config(ROOT / "configs" / "default.yaml").model_copy(deep=True)
    cfg.raqic.full_gpu_execution_tier = "persistent"
    state = initialize_world(cfg, np.random.default_rng(cfg.world.seed))
    with pytest.raises(RuntimeError, match="stage-once compatibility API"):
        step_gpu_full(state, cfg)


def test_hotspot_scanner_classifies_every_finding(tmp_path: Path) -> None:
    output = tmp_path / "hotspots.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "audit_gpu_hotspots.py"),
            "--mode",
            "json",
            "--out",
            str(output),
            "--require-classified",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["summary"]["strict_findings"] == 0
    assert payload["summary"]["classified_findings"] == payload["summary"]["all_findings"]
