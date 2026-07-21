from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np

from owl.gpu.layout import build_layout_ledger, write_layout_ledger
from owl.gpu.multi_device import partition_rows, split_live_indices, validate_partition
from owl_raqic.qiskit_backend.export_bundle import QuantumExportRecord, export_quantum_experiment


def test_multi_device_partition_is_total_and_deterministic():
    shards = partition_rows(11, [0, 1, 2], halo=1)
    validate_partition(11, shards)
    assert [s.owned_rows for s in shards] == [4, 4, 3]
    split = split_live_indices(np.arange(10), [0, 1, 2])
    assert sum(len(v) for v in split.values()) == 10
    assert np.concatenate(list(split.values())).tolist() == list(range(10))


def test_layout_ledger_is_serializable(tmp_path: Path):
    entries = build_layout_ledger()
    assert entries
    out = write_layout_ledger(tmp_path / "layout.json")
    data = json.loads(out.read_text())
    assert len(data) == len(entries)
    assert all("proposed_layout" in row for row in data)


def test_quantum_export_bundle_without_qiskit(tmp_path: Path):
    record = QuantumExportRecord(
        tick=3,
        cell_id=7,
        yx=(1, 2),
        circuit_kind="static_state_prep",
        action_names=("REST", "FEED"),
        active_primes=(2, 3),
        mask=(True, True),
        expected_probabilities=(0.25, 0.75),
        qiskit_probabilities=None,
        metadata={"backend": "not_run"},
    )
    archive = export_quantum_experiment(tmp_path / "experiment", record, circuit=None)
    assert archive.exists()
    with zipfile.ZipFile(archive) as zf:
        payload = json.loads(zf.read("experiment.json"))
    assert payload["expected_probabilities"] == [0.25, 0.75]
    assert "not evidence" in payload["claim_boundary"].lower()
