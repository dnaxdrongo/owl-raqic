from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from owl.runtime.capabilities import RuntimeCapabilities
from owl.runtime.json_types import json_native
from owl.runtime.run_paths import derive_run_paths

ROOT = Path(__file__).resolve().parents[1]


def test_runtime_capabilities_are_json_native():
    capabilities = RuntimeCapabilities(
        False, 0, False, False, False, False, False, False, {"x": np.int64(2)}
    )
    payload = capabilities.to_dict()
    assert payload["has_cuda"] is False
    assert payload["details"]["x"] == 2


def test_json_native_preserves_numpy_boolean():
    assert json_native(np.bool_(False)) is False


def test_content_addressed_run_paths_are_stable_and_isolated(tmp_path):
    class Plan:
        scientific_contract_version = "science"

        def to_dict(self):
            return {"plan_hash": "abc"}

    paths_a = derive_run_paths(cfg={"seed": 1}, plan=Plan(), root=tmp_path)
    paths_b = derive_run_paths(cfg={"seed": 1}, plan=Plan(), root=tmp_path)
    paths_c = derive_run_paths(cfg={"seed": 2}, plan=Plan(), root=tmp_path)
    assert paths_a.run_id == paths_b.run_id
    assert paths_a.run_id != paths_c.run_id
    assert paths_a.reports.exists()


def test_graph_selected_actions_match_persistent_science(tmp_path):
    output = tmp_path / "graph_science.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/certify_v093_graph_science.py",
            "--ticks",
            "2",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(output.read_text())
    assert payload["passed"] is True
    assert payload["report_sha256"]
