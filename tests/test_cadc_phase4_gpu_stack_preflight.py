from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


def test_gpu_stack_preflight_truthfully_skips_cpu_reference(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "smoke.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "preflight_cadc_phase4_gpu_stack.py"),
            "--config",
            str(root / "configs" / "cadc_phase4_cpu_smoke.yaml"),
            "--output",
            str(output),
        ],
        cwd=root,
        check=False,
    )
    assert completed.returncode == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["skipped"] is True
    assert payload["reason"] == "cpu_reference"


def test_gpu_stack_preflight_avoids_unsupported_cudf_merge_validate() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts" / "preflight_cadc_phase4_gpu_stack.py").read_text(
        encoding="utf-8"
    )
    assert re.search(r"\.merge\([\s\S]*?validate\s*=", source) is None
    assert "many-to-one right-key uniqueness contract" in source
    assert "many-to-one join cardinality contract" in source
