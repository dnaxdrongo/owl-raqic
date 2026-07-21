from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_missing_evidence_materializes_failed_closed_certificate(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "certificate.json"
    missing = tmp_path / "missing.json"
    command = [
        sys.executable,
        str(root / "scripts/certify_cadc_phase4.py"),
        "--config",
        str(root / "configs/cadc_phase4_cpu_smoke.yaml"),
        "--corpus-certificate",
        str(missing),
        "--dataset-receipt",
        str(missing),
        "--repeat-pilot",
        str(missing),
        "--training-receipt",
        str(missing),
        "--calibration-receipt",
        str(missing),
        "--score-receipt",
        str(missing),
        "--evaluation",
        str(missing),
        "--negative-controls",
        str(missing),
        "--math-verification",
        str(missing),
        "--casebook-manifest",
        str(missing),
        "--environment-manifest",
        str(missing),
        "--gpu-stack-smoke",
        str(missing),
        "--performance",
        str(missing),
        "--hotpath-audit",
        str(missing),
        "--synthetic-scenarios",
        str(missing),
        "--command-status",
        str(missing),
        "--output",
        str(output),
    ]
    completed = subprocess.run(command, cwd=root, check=False)
    assert completed.returncode != 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert payload["classification"] == "FAILED_CLOSED"
    assert payload["phase5_unlocked"] is False
