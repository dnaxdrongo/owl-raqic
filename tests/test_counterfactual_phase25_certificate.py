from __future__ import annotations

import json
import subprocess
import sys


def test_certifier_missing_input_materializes_failed_closed_certificate(tmp_path) -> None:
    output = tmp_path / "phase3_certificate.json"
    process = subprocess.run(
        [
            sys.executable,
            "scripts/certify_cadc_phase3.py",
            "--input",
            str(tmp_path / "missing"),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert process.returncode != 0
    certificate = json.loads(output.read_text())
    assert certificate["passed"] is False
    assert certificate["phase4_unlocked"] is False
