from __future__ import annotations

import json
import subprocess
import sys


def _failed_manifest(path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "owl.cadc.phase3-acceptance.v1",
                "passed": False,
                "phase4_unlocked": False,
                "failure_stage": "control_state_hash",
                "failures": ["TypeError: ndarray"],
            }
        )
    )


def test_certifier_classifies_incomplete_acceptance_without_key_error(tmp_path) -> None:
    _failed_manifest(tmp_path / "phase3_acceptance_manifest.json")
    (tmp_path / "command_status.json").write_text('{"acceptance_runner": 1}\n')
    output = tmp_path / "phase3_certificate.json"
    process = subprocess.run(
        [
            sys.executable,
            "scripts/certify_cadc_phase3.py",
            "--input",
            str(tmp_path),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    certificate = json.loads(output.read_text())
    assert process.returncode != 0
    assert certificate["classification"] == "FAILED_CLOSED_UPSTREAM_ACCEPTANCE"
    assert certificate["upstream_failure_stage"] == "control_state_hash"
    assert "certifier_exception" not in " ".join(certificate["failures"])
    assert certificate["phase4_unlocked"] is False


def test_profiler_classifies_incomplete_acceptance_without_empty_success(tmp_path) -> None:
    _failed_manifest(tmp_path / "phase3_acceptance_manifest.json")
    output = tmp_path / "performance.json"
    process = subprocess.run(
        [
            sys.executable,
            "scripts/profile_cadc_phase3.py",
            "--config",
            "configs/cadc_phase3_phase25_numpy_smoke.yaml",
            "--input",
            str(tmp_path),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    profile = json.loads(output.read_text())
    assert process.returncode != 0
    assert profile["classification"] == "FAILED_CLOSED_UPSTREAM_OR_PROFILE"
    assert profile["upstream_failure_stage"] == "control_state_hash"
    assert any("upstream_acceptance_failed" in value for value in profile["failures"])
