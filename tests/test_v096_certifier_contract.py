from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts.certify_v096_actualization_extensions import (
    _probe_sys_executable,
    _resolve_executable,
    strict_pytest_command,
)

ROOT = Path(__file__).resolve().parents[1]
CERTIFIER = ROOT / "scripts/certify_v096_actualization_extensions.py"


def test_certifier_uses_strict_pytest_commands() -> None:
    command = strict_pytest_command(sys.executable, ["tests/test_v096_actualization_math.py"])
    assert command[:3] == [sys.executable, "-m", "pytest"]
    assert command[command.index("-W") + 1] == "error"


def test_certifier_interpreter_contract_resolves_current_python() -> None:
    executable = _resolve_executable(sys.executable)
    probe = _probe_sys_executable(executable)
    assert probe["passed"] is True
    assert probe["requested"] == probe["sys_executable"]


def test_certifier_rejects_nonempty_or_locked_output(tmp_path: Path) -> None:
    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    (nonempty / "existing.txt").write_text("existing\n", encoding="utf-8")
    rejected = subprocess.run(
        [
            sys.executable,
            str(CERTIFIER),
            "--output-dir",
            str(nonempty),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert rejected.returncode != 0
    assert "refusing nonempty certificate output directory" in rejected.stderr

    locked = tmp_path / "locked"
    locked.mkdir()
    (locked / ".certificate.lock").write_text("{}\n", encoding="utf-8")
    lock_rejected = subprocess.run(
        [
            sys.executable,
            str(CERTIFIER),
            "--output-dir",
            str(locked),
            "--allow-existing-output",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert lock_rejected.returncode != 0
    assert "already locked" in lock_rejected.stderr
