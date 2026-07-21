from __future__ import annotations

from pathlib import Path

from scripts.certify_v096_actualization_extensions import (
    _default_repo_python,
    _probe_sys_executable,
    _resolve_executable,
)

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_v096_parity_repair_certificate.sh"


def test_certificate_runner_ignores_legacy_python_environment_variables() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "V096_QUALITY_PYTHON" in text
    assert "V096_RUNTIME_PYTHON" in text
    assert 'QUALITY_PYTHON="${QUALITY_PYTHON:-' not in text
    assert 'RUNTIME_PYTHON="${RUNTIME_PYTHON:-' not in text


def test_certifier_prefers_repository_virtual_environments_when_present() -> None:
    candidates = (
        (ROOT / ".venv_quality/bin/python", ".venv_quality/bin/python"),
        (ROOT / ".venv/bin/python", ".venv/bin/python"),
    )

    for candidate, relative in candidates:
        if not candidate.is_file():
            continue
        expected = str(candidate.absolute())
        default = _default_repo_python(relative)
        resolved = _resolve_executable(default)
        probe = _probe_sys_executable(resolved)

        assert default == expected
        assert resolved == expected
        assert probe["passed"] is True
        assert probe["requested"] == expected
        assert probe["sys_executable"] == expected
