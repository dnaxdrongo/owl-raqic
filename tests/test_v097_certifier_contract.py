from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CERTIFIER = ROOT / "scripts/certify_v097_parity_contract.py"
RUNNER = ROOT / "scripts/run_v097_parity_certificate.sh"


def _load_certifier() -> object:
    spec = importlib.util.spec_from_file_location("v097_certifier_contract", CERTIFIER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v097_certificate_schema_and_scope() -> None:
    certifier = _load_certifier()

    assert certifier.SCHEMA_VERSION == "owl.raqic.v097.parity-certificate.v1"  # type: ignore[attr-defined]
    mypy_files = certifier.MYPY_FILES  # type: ignore[attr-defined]
    assert "apply_v097_parity_contract.py" in mypy_files
    assert "rollback_v097_parity_contract.py" in mypy_files
    assert "src/owl/gpu/shadow_audit.py" in mypy_files
    assert "tests/test_v097_certifier_contract.py" in mypy_files
    assert "tests/test_v097_parity_tolerance.py" in mypy_files


def test_v097_runner_uses_separate_quality_and_runtime_venvs() -> None:
    text = RUNNER.read_text(encoding="utf-8")

    assert ".venv_quality/bin/python" in text
    assert ".venv/bin/python" in text
    assert "--quality-python" in text
    assert "--runtime-python" in text
    assert "V097_QUALITY_PYTHON" in text
    assert "V097_RUNTIME_PYTHON" in text
    assert "RUN_250_CADENCE" in text
