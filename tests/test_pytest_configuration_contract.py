from __future__ import annotations

import ast
import importlib.util
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_certifier() -> object:
    path = ROOT / "scripts/certify_v096_actualization_extensions.py"
    spec = importlib.util.spec_from_file_location("v096_certifier_contract", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pytest_configuration_has_no_unowned_asyncio_option() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    options = data["tool"]["pytest"]["ini_options"]
    assert "asyncio_default_fixture_loop_scope" not in options


def test_repository_does_not_declare_unused_pytest_asyncio_dependency() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = [
        *data.get("project", {}).get("dependencies", []),
        *data.get("project", {}).get("optional-dependencies", {}).get("dev", []),
    ]
    has_async_tests = any(
        isinstance(node, ast.AsyncFunctionDef) and node.name.startswith("test_")
        for path in (ROOT / "tests").glob("test_*.py")
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
    )
    declares_plugin = any(str(item).lower().startswith("pytest-asyncio") for item in dependencies)
    assert not has_async_tests
    assert not declares_plugin


def test_certificate_pytest_commands_keep_warning_strictness() -> None:
    certifier = _load_certifier()
    command = certifier.strict_pytest_command(  # type: ignore[attr-defined]
        sys.executable,
        ["tests/test_v096_baseline_recovery.py"],
    )
    assert command[:3] == [sys.executable, "-m", "pytest"]
    assert command[-3:] == ["-W", "error", "tests/test_v096_baseline_recovery.py"]
    assert command[3:6] == ["-q", "-W", "error"]

    collect = certifier.strict_pytest_command(  # type: ignore[attr-defined]
        sys.executable,
        collect_only=True,
    )
    assert collect[3:] == ["--collect-only", "-q", "-W", "error"]
