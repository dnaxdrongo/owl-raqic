"""Quality-tool and formatting contract tests."""

from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path

import owl_raqic

ROOT = Path(__file__).resolve().parents[1]


def test_quality_toolchain_is_exactly_pinned() -> None:
    toolchain = json.loads((ROOT / "quality_toolchain.json").read_text(encoding="utf-8"))
    lock = (ROOT / "requirements-quality.lock").read_text(encoding="utf-8")
    assert toolchain["ruff"] == "0.15.21"
    assert f"ruff=={toolchain['ruff']}" in lock
    assert f"mypy=={toolchain['mypy']}" in lock
    assert f"pytest=={toolchain['pytest']}" in lock


def test_quality_toolchain_checker_passes() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/check_quality_toolchain.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads((ROOT / "reports/quality_toolchain_check.json").read_text())
    assert report["passed"] is True


def test_package_version_matches_active_package() -> None:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)["project"]
    assert project["version"] == owl_raqic.__version__
