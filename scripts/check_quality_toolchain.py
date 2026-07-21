#!/usr/bin/env python3
"""Verify the exact quality-tool versions used by repository certification."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _version(command: list[str], pattern: str) -> str:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    match = re.search(pattern, result.stdout.strip())
    if match is None:
        raise RuntimeError(f"unable to parse version from: {result.stdout!r}")
    return match.group(1)


def main() -> int:
    expected = json.loads((ROOT / "quality_toolchain.json").read_text(encoding="utf-8"))
    actual = {
        "ruff": _version([sys.executable, "-m", "ruff", "--version"], r"ruff ([0-9.]+)"),
        "mypy": _version([sys.executable, "-m", "mypy", "--version"], r"mypy ([0-9.]+)"),
        "pytest": _version([sys.executable, "-m", "pytest", "--version"], r"pytest ([0-9.]+)"),
    }
    failures = {
        name: {"expected": expected[name], "actual": actual[name]}
        for name in actual
        if actual[name] != expected[name]
    }
    report = {
        "schema_version": "owl.quality-toolchain-check.v1",
        "passed": not failures,
        "python_running": ".".join(str(value) for value in sys.version_info[:3]),
        "python_target": expected["python_target"],
        "expected": {name: expected[name] for name in actual},
        "actual": actual,
        "failures": failures,
    }
    output = ROOT / "reports" / "quality_toolchain_check.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
