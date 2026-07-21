#!/usr/bin/env python3
"""Run Ruff only on changed Python files while preserving the full config."""

from __future__ import annotations

import subprocess
from pathlib import Path


def main() -> int:
    result = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=ACMRT", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    files = [
        name
        for name in result.stdout.splitlines()
        if name.endswith((".py", ".pyi")) and Path(name).exists()
    ]
    if not files:
        print("No changed Python files.")
        return 0
    subprocess.run(["ruff", "check", *files], check=True)
    subprocess.run(["ruff", "format", "--check", *files], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
