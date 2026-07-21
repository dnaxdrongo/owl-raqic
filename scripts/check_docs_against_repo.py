#!/usr/bin/env python3
# ruff: noqa: E402 -- approved source-tree bootstrap or optional import gate
"""Check that paths and runnable commands named in docs exist in the repo."""

from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath
from typing import Any

_scripts_dir = _BootstrapPath(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))
from _repo_bootstrap import bootstrap_repo
from _v092_support import write_report

bootstrap_repo()

import argparse
import re
from pathlib import Path

PATH_RE = re.compile(
    r"(?:^|[\s`'\"])((?:scripts|configs|tests|docs|benchmarks)/"
    r"[A-Za-z0-9_./-]+\.(?:py|sh|yaml|yml|md))"
)
MODE_RE = re.compile(r"\b(?:mode|full_gpu_execution_tier)\s*:\s*([A-Za-z0-9_]+)")


def check(root: Path) -> dict[str, Any]:
    missing: set[str] = set()
    references: set[str] = set()
    modes: set[str] = set()
    for doc in root.joinpath("docs").rglob("*.md"):
        if "PROMPT" in doc.name.upper():
            continue
        text = doc.read_text(encoding="utf-8", errors="replace")
        for match in PATH_RE.finditer(text):
            rel = match.group(1).rstrip(".,:;)")
            references.add(rel)
            if not (root / rel).exists():
                missing.add(rel)
        modes.update(MODE_RE.findall(text))
    return {
        "certificate": "documentation_consistency",
        "references": sorted(references),
        "missing": sorted(missing),
        "modes_named": sorted(modes),
        "passed": not missing,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default="reports/docs_consistency.json")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    result = check(Path(args.root).resolve())
    out = write_report(args.out, result)
    print(out)
    if args.strict and not result["passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
