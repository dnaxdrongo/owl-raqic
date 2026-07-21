#!/usr/bin/env python3
# ruff: noqa: E402 -- approved source-tree bootstrap or optional import gate
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from pathlib import Path as _BootstrapPath
from typing import Any

_scripts_dir = _BootstrapPath(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))
from _repo_bootstrap import bootstrap_repo

root = bootstrap_repo()

from owl.core.config import load_config


def validate(directory: str | Path = "configs") -> dict[str, Any]:
    paths = sorted((root / directory).glob("*.yaml"))
    failures = []
    for path in paths:
        try:
            load_config(path)
        except Exception as exc:
            failures.append(
                {
                    "path": str(path.relative_to(root)),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return {
        "count": len(paths),
        "failures": failures,
        "passed": not failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/v09_all_config_results.json")
    args = parser.parse_args()
    result = validate()
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out)
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
