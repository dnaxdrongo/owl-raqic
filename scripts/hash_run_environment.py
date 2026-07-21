#!/usr/bin/env python3
# ruff: noqa: E402 -- approved source-tree bootstrap or optional import gate
from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath

_scripts_dir = _BootstrapPath(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))
from _repo_bootstrap import bootstrap_repo

bootstrap_repo()

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/run_environment.json")
    args = ap.parse_args()
    try:
        freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    except Exception as exc:
        freeze = f"ERROR: {type(exc).__name__}: {exc}"
    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "pip_freeze": freeze.splitlines(),
        "environment_sha256": sha256_bytes(freeze.encode()),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
