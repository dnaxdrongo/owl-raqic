#!/usr/bin/env python3
# ruff: noqa: E402 -- approved source-tree bootstrap or optional import gate
"""Collect a redacted, bounded experiment artifact archive."""

from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath

_scripts_dir = _BootstrapPath(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))
from _repo_bootstrap import bootstrap_repo

bootstrap_repo()

import argparse
import re
import zipfile
from pathlib import Path

DEFAULT_ROOTS = ("reports", "results", "runs", "configs", "docs")
EXCLUDED_PARTS = {
    ".git",
    ".venv",
    ".venv-gpu",
    ".venv-gpu-full",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    ".ssh",
}
SECRET_NAMES = re.compile(r"(id_rsa|id_ed25519|private[_-]?key|credentials|token|secret)", re.I)
SECRET_LINE = re.compile(r"(?i)(api[_-]?key|access[_-]?token|secret|password)\s*[:=]\s*\S+")
TEXT_SUFFIXES = {".txt", ".log", ".json", ".yaml", ".yml", ".md", ".csv", ".toml", ".ini"}


def safe_file(path: Path, repo: Path, max_bytes: int) -> bool:
    try:
        rel = path.resolve().relative_to(repo.resolve())
    except ValueError:
        return False
    if any(part in EXCLUDED_PARTS for part in rel.parts):
        return False
    if SECRET_NAMES.search(path.name):
        return False
    if path.is_symlink() or not path.is_file():
        return False
    return path.stat().st_size <= max_bytes


def redacted_bytes(path: Path) -> bytes:
    data = path.read_bytes()
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return data
    text = data.decode("utf-8", errors="replace")
    return SECRET_LINE.sub(lambda m: m.group(1) + "=<REDACTED>", text).encode("utf-8")


def collect(repo: Path, output: Path, roots: tuple[str, ...], max_bytes: int) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in roots:
            base = repo / name
            if not base.exists():
                continue
            for path in sorted(base.rglob("*")):
                if safe_file(path, repo, max_bytes):
                    zf.writestr(str(path.relative_to(repo)), redacted_bytes(path))
    return output


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out", default="owl_safe_artifacts.zip")
    ap.add_argument("--roots", nargs="*", default=list(DEFAULT_ROOTS))
    ap.add_argument("--max-file-mb", type=float, default=256.0)
    args = ap.parse_args()
    out = collect(
        Path(args.repo).resolve(),
        Path(args.out).resolve(),
        tuple(args.roots),
        int(args.max_file_mb * 1024 * 1024),
    )
    print(out)


if __name__ == "__main__":
    main()
