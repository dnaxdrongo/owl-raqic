from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_tree(
    root: str | Path, *, exclude: Any = ("results", "runs", ".git", "__pycache__", ".pytest_cache")
) -> str:
    root = Path(root)
    h = hashlib.sha256()
    excluded = set(exclude)
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root)
        if any(part in excluded for part in rel.parts):
            continue
        h.update(rel.as_posix().encode())
        h.update(b"\0")
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class RunManifest:
    repo_sha256: str
    config_sha256: str
    environment_sha256: str
    seed: int
    precision: str
    run_class: str
    all_cell_semantics: bool
    fallback_count: int
    python: str
    command: list[str]
    certification_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return path


def environment_fingerprint() -> str:
    try:
        freeze = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True, timeout=60
        )
    except Exception:
        freeze = ""
    payload = json.dumps(
        {
            "python": sys.version,
            "executable": sys.executable,
            "pip_freeze": sorted(line.strip() for line in freeze.splitlines() if line.strip()),
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def build_run_manifest(
    repo_root: str | Path,
    config_path: str | Path,
    *,
    seed: int,
    precision: str,
    run_class: str,
    all_cell_semantics: bool,
    fallback_count: int,
    certification_path: str | Path | None = None,
    command: list[str] | None = None,
) -> RunManifest:
    return RunManifest(
        repo_sha256=sha256_tree(repo_root),
        config_sha256=sha256_file(config_path),
        environment_sha256=environment_fingerprint(),
        seed=int(seed),
        precision=str(precision),
        run_class=str(run_class),
        all_cell_semantics=bool(all_cell_semantics),
        fallback_count=int(fallback_count),
        python=sys.version,
        command=list(command or sys.argv),
        certification_sha256=sha256_file(certification_path) if certification_path else None,
    )
