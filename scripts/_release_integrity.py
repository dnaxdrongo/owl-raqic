"""Deterministic helpers for computing source and package identities."""

from __future__ import annotations

import hashlib
import json
import tomllib
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

RELEASE_VERSION = "0.9.5.1"
RELEASE_CLASSIFICATION = "source_release_integrity_closed_target_gpu_certification_pending"

_EXCLUDED_DIR_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "runs",
    "venv",
}
_EXCLUDED_TOP_LEVEL = {"reports", "results"}
_RELEASE_SELF_EXCLUDES = {
    "MANIFEST_SHA256.txt",
    "RELEASE_MANIFEST.json",
    "quality/V0_9_5_1_RELEASE_INTEGRITY.json",
}
_SHA_SELF_EXCLUDES = {
    "MANIFEST_SHA256.txt",
    "quality/V0_9_5_1_RELEASE_INTEGRITY.json",
}


def _is_excluded(relative: Path) -> bool:
    if not relative.parts:
        return False
    if relative.parts[0] in _EXCLUDED_TOP_LEVEL:
        return True
    if any(part in _EXCLUDED_DIR_NAMES or part.startswith(".venv") for part in relative.parts):
        return True
    return relative.suffix == ".pyc"


def iter_python_files(root: Path, bases: Sequence[str]) -> list[Path]:
    files: list[Path] = []
    for base in bases:
        location = root / base
        if location.is_file() and location.suffix == ".py":
            files.append(location)
            continue
        if location.is_dir():
            files.extend(
                path for path in location.rglob("*.py") if not _is_excluded(path.relative_to(root))
            )
    return sorted(set(files), key=lambda path: path.relative_to(root).as_posix())


def mypy_scope(root: Path) -> list[Path]:
    files = iter_python_files(root, ("src", "scripts", "benchmarks"))
    for name in ("pyproject.toml", "quality_toolchain.json", "requirements-quality.lock"):
        path = root / name
        if path.exists():
            files.append(path)
    return sorted(set(files), key=lambda path: path.relative_to(root).as_posix())


def ruff_scope(root: Path) -> list[Path]:
    files = [path for path in root.rglob("*.py") if not _is_excluded(path.relative_to(root))]
    for name in ("pyproject.toml", "quality_toolchain.json", "requirements-quality.lock"):
        path = root / name
        if path.exists():
            files.append(path)
    return sorted(set(files), key=lambda path: path.relative_to(root).as_posix())


def release_scope(root: Path, *, for_sha_manifest: bool = False) -> list[Path]:
    excludes = _SHA_SELF_EXCLUDES if for_sha_manifest else _RELEASE_SELF_EXCLUDES
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        relative_name = relative.as_posix()
        if _is_excluded(relative) or relative_name in excludes:
            continue
        if relative.suffix in {".zip", ".whl", ".gz"} and relative.parts[0] == "artifacts":
            continue
        files.append(path)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hash_scope(root: Path, paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda value: value.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def relative_names(root: Path, paths: Iterable[Path]) -> list[str]:
    return [path.relative_to(root).as_posix() for path in paths]


def read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def project_version(root: Path) -> str:
    with (root / "pyproject.toml").open("rb") as handle:
        payload = tomllib.load(handle)
    project = payload.get("project")
    if not isinstance(project, dict):
        raise TypeError("pyproject.toml [project] must be a table")
    value = project.get("version")
    if not isinstance(value, str):
        raise TypeError("pyproject.toml project.version must be a string")
    return value


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
