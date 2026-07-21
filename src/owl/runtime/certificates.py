from __future__ import annotations

import json
import platform
import subprocess
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib import metadata
from pathlib import Path
from typing import Any

from owl.science.contract import current_scientific_contract, sha256_canonical


def sha256_file(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def sha256_tree(
    root: str | Path,
    *,
    include: Iterable[str] = ("src", "configs", "scripts", "tests", "pyproject.toml"),
) -> str:
    root = Path(root)
    digest = sha256()
    paths: list[Path] = []
    for item in include:
        path = root / item
        if path.is_file():
            paths.append(path)
        elif path.exists():
            paths.extend(p for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    for path in sorted(paths, key=lambda p: p.relative_to(root).as_posix()):
        rel = path.relative_to(root).as_posix()
        digest.update(rel.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _package_versions(names: Iterable[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "not-installed"
    return out


def _command_text(command: list[str]) -> str:
    try:
        return subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except Exception:
        return "unavailable"


@dataclass(frozen=True)
class EnvironmentIdentity:
    source_sha256: str
    config_sha256: str
    plan_sha256: str
    scientific_contract_version: str
    scientific_contract_sha256: str
    python_version: str
    package_versions: dict[str, str]
    platform: str
    driver_version: str
    cuda_runtime_version: str
    gpu_devices: tuple[dict[str, Any], ...]
    nccl_version: str | None

    def canonical_dict(self) -> dict[str, Any]:
        return asdict(self)

    def sha256(self) -> str:
        return sha256_canonical(self.canonical_dict())


@dataclass(frozen=True)
class RunCertificate:
    identity: EnvironmentIdentity
    required_checks: dict[str, str]
    artifact_hashes: dict[str, str]
    passed: bool
    schema_version: str = "2"

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["environment_sha256"] = self.identity.sha256()
        return out

    def sha256(self) -> str:
        return sha256_canonical(self.to_dict())

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return path


def build_environment_identity(
    *,
    root: str | Path,
    config_path: str | Path,
    plan: Any,
    capabilities: Any | None = None,
) -> EnvironmentIdentity:
    root = Path(root)
    contract = current_scientific_contract(root)
    devices: tuple[dict[str, Any], ...] = ()
    details = dict(getattr(capabilities, "details", {}) or {})
    raw_devices = details.get("devices") or details.get("gpu_devices") or ()
    if isinstance(raw_devices, list):
        devices = tuple(dict(item) for item in raw_devices if isinstance(item, dict))
    driver = str(
        details.get("driver_version")
        or _command_text(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    )
    cuda_runtime = str(
        details.get("cuda_runtime_version") or details.get("cuda_runtime") or "unavailable"
    )
    nccl = details.get("nccl_version")
    return EnvironmentIdentity(
        source_sha256=sha256_tree(root),
        config_sha256=sha256_file(config_path),
        plan_sha256=str(getattr(plan, "plan_hash", plan.sha256())),
        scientific_contract_version=contract.version,
        scientific_contract_sha256=contract.sha256(),
        python_version=platform.python_version(),
        package_versions=_package_versions(
            (
                "numpy",
                "cupy-cuda12x",
                "numba",
                "qiskit",
                "qiskit-aer-gpu",
                "qiskit-machine-learning",
                "vispy",
                "pygame",
                "pydantic",
            )
        ),
        platform=platform.platform(),
        driver_version=driver,
        cuda_runtime_version=cuda_runtime,
        gpu_devices=devices,
        nccl_version=None if nccl is None else str(nccl),
    )


def compare_identities(
    expected: EnvironmentIdentity, actual: EnvironmentIdentity
) -> tuple[str, ...]:
    failures: list[str] = []
    left = expected.canonical_dict()
    right = actual.canonical_dict()
    for key in sorted(left):
        if left[key] != right.get(key):
            failures.append(f"{key}: expected {left[key]!r}, got {right.get(key)!r}")
    return tuple(failures)
