from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from owl.runtime.certificates import EnvironmentIdentity, compare_identities


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def environment_fingerprint() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable": sys.executable,
    }


@dataclass(frozen=True)
class CertificationCompatibility:
    certificate: Path
    passed: bool
    failures: tuple[str, ...]

    def require(self) -> None:
        if not self.passed:
            raise RuntimeError("no compatible GPU certification: " + "; ".join(self.failures))


class CertificationStore:
    """Exact-identity lookup for production certification records.

     deliberately removes the broad compatibility acceptance rule.  A graph,
    Qiskit, or distributed certificate is valid only for the source, config,
    execution plan, scientific contract, environment, and devices that
    produced it.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def records(self) -> list[Path]:
        if not self.directory.exists():
            return []
        return sorted(self.directory.glob("**/*.json"))

    @staticmethod
    def _identity_from_data(data: dict[str, Any]) -> EnvironmentIdentity | None:
        raw = data.get("identity")
        if not isinstance(raw, dict):
            return None
        required = {
            "source_sha256",
            "config_sha256",
            "plan_sha256",
            "scientific_contract_version",
            "scientific_contract_sha256",
            "python_version",
            "package_versions",
            "platform",
            "driver_version",
            "cuda_runtime_version",
            "gpu_devices",
            "nccl_version",
        }
        if not required.issubset(raw):
            return None
        payload = dict(raw)
        payload["gpu_devices"] = tuple(dict(item) for item in payload.get("gpu_devices", ()))
        return EnvironmentIdentity(**payload)

    def require_compatible(
        self,
        plan: Any,
        *,
        identity: EnvironmentIdentity | None = None,
        config_hash: str | None = None,
    ) -> Path:
        failures: list[str] = []
        for path in reversed(self.records()):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                failures.append(f"{path}: unreadable ({exc})")
                continue
            if not bool(data.get("passed")):
                continue
            actual_identity = self._identity_from_data(data)
            if identity is not None:
                if actual_identity is None:
                    failures.append(f"{path}: missing v0.9.2 identity")
                    continue
                mismatch = compare_identities(identity, actual_identity)
                if mismatch:
                    failures.append(f"{path}: " + "; ".join(mismatch[:3]))
                    continue
                return path

            # The compatibility path still requires the complete execution plan.
            # hash and exact config hash. This path cannot authorize a
            # production marker because it lacks the full environment identity.
            certified_plan_hash = data.get("plan_sha256") or (
                (data.get("identity") or {}).get("plan_sha256")
            )
            expected_plan_hash = getattr(plan, "plan_hash", None)
            if not certified_plan_hash or certified_plan_hash != expected_plan_hash:
                failures.append(f"{path}: plan hash mismatch or missing")
                continue
            certified_config = data.get("config_sha256") or (
                (data.get("identity") or {}).get("config_sha256")
            )
            if config_hash is None or certified_config != config_hash:
                failures.append(f"{path}: config hash mismatch or missing")
                continue
            return path
        message = "no passed certification record matched the exact requested identity"
        if failures:
            message += "; " + "; ".join(failures[-3:])
        raise RuntimeError(message)
