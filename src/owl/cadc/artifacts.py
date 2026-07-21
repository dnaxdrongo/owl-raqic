"""Atomic, checksum-bound helpers for CADC-MORE 2 artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def canonical_json(payload: Any) -> bytes:
    """Encode JSON deterministically for content-addressed identities."""
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str) + "\n").encode()


def sha256_file(path: str | Path) -> str:
    """Return the streaming SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: str | Path, payload: Any) -> None:
    """Write a JSON object through an atomic same-directory replacement."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True, default=str).encode())
        handle.write(b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def atomic_manifest(path: str | Path, payload: Any) -> None:
    """Write an atomic JSON manifest with a digest over its undigested body."""
    body = dict(payload)
    body.pop("manifest_digest", None)
    body["manifest_digest"] = hashlib.sha256(canonical_json(body)).hexdigest()
    atomic_json(path, body)


@dataclass(frozen=True)
class ModelArtifactReceipt:
    """Checksum and provenance receipt for one model artifact."""

    path: str
    sha256: str
    bytes: int
    schema_version: str
    source_sha256: str
    config_sha256: str
    status: str


def receipt_for(
    path: str | Path,
    *,
    schema_version: str,
    source_sha256: str,
    config_sha256: str,
    status: str = "passed",
) -> ModelArtifactReceipt:
    """Create a provenance-bound receipt for an existing artifact."""
    value = Path(path)
    return ModelArtifactReceipt(
        path=value.name,
        sha256=sha256_file(value),
        bytes=value.stat().st_size,
        schema_version=schema_version,
        source_sha256=source_sha256,
        config_sha256=config_sha256,
        status=status,
    )


def write_receipt(path: str | Path, receipt: ModelArtifactReceipt) -> None:
    """Persist one model artifact receipt atomically."""
    atomic_json(path, asdict(receipt))


def verify_receipt(root: str | Path, receipt: ModelArtifactReceipt) -> None:
    """Verify an artifact still matches its registered checksum."""
    target = Path(root) / receipt.path
    if not target.is_file():
        raise FileNotFoundError(target)
    actual = sha256_file(target)
    if actual != receipt.sha256:
        raise RuntimeError(f"artifact checksum mismatch: {target}")


def write_failure_receipt(
    path: str | Path,
    *,
    stage: str,
    exception: BaseException,
    source_sha256: str,
    config_sha256: str,
) -> None:
    """Materialize a typed failed-closed receipt for an interrupted component."""
    atomic_json(
        path,
        {
            "schema_version": "owl.cadc.phase4-failure.v1",
            "passed": False,
            "stage": stage,
            "exception_type": type(exception).__name__,
            "message": str(exception),
            "source_sha256": source_sha256,
            "config_sha256": config_sha256,
        },
    )


def write_model_card(path: str | Path, fields: dict[str, Any]) -> None:
    """Write the mandatory human-readable model-card fields."""
    required = {
        "model_name",
        "role",
        "source_sha256",
        "dataset_sha256",
        "feature_schema_digest",
        "outcome_registry_digest",
        "split_registry_digest",
        "intended_use",
        "forbidden_use",
        "metrics",
        "calibration",
        "support",
        "negative_controls",
        "limitations",
    }
    missing = sorted(required.difference(fields))
    if missing:
        raise KeyError(f"model card fields missing: {missing}")
    lines = [f"# {fields['model_name']}", ""]
    for key in sorted(fields):
        if key == "model_name":
            continue
        title = key.replace("_", " ").title()
        value = fields[key]
        rendered = json.dumps(value, indent=2, sort_keys=True, default=str)
        lines.extend((f"## {title}", "", f"```json\n{rendered}\n```", ""))
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary, destination)
