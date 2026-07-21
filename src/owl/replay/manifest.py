from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SUPPORTED_REPLAY_MAJOR = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ReplayManifest:
    schema_version: str
    run_id: str
    condition: str
    seed: int
    requested_ticks: int
    completed_ticks: int
    world_shape: tuple[int, int]
    boundary_mode: str
    recording_tier: str
    source_sha256: str
    config_sha256: str
    action_names: tuple[str, ...]
    array_fields: tuple[str, ...]
    created_at: str
    hardware: dict[str, Any] = field(default_factory=dict)
    qiskit_execution: dict[str, Any] = field(default_factory=dict)
    materialization_mode: str = "inline"
    materialization_state: str = "complete"
    columnar_schema_digest: str = "unknown"
    claims_boundary: str = (
        "Replay data is simulation evidence and interpretability output; it does not prove "
        "consciousness, quantum advantage, or empirical truth of a theory."
    )

    @property
    def major_version(self) -> int:
        tail = self.schema_version.rsplit("v", 1)[-1]
        return int(tail.split(".", 1)[0])

    def validate(self) -> None:
        if self.major_version != SUPPORTED_REPLAY_MAJOR:
            raise ValueError(
                f"unsupported replay schema major {self.major_version}; "
                f"supported={SUPPORTED_REPLAY_MAJOR}"
            )
        if self.completed_ticks < 0 or self.completed_ticks > self.requested_ticks:
            raise ValueError("invalid completed tick range")
        if min(self.world_shape) <= 0:
            raise ValueError("world_shape must be positive")
        if not self.array_fields:
            raise ValueError("replay manifest contains no array fields")
        if self.materialization_mode not in {"inline", "deferred"}:
            raise ValueError(f"unknown materialization mode: {self.materialization_mode}")
        if self.materialization_state not in {"complete", "pending", "materializing", "failed"}:
            raise ValueError(f"unknown materialization state: {self.materialization_state}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "condition": self.condition,
            "seed": self.seed,
            "requested_ticks": self.requested_ticks,
            "completed_ticks": self.completed_ticks,
            "world_shape": list(self.world_shape),
            "boundary_mode": self.boundary_mode,
            "recording_tier": self.recording_tier,
            "source_sha256": self.source_sha256,
            "config_sha256": self.config_sha256,
            "action_names": list(self.action_names),
            "array_fields": list(self.array_fields),
            "created_at": self.created_at,
            "hardware": self.hardware,
            "qiskit_execution": self.qiskit_execution,
            "materialization_mode": self.materialization_mode,
            "materialization_state": self.materialization_state,
            "columnar_schema_digest": self.columnar_schema_digest,
            "claims_boundary": self.claims_boundary,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ReplayManifest:
        raw_shape = tuple(int(item) for item in value["world_shape"])
        if len(raw_shape) != 2:
            raise ValueError("world_shape must contain exactly two dimensions")
        world_shape = (raw_shape[0], raw_shape[1])
        manifest = cls(
            schema_version=str(value["schema_version"]),
            run_id=str(value["run_id"]),
            condition=str(value.get("condition", "unknown")),
            seed=int(value["seed"]),
            requested_ticks=int(value["requested_ticks"]),
            completed_ticks=int(value["completed_ticks"]),
            world_shape=world_shape,
            boundary_mode=str(value.get("boundary_mode", "toroidal")),
            recording_tier=str(value.get("recording_tier", "replay_standard")),
            source_sha256=str(value.get("source_sha256", "unknown")),
            config_sha256=str(value.get("config_sha256", "unknown")),
            action_names=tuple(str(item) for item in value.get("action_names", ())),
            array_fields=tuple(str(item) for item in value.get("array_fields", ())),
            created_at=str(value.get("created_at", "unknown")),
            hardware=dict(value.get("hardware", {})),
            qiskit_execution=dict(value.get("qiskit_execution", {})),
            materialization_mode=str(value.get("materialization_mode", "inline")),
            materialization_state=str(value.get("materialization_state", "complete")),
            columnar_schema_digest=str(value.get("columnar_schema_digest", "unknown")),
            claims_boundary=str(value.get("claims_boundary", cls.claims_boundary)),
        )
        manifest.validate()
        return manifest

    @classmethod
    def load(cls, bundle: str | Path) -> ReplayManifest:
        path = Path(bundle) / "run_manifest.json"
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
