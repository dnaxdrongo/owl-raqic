from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentCondition:
    name: str
    config: str
    full_replay: bool = False


@dataclass(frozen=True)
class ExperimentManifest:
    schema_version: str
    name: str
    ticks: int
    seed: int
    conditions: tuple[ExperimentCondition, ...]
    seeds: tuple[int, ...] = ()
    recording_tier: str = "analysis_full"
    max_concurrent_conditions: int = 1
    progress_every: int = 5

    def validate(self) -> None:
        if self.schema_version != "owl.experiment.v1":
            raise ValueError(f"unsupported experiment manifest: {self.schema_version}")
        if self.ticks < 1:
            raise ValueError("experiment ticks must be positive")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("experiment seeds must be nonempty and unique")
        if self.seed != self.seeds[0]:
            raise ValueError("legacy seed alias must equal the first registered seed")
        names = [item.name for item in self.conditions]
        if len(names) != len(set(names)):
            raise ValueError("experiment condition names must be unique")
        if sum(item.full_replay for item in self.conditions) != 1:
            raise ValueError("exactly one experiment condition must receive full replay")
        if not 1 <= self.max_concurrent_conditions <= len(self.conditions) * len(
            self.seeds
        ):
            raise ValueError("max_concurrent_conditions must fit the world count")
        if self.progress_every < 1:
            raise ValueError("progress_every must be positive")

    @classmethod
    def load(cls, path: str | Path) -> ExperimentManifest:
        source = Path(path)
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        raw_seeds = raw.get("seeds")
        seeds: tuple[int, ...]
        if raw_seeds is None:
            seeds = (int(raw["seed"]),)
        else:
            if not isinstance(raw_seeds, list) or not raw_seeds:
                raise ValueError("experiment seeds must be a nonempty list")
            seeds = tuple(int(value) for value in raw_seeds)
            if "seed" in raw and int(raw["seed"]) != seeds[0]:
                raise ValueError("seed and seeds[0] disagree")
        manifest = cls(
            schema_version=str(raw.get("schema_version", "")),
            name=str(raw["name"]),
            ticks=int(raw["ticks"]),
            seed=seeds[0],
            conditions=tuple(
                ExperimentCondition(
                    name=str(item["name"]),
                    config=str((source.parent / str(item["config"])).resolve()),
                    full_replay=bool(item.get("full_replay", False)),
                )
                for item in raw["conditions"]
            ),
            seeds=seeds,
            recording_tier=str(raw.get("recording_tier", "analysis_full")),
            max_concurrent_conditions=int(raw.get("max_concurrent_conditions", 1)),
            progress_every=int(raw.get("progress_every", 5)),
        )
        manifest.validate()
        return manifest

    def stable_hash(self) -> str:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "name": self.name,
            "ticks": self.ticks,
            "seed": self.seed,
            "seeds": list(self.seeds),
            "conditions": [item.__dict__ for item in self.conditions],
            "recording_tier": self.recording_tier,
            "max_concurrent_conditions": self.max_concurrent_conditions,
            "progress_every": self.progress_every,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
