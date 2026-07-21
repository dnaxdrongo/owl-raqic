from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

DEFAULT_ACTIONS = (
    "REST",
    "SENSE",
    "MOVE",
    "FEED",
    "COMMUNICATE",
    "INHIBIT",
    "INTEGRATE",
    "REPAIR",
    "REPRODUCE",
    "INGEST",
)


@dataclass(frozen=True)
class RAQICActionSet:
    names: tuple[str, ...] = DEFAULT_ACTIONS

    def __post_init__(self) -> Any:
        if len(self.names) < 2:
            raise ValueError("RAQICActionSet must contain at least two actions")
        if len(set(self.names)) != len(self.names):
            raise ValueError("action names must be unique")

    def index(self, name: str) -> int:
        return self.names.index(name)

    def __len__(self) -> int:
        return len(self.names)


@dataclass(frozen=True)
class RAQICFeatureVector:
    resource: float
    risk: float
    memory: float
    coherence: float
    phase: float
    boundary: float
    signal: float
    prediction_error: float
    parent_context: float = 0.0
    food: float = 0.0
    toxin: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.array(
            [
                self.resource,
                self.risk,
                self.memory,
                self.coherence,
                self.phase,
                self.boundary,
                self.signal,
                self.prediction_error,
                self.parent_context,
                self.food,
                self.toxin,
            ],
            dtype=float,
        )


@dataclass(frozen=True)
class RAQICFeaturePacket:
    ow_id: int
    scale_id: int
    tick: int
    feature_bins: Mapping[str, int | float]
    adelic_codes: Mapping[str, int | float] = field(default_factory=dict)
    parent_intention: np.ndarray | None = None
    authority_mask: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __eq__(self, other: Any) -> Any:
        if not isinstance(other, RAQICFeaturePacket):
            return False
        arrays_equal = (
            (self.parent_intention is None and other.parent_intention is None)
            or (
                self.parent_intention is not None
                and other.parent_intention is not None
                and np.array_equal(self.parent_intention, other.parent_intention)
            )
        ) and (
            (self.authority_mask is None and other.authority_mask is None)
            or (
                self.authority_mask is not None
                and other.authority_mask is not None
                and np.array_equal(self.authority_mask, other.authority_mask)
            )
        )
        return (
            self.ow_id == other.ow_id
            and self.scale_id == other.scale_id
            and self.tick == other.tick
            and dict(self.feature_bins) == dict(other.feature_bins)
            and dict(self.adelic_codes) == dict(other.adelic_codes)
            and dict(self.metadata) == dict(other.metadata)
            and arrays_equal
        )


@dataclass(frozen=True)
class RAQICFeatureBatch:
    features: np.ndarray
    action_mask: np.ndarray | None
    scale_ids: np.ndarray
    ow_ids: np.ndarray
    parent_ids: np.ndarray | None = None


@dataclass(frozen=True)
class RAQICDecisionResult:
    ow_id: int
    scale_id: int
    tick: int
    action_probabilities: np.ndarray
    sampled_action: int | None
    sampled_action_name: str | None
    measurement_record: dict[str, Any]
    simulator_metadata: dict[str, Any]
    recovery_checks: dict[str, Any]

    @property
    def probabilities(self) -> np.ndarray:
        return self.action_probabilities
