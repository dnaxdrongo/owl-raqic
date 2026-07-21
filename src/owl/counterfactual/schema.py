"""Define versioned counterfactual schemas, registries, and deterministic identifiers."""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import IntEnum, StrEnum
from typing import Any

COUNTERFACTUAL_SCHEMA_VERSION = "owl.cadc.counterfactual.v1"
COUNTERFACTUAL_CERTIFICATE_VERSION = "owl.cadc.phase3-target-certificate.v1"
STATE_HASH_VERSION = "owl.device-state-merkle-sha256.v1"
ID_VERSION = "owl.counterfactual-id-sha256.v1"


class BranchStatus(StrEnum):
    REQUESTED = "requested"
    RUNNING = "running"
    COMPLETED = "completed"
    NONEXECUTABLE = "nonexecutable"
    FAILED = "failed"
    OVERFLOW = "overflow"


class CandidateStatus(StrEnum):
    EXECUTABLE = "executable"
    NONEXECUTABLE = "nonexecutable"
    UNSUPPORTED_ENGINE_MODE = "unsupported_engine_mode"


class NonExecutableReason(StrEnum):
    NOT_ALIVE = "not_alive"
    POLICY_ILLEGAL = "policy_illegal"
    TRANSITION_DISABLED = "transition_disabled"
    INSUFFICIENT_RESOURCE = "insufficient_resource"
    NO_TARGET = "no_target"
    NO_EXECUTABLE_DIRECTION = "no_executable_direction"
    SOURCE_JOIN_MISMATCH = "source_join_mismatch"


class SourceBoundary(StrEnum):
    POST_SELECTION_PRE_ACTIONS = "post_selection_pre_actions"


class TargetValidation(StrEnum):
    NOT_APPLICABLE = "not_applicable"
    MATCHED = "matched"
    MISMATCH = "mismatch"


class HashStatus(StrEnum):
    MATCHED = "matched"
    DIFFERENT_EXPECTED = "different_expected"
    FAILED = "failed"


class HorizonStatus(StrEnum):
    COMPLETED = "completed"
    FOCAL_DEAD = "focal_dead"
    FOCAL_ABSENT = "focal_absent"
    BRANCH_FAILED = "branch_failed"


class DeathEvidence(IntEnum):
    NONE = 0
    DEAD = 1
    ABSENT_AMBIGUOUS = 2


class PairStatus(StrEnum):
    COMPLETE = "complete"
    ONE_NONEXECUTABLE = "one_nonexecutable"
    FAILED = "failed"


class ActionExecutionMode(StrEnum):
    HIGH_LEVEL_COMPILED = "high_level_compiled"
    DIRECT = "direct"


@dataclass(frozen=True)
class TableContract:
    name: str
    grain: str
    required_foreign_keys: tuple[str, ...]
    deterministic_order: tuple[str, ...]


TABLE_CONTRACTS: tuple[TableContract, ...] = (
    TableContract("source_states", "source world", (), ("source_tick", "source_state_id")),
    TableContract(
        "source_decisions",
        "sampled focal decision",
        ("source_state_id",),
        ("source_tick", "decision_sequence", "source_decision_id"),
    ),
    TableContract(
        "branch_attempts",
        "requested decision/action/repeat",
        ("source_state_id", "source_decision_id"),
        ("source_tick", "decision_sequence", "repeat_index", "forced_action", "branch_id"),
    ),
    TableContract(
        "counterfactual_micro_rollouts",
        "executable branch/horizon",
        ("branch_id", "source_decision_id"),
        ("source_tick", "decision_sequence", "repeat_index", "forced_action", "horizon"),
    ),
    TableContract(
        "branch_events",
        "exact branch event",
        ("branch_id",),
        ("branch_id", "branch_tick", "event_code", "event_id"),
    ),
    TableContract(
        "branch_event_summaries",
        "branch/horizon/event summary",
        ("branch_id",),
        ("branch_id", "horizon", "event_code"),
    ),
    TableContract(
        "branch_contributions",
        "branch/tick/stage/field contribution",
        ("branch_id",),
        ("branch_id", "branch_tick", "stage_code", "contribution_code", "field"),
    ),
    TableContract(
        "candidate_pairs",
        "decision/repeat/action pair/horizon",
        ("source_decision_id",),
        ("source_decision_id", "repeat_index", "action_a", "action_b", "horizon"),
    ),
    TableContract(
        "nonexecutable_candidates",
        "candidate without branch",
        ("source_decision_id",),
        ("source_decision_id", "repeat_index", "forced_action"),
    ),
)


def _canonical(value: Any) -> bytes:
    if value is None:
        return b"n"
    if isinstance(value, bool):
        return b"b" + (b"1" if value else b"0")
    if isinstance(value, int):
        return b"i" + str(value).encode("ascii")
    if isinstance(value, float):
        return b"f" + struct.pack(">d", value)
    if isinstance(value, bytes):
        return b"y" + value
    if isinstance(value, str):
        return b"s" + value.encode("utf-8")
    if isinstance(value, Mapping):
        entries = []
        for key in sorted(value, key=lambda item: str(item)):
            entries.append(_length_prefixed((_canonical(str(key)), _canonical(value[key]))))
        return b"m" + b"".join(entries)
    if isinstance(value, Sequence):
        return b"q" + _length_prefixed(tuple(_canonical(item) for item in value))
    raise TypeError(f"unsupported stable-ID component: {type(value).__name__}")


def _length_prefixed(parts: Sequence[bytes]) -> bytes:
    return b"".join(struct.pack(">Q", len(part)) + part for part in parts)


def stable_id(kind: str, *components: Any) -> str:
    """Return a restart- and ordering-stable length-prefixed SHA-256 ID."""
    payload = _length_prefixed(
        (_canonical(ID_VERSION), _canonical(kind), *(_canonical(item) for item in components))
    )
    return hashlib.sha256(payload).hexdigest()


def source_state_id(*components: Any) -> str:
    return stable_id("source_state", *components)


def source_decision_id(*components: Any) -> str:
    return stable_id("source_decision", *components)


def branch_id(*components: Any) -> str:
    return stable_id("branch", *components)


def pair_id(*components: Any) -> str:
    return stable_id("pair", *components)


def event_id(*components: Any) -> str:
    return stable_id("event", *components)


def contribution_id(*components: Any) -> str:
    return stable_id("contribution", *components)


def part_id(*components: Any) -> str:
    return stable_id("part", *components)


def schema_manifest() -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema_version": COUNTERFACTUAL_SCHEMA_VERSION,
        "id_version": ID_VERSION,
        "state_hash_version": STATE_HASH_VERSION,
        "tables": [asdict(contract) for contract in TABLE_CONTRACTS],
        "registries": {
            "branch_status": [item.value for item in BranchStatus],
            "candidate_status": [item.value for item in CandidateStatus],
            "nonexecutable_reason": [item.value for item in NonExecutableReason],
            "source_boundary": [item.value for item in SourceBoundary],
            "target_validation": [item.value for item in TargetValidation],
            "hash_status": [item.value for item in HashStatus],
            "horizon_status": [item.value for item in HorizonStatus],
            "death_evidence": [int(item) for item in DeathEvidence],
            "pair_status": [item.value for item in PairStatus],
            "action_execution_mode": [item.value for item in ActionExecutionMode],
        },
    }
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["schema_digest"] = hashlib.sha256(encoded).hexdigest()
    return manifest


COUNTERFACTUAL_SCHEMA_DIGEST = str(schema_manifest()["schema_digest"])
