"""Versioned contracts for Context-Sensitive Adaptive Decision Competence evidence.

This module contains metadata only.  It deliberately has no PyArrow or CuPy
dependency and never participates in scientific decision or transition code.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import IntEnum, StrEnum
from typing import Any

from owl.core.actions import Action

CADC_SCHEMA_VERSION = "owl.cadc.factual.v1"
CADC_REGISTRY_VERSION = "owl.cadc.registries.v1"
CADC_ACTION_TRANSITION_SCHEMA_VERSION = "owl.cadc.factual.v2"
CADC_ACTION_TRANSITION_REGISTRY_VERSION = "owl.cadc.registries.v2"
CADC_ACTION_COUNT = 22
ABSENT_INT = -1
NONE_CODE = 0


class CADCProfile(StrEnum):
    COMPACT = "compact"
    EXACT = "exact"


class PerspectiveCode(IntEnum):
    AGENT = 1
    ORACLE_LOCAL = 2
    ORACLE_PATCH = 3
    ORACLE_GLOBAL = 4
    POST_DECISION = 5
    DIAGNOSTIC = 6


class TargetKind(IntEnum):
    NONE = 0
    SELF = 1
    CELL = 2
    OW = 3
    OW_SET = 4
    EMPTY_NEIGHBOR_SET = 5
    LOCAL_BROADCAST = 6
    SEMANTIC_DIRECTION_SET = 7


class ReasonCode(IntEnum):
    NONE = 0
    NOT_ALIVE = 1
    POLICY_ILLEGAL = 2
    DISABLED = 3
    INSUFFICIENT_RESOURCE = 4
    NO_TARGET = 5
    BOUNDARY_BLOCKED = 6
    OBSTACLE = 7
    OCCUPIED = 8
    CONFLICT_LOST = 9
    STAGE_NOT_ATTEMPTED = 10
    NO_EXECUTION_CONTRACT = 11
    STOCHASTIC_GATE_FAILED = 12
    NO_EFFECT = 13
    CAUSE_AMBIGUOUS = 14
    ACTIVE_SENSE_NO_NEW_INFORMATION = 15
    NO_EXECUTABLE_DIRECTION = 16


class CADCEventCode(IntEnum):
    MOVEMENT_ATTEMPT = 100
    MOVEMENT_SUCCESS = 101
    MOVEMENT_REJECTION = 102
    COLLISION = 110
    INHIBITION = 111
    FEEDING = 120
    INGESTION = 121
    REPAIR = 130
    INTEGRATION = 131
    SIGNAL_EMIT = 140
    SIGNAL_RECEPTION = 141
    BIRTH = 150
    MERGE = 160
    SPLIT = 161
    EXPULSION = 162
    STARVATION_EVIDENCE = 170
    TOXIN_DAMAGE_EVIDENCE = 171
    DEATH = 180
    ACTION_TARGET_ACQUIRED = 190
    ACTIVE_SENSE_ATTEMPT = 191
    ACTIVE_SENSE_SUCCESS = 192


class ContributionCode(IntEnum):
    ENVIRONMENT = 10
    MOVEMENT = 20
    COLLISION_INHIBITION = 21
    FEEDING = 22
    REPAIR_INTEGRATION = 23
    COMMUNICATION = 24
    REPRODUCTION = 25
    TOPOLOGY = 26
    ACTIVE_SENSE = 27
    METABOLISM_TOXIN = 30
    MEMORY_TRUST = 31
    DEATH_CLEANUP = 32
    CLIPPING = 33
    IDENTITY_TRANSFER = 90
    RESIDUAL = 99


class CaptureStageCode(IntEnum):
    TICK_OPEN = 10
    POST_SENSING = 20
    POST_CONTEXT = 30
    PRE_CHOICE = 40
    POST_SELECTION = 50
    MOVEMENT = 60
    COLLISION_INHIBITION = 61
    FEEDING = 62
    REPAIR_INTEGRATE = 63
    COMMUNICATION = 64
    REPRODUCTION = 65
    TOPOLOGY = 66
    ACTIVE_SENSE = 67
    METABOLISM_DAMAGE = 70
    MEMORY = 71
    TRUST_INTEGRATION = 72
    DEATH = 73
    CLIP = 74
    TICK_COMMIT = 80


@dataclass(frozen=True)
class RegistryEntry:
    code: int
    label: str
    stage: str
    description: str
    deprecated: bool = False


@dataclass(frozen=True)
class TableContract:
    name: str
    grain: str
    required: bool
    rows_per_decision: int | None = None


TABLE_CONTRACTS: tuple[TableContract, ...] = (
    TableContract("decisions", "decision", True),
    TableContract("agent_context", "decision", True),
    TableContract("oracle_context", "decision", True),
    TableContract("candidates", "decision_action", True, CADC_ACTION_COUNT),
    TableContract("execution", "decision", True),
    TableContract("events", "event", True),
    TableContract("contributions", "decision_variable_channel", True),
    TableContract("information", "information_action", False),
    TableContract("dense_context", "decision_exact_local_tensor", False),
)

ACTION_TRANSITION_TABLE_CONTRACTS: tuple[TableContract, ...] = (
    *TABLE_CONTRACTS,
    TableContract("action_directions", "decision_action_family_direction", True, 16),
)

_V1_REASON_CODES = tuple(code for code in ReasonCode if int(code) <= 14)
_V1_EVENT_CODES = tuple(code for code in CADCEventCode if int(code) <= 180)
_V1_CONTRIBUTION_CODES = tuple(
    code for code in ContributionCode if int(code) != int(ContributionCode.ACTIVE_SENSE)
)
_V1_CAPTURE_STAGE_CODES = tuple(
    code for code in CaptureStageCode if int(code) != int(CaptureStageCode.ACTIVE_SENSE)
)


def _entries(enum: type[IntEnum], *, stage: str, description: str) -> list[RegistryEntry]:
    return [
        RegistryEntry(int(member), member.name.lower(), stage, description)
        for member in enum
    ]


def _member_entries(
    members: tuple[IntEnum, ...], *, stage: str, description: str
) -> list[RegistryEntry]:
    return [
        RegistryEntry(int(member), member.name.lower(), stage, description)
        for member in members
    ]


def action_names() -> tuple[str, ...]:
    """Return the immutable scientific action axis after validating its values."""
    ordered = tuple(action.name for action in Action)
    values = tuple(int(action) for action in Action)
    if len(ordered) != CADC_ACTION_COUNT or values != tuple(range(CADC_ACTION_COUNT)):
        raise RuntimeError("CADC schema requires the immutable contiguous 22-action contract")
    return ordered


def schema_manifest(*, action_transitions: bool = False) -> dict[str, Any]:
    """Return deterministic schema metadata suitable for run manifests."""
    reason_codes = tuple(ReasonCode) if action_transitions else _V1_REASON_CODES
    event_codes = tuple(CADCEventCode) if action_transitions else _V1_EVENT_CODES
    contribution_codes = (
        tuple(ContributionCode) if action_transitions else _V1_CONTRIBUTION_CODES
    )
    capture_codes = (
        tuple(CaptureStageCode) if action_transitions else _V1_CAPTURE_STAGE_CODES
    )
    manifest: dict[str, Any] = {
        "schema_version": (
            CADC_ACTION_TRANSITION_SCHEMA_VERSION
            if action_transitions
            else CADC_SCHEMA_VERSION
        ),
        "registry_version": (
            CADC_ACTION_TRANSITION_REGISTRY_VERSION
            if action_transitions
            else CADC_REGISTRY_VERSION
        ),
        "action_count": CADC_ACTION_COUNT,
        "action_names": list(action_names()),
        "row_order": "tick,source_flat_c,action_index",
        "sentinels": {"absent_int": ABSENT_INT, "none_code": NONE_CODE},
        "tables": [
            asdict(item)
            for item in (
                ACTION_TRANSITION_TABLE_CONTRACTS
                if action_transitions
                else TABLE_CONTRACTS
            )
        ],
        "registries": {
            "reason": [
                asdict(item)
                for item in _member_entries(
                    reason_codes,
                    stage="versioned_by_entry",
                    description="pre-choice or execution classification",
                )
            ],
            "event": [
                asdict(item)
                for item in _member_entries(
                    event_codes,
                    stage="versioned_by_entry",
                    description="typed factual event family",
                )
            ],
            "contribution": [
                asdict(item)
                for item in _member_entries(
                    contribution_codes,
                    stage="versioned_by_entry",
                    description="tracked state-change channel",
                )
            ],
            "capture_stage": [
                asdict(item)
                for item in _member_entries(
                    capture_codes,
                    stage="runtime_boundary",
                    description="immutable observational capture boundary",
                )
            ],
        },
    }
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["schema_digest"] = hashlib.sha256(payload).hexdigest()
    return manifest


CADC_SCHEMA_DIGEST = str(schema_manifest()["schema_digest"])
CADC_ACTION_TRANSITION_SCHEMA_DIGEST = str(
    schema_manifest(action_transitions=True)["schema_digest"]
)


def schema_contract_for_config(cfg: Any) -> tuple[str, str, tuple[int, ...], tuple[int, ...]]:
    """Return schema identity and fixed recorder registries for one run."""
    enabled = bool(getattr(getattr(cfg, "action_transitions", None), "enabled", False))
    if enabled:
        return (
            CADC_ACTION_TRANSITION_SCHEMA_VERSION,
            CADC_ACTION_TRANSITION_SCHEMA_DIGEST,
            tuple(int(code) for code in CADCEventCode),
            tuple(int(code) for code in ContributionCode),
        )
    return (
        CADC_SCHEMA_VERSION,
        CADC_SCHEMA_DIGEST,
        tuple(int(code) for code in _V1_EVENT_CODES),
        tuple(int(code) for code in _V1_CONTRIBUTION_CODES),
    )
