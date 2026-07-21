"""Register metadata and derive paired branch seeds for counterfactual rollouts.

Candidate branches share named exogenous random streams to implement
common-random-number variance reduction while keeping focal policy draws
branch-local. See Nelson and Matejcik (1995) in ``docs/REFERENCES.md`` [R38].
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import asdict, dataclass
from typing import Any

from owl.counterfactual.schema import stable_id
from owl_raqic.random_contract import RNGStream

RNG_REGISTRY_VERSION = "owl.counterfactual-rng-registry.v1"
BRANCH_SEED_VERSION = "owl.branch-seed-sha256.v1"


@dataclass(frozen=True)
class RNGRegistryEntry:
    stream_id: int
    name: str
    source_owners: tuple[str, ...]
    draw_slots: tuple[int, ...]
    role: str
    branch_behavior: str
    active: bool = True


RNG_REGISTRY: tuple[RNGRegistryEntry, ...] = (
    RNGRegistryEntry(
        100,
        "RAQIC_READOUT",
        ("owl_raqic.random_contract:categorical",),
        (0,),
        "focal_policy",
        "paired seed; action may diverge after source tick",
    ),
    RNGRegistryEntry(
        200,
        "MOVEMENT_TIE",
        ("owl.science.action_contract:movement_plan",),
        (0,),
        "exogenous_conflict",
        "common random number",
    ),
    RNGRegistryEntry(
        250,
        "INGESTION_OUTCOME",
        ("owl.science.ingestion_contract",),
        (),
        "exogenous_consequence",
        "common random number",
    ),
    RNGRegistryEntry(
        300,
        "REPRODUCTION_TIE",
        ("owl.science.action_contract:reproduction_plan",),
        (0, 1, 2),
        "exogenous_consequence",
        "common random number",
    ),
    RNGRegistryEntry(310, "REPRODUCTION_GATE", (), (), "declared", "reserved", False),
    RNGRegistryEntry(320, "REPRODUCTION_SITE", (), (), "declared", "reserved", False),
    RNGRegistryEntry(330, "REPRODUCTION_MUTATION", (), (), "declared", "reserved", False),
    RNGRegistryEntry(
        400,
        "TOPOLOGY_TIE",
        ("owl.science.topology_contract",),
        (),
        "exogenous_conflict",
        "common random number",
    ),
    RNGRegistryEntry(
        500,
        "ENVIRONMENT_NOISE",
        ("owl.science.environment_contract",),
        (),
        "exogenous_environment",
        "common random number",
    ),
    RNGRegistryEntry(
        600,
        "PHASE_NOISE",
        ("owl.gpu.stages.phase_gpu",),
        (),
        "exogenous_phase",
        "common random number",
    ),
    RNGRegistryEntry(
        700,
        "QISKIT_READOUT",
        ("owl_raqic.qiskit_backend",),
        (),
        "focal_policy",
        "fail closed unless branch-local certified",
    ),
)


def _validate_registry() -> None:
    actual = {item.name: int(item) for item in RNGStream}
    declared = {item.name: item.stream_id for item in RNG_REGISTRY}
    if actual != declared:
        raise RuntimeError(f"counterfactual RNG registry mismatch: {actual!r} != {declared!r}")


_validate_registry()


def branch_seed(factual_seed: int, source_state_id: str, repeat_index: int) -> int:
    """Derive one uint64 seed shared by every candidate in a matched repeat."""
    if repeat_index < 0:
        raise ValueError("repeat_index must be non-negative")
    fields = (BRANCH_SEED_VERSION, str(factual_seed), source_state_id, str(repeat_index))
    payload = b"".join(struct.pack(">Q", len(item.encode())) + item.encode() for item in fields)
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def registry_manifest() -> dict[str, Any]:
    entries = [asdict(entry) for entry in RNG_REGISTRY]
    digest = stable_id("rng_registry", RNG_REGISTRY_VERSION, entries)
    return {
        "schema_version": RNG_REGISTRY_VERSION,
        "entries": entries,
        "registry_digest": digest,
    }
