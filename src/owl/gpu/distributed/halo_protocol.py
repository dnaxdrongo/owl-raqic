"""Generate distributed halo dependencies from the scientific stage contract.

The protocol derives a conservative dependency set from the checked-in stage
contract and the active device-state registry. Ambiguous aggregate tokens such
as ``cell_fields`` expand to all spatial device arrays so boundary certification
cannot omit a neighbor-dependent field.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from owl.science.stage_contract import STAGE_CONTRACTS

_NON_ARRAY_TOKENS = {
    "patches",
    "global_state",
    "parent_bias",
    "parent_phase",
    "bounded_fields",
    "cell_fields",
    "traits",
    "genome",
    "pre_utilities",
    "pre_authority",
}

# Tokens in the stage contract that represent a family of concrete state
# arrays. These aliases are kept here rather than in communication call sites.
_TOKEN_ALIASES: dict[str, tuple[str, ...]] = {
    "traits": (
        "movement_rate",
        "metabolic_efficiency",
        "toxin_resistance",
        "communication_efficiency",
        "repair_rate",
        "reproduction_threshold",
        "mutation_rate",
        "integration_capacity",
        "coupling_strength",
        "receive_sensitivity",
        "digestion_efficiency",
    ),
    "genome": (
        "movement_rate",
        "metabolic_efficiency",
        "toxin_resistance",
        "communication_efficiency",
        "repair_rate",
        "reproduction_threshold",
        "mutation_rate",
        "integration_capacity",
        "coupling_strength",
        "receive_sensitivity",
        "digestion_efficiency",
    ),
    "bounded_fields": ("health", "resource", "memory", "boundary", "integration"),
    "pre_authority": ("pre_authority", "_authority_bool"),
}


@dataclass(frozen=True)
class HaloProtocol:
    phase: str
    fields: tuple[str, ...]
    halo_width: int
    source_stages: tuple[str, ...]
    conservative_expansion: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "owl.distributed.halo.v2",
            "phase": self.phase,
            "fields": list(self.fields),
            "halo_width": int(self.halo_width),
            "source_stages": list(self.source_stages),
            "conservative_expansion": bool(self.conservative_expansion),
            "passed": bool(self.fields) and self.halo_width >= 0,
        }


def _spatial_arrays(ds: Any) -> tuple[str, ...]:
    height = int(ds.health.shape[0])
    out = []
    for name, array in sorted(ds.arrays.items()):
        shape: tuple[Any, ...] = tuple(getattr(array, "shape", ()))
        if len(shape) >= 2 and int(shape[0]) == height:
            out.append(name)
    return tuple(out)


def _expand_token(token: str, available: set[str], spatial: tuple[str, ...]) -> tuple[str, ...]:
    if token in {"cell_fields", "traits", "genome"}:
        aliases = _TOKEN_ALIASES.get(token)
        if aliases is None or token == "cell_fields":
            return spatial
        # Trait families are explicit when possible, but unknown/new state is
        # still covered by a conservative all-spatial expansion at the caller.
        return tuple(name for name in aliases if name in available)
    aliases = _TOKEN_ALIASES.get(token, (token,))
    return tuple(name for name in aliases if name in available)


def generate_halo_protocol(
    ds: Any,
    *,
    phase: str = "predecision",
    stage_names: Iterable[str] | None = None,
) -> HaloProtocol:
    """Derive fields and radius needed before the next synchronization point.

    ``phase='predecision'`` covers all neighbor-reading stages in a tick.  A
    caller may pass an explicit stage subset for a narrower future protocol.
    Unknown aggregate dependencies fail safely by expanding to every spatial
    array rather than silently dropping a field.
    """
    selected = set(stage_names or ())
    contracts = tuple(
        stage
        for stage in STAGE_CONTRACTS
        if (not selected or stage.name in selected) and stage.neighborhood_radius > 0
    )
    spatial = _spatial_arrays(ds)
    available = set(ds.arrays)
    fields: set[str] = set()
    conservative = False
    for stage in contracts:
        for token in stage.reads:
            expanded = _expand_token(token, available, spatial)
            if token in _NON_ARRAY_TOKENS and not expanded:
                conservative = True
                fields.update(spatial)
            else:
                fields.update(expanded)
        if any(token in {"cell_fields", "traits", "genome"} for token in stage.reads):
            conservative = True
            fields.update(spatial)
    # Occupancy and identity are always part of neighbor topology, even when a
    # future stage-contract edit forgets to name them explicitly.
    for required in ("occupancy", "parent_id", "health"):
        if required in available:
            fields.add(required)
    radius = max((int(stage.neighborhood_radius) for stage in contracts), default=0)
    return HaloProtocol(
        phase=str(phase),
        fields=tuple(sorted(fields)),
        halo_width=radius,
        source_stages=tuple(stage.name for stage in contracts),
        conservative_expansion=conservative,
    )
