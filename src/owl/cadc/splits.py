"""Register seed- and world-grouped splits with sealed confirmatory and final-holdout seeds."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from owl.cadc.schema import SplitRole, stable_id


@dataclass(frozen=True)
class SplitAssignment:
    """One immutable world-group assignment to a role and nested folds."""
    group_id: str
    seed: int
    role: SplitRole
    outer_fold: int
    inner_fold: int


class SplitRegistry:
    """Immutable split assignments and confirmatory-seed seal."""

    def __init__(self, assignments: Sequence[SplitAssignment]) -> None:
        values = tuple(sorted(assignments, key=lambda value: value.group_id))
        group_ids = [value.group_id for value in values]
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("split registry contains duplicate group IDs")
        roles_by_seed: dict[int, set[SplitRole]] = {}
        for value in values:
            roles_by_seed.setdefault(value.seed, set()).add(value.role)
        mixed = {seed: roles for seed, roles in roles_by_seed.items() if len(roles) != 1}
        if mixed:
            raise ValueError(f"seeds cross split roles: {mixed}")
        self._assignments = values

    @property
    def assignments(self) -> tuple[SplitAssignment, ...]:
        """Return immutable assignments sorted by group identity."""
        return self._assignments

    @property
    def digest(self) -> str:
        """Return the canonical SHA-256 identity of all assignments."""
        payload = json.dumps(
            [asdict(value) for value in self._assignments],
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    def role_for_seed(self, seed: int) -> SplitRole:
        """Resolve the registered modeling role for a seed."""
        roles = {value.role for value in self._assignments if value.seed == seed}
        if len(roles) != 1:
            raise KeyError(f"seed has no unique split role: {seed}")
        return next(iter(roles))

    def outer_fold_for_seed(self, seed: int) -> int:
        """Resolve the one leave-seed-out fold registered for a seed."""
        folds = {value.outer_fold for value in self._assignments if value.seed == seed}
        if len(folds) != 1:
            raise KeyError(f"seed has no unique outer fold: {seed}")
        return next(iter(folds))

    def manifest(self) -> dict[str, Any]:
        """Return assignments and explicit confirmatory-lock status."""
        return {
            "schema_version": "owl.cadc.phase4-split-registry.v1",
            "split_registry_digest": self.digest,
            "assignments": [asdict(value) for value in self._assignments],
            "phase5_locked": True,
            "phase6_locked": True,
        }


def _fold(group_id: str, *, salt: str, count: int) -> int:
    payload = f"{salt}\0{group_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % count


def build_grouped_splits(
    groups: Sequence[Mapping[str, Any]],
    *,
    seed_roles: Mapping[int, SplitRole],
    outer_folds: int,
    inner_folds: int,
    master_seed: int,
) -> SplitRegistry:
    """Hash unique world groups into folds after assigning immutable seed roles."""
    if outer_folds < 2 or inner_folds < 2:
        raise ValueError("outer and inner folds must both be at least two")
    assignments: list[SplitAssignment] = []
    materialized_seeds = sorted({int(group["seed"]) for group in groups})
    outer_by_seed: dict[int, int] = {}
    for role in SplitRole:
        seed_order = sorted(
            [seed for seed in materialized_seeds if seed_roles.get(seed) is role],
            key=lambda seed: hashlib.sha256(
                f"outer:{master_seed}:{role.value}\0{seed}".encode()
            ).digest(),
        )
        outer_by_seed.update(
            {seed: index % outer_folds for index, seed in enumerate(seed_order)}
        )
    for group in groups:
        seed = int(group["seed"])
        if seed not in seed_roles:
            raise KeyError(f"unregistered seed would bypass the seal: {seed}")
        group_id = stable_id(
            "split_group",
            seed,
            str(group.get("run_id", "")),
            str(group.get("condition", "")),
            str(group.get("world_id", "")),
        )
        assignments.append(
            SplitAssignment(
                group_id=group_id,
                seed=seed,
                role=seed_roles[seed],
                outer_fold=outer_by_seed[seed],
                inner_fold=_fold(
                    group_id, salt=f"inner:{master_seed}", count=inner_folds
                ),
            )
        )
    return SplitRegistry(assignments)


def seed_role_map(
    *,
    development: Sequence[int],
    validation: Sequence[int],
    calibration: Sequence[int],
    phase5: Sequence[int],
    phase6: Sequence[int],
) -> dict[int, SplitRole]:
    """Construct a role map and reject every seed overlap."""
    output: dict[int, SplitRole] = {}
    for role, values in (
        (SplitRole.TRAIN, development),
        (SplitRole.VALIDATION, validation),
        (SplitRole.CALIBRATION, calibration),
        (SplitRole.PHASE5_SEALED, phase5),
        (SplitRole.PHASE6_SEALED, phase6),
    ):
        for seed in values:
            if int(seed) in output:
                raise ValueError(f"seed appears in multiple roles: {seed}")
            output[int(seed)] = role
    return output


def validate_no_leakage(
    source_decision_ids: npt.NDArray[Any],
    split_labels: npt.NDArray[Any],
    *,
    forbidden_roles: Sequence[SplitRole] = (
        SplitRole.PHASE5_SEALED,
        SplitRole.PHASE6_SEALED,
    ),
) -> None:
    """Reject decision-level crossing and any materialized sealed-seed row."""
    decisions = np.asarray(source_decision_ids).astype(str)
    labels = np.asarray(split_labels).astype(str)
    if decisions.size != labels.size:
        raise ValueError("decision and split arrays have different lengths")
    order = np.argsort(decisions, kind="stable")
    sorted_ids = decisions[order]
    sorted_labels = labels[order]
    if sorted_ids.size:
        starts = np.r_[0, np.flatnonzero(sorted_ids[1:] != sorted_ids[:-1]) + 1]
        ends = np.r_[starts[1:], sorted_ids.size]
        for start, end in zip(starts, ends, strict=True):
            if np.unique(sorted_labels[start:end]).size != 1:
                raise ValueError(f"source decision crosses splits: {sorted_ids[start]}")
    forbidden = {role.value for role in forbidden_roles}
    leaked = sorted(forbidden.intersection(set(labels.tolist())))
    if leaked:
        raise ValueError(f"sealed confirmatory rows were materialized: {leaked}")
