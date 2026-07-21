"""Define versioned raw outcome, event, contribution, and censor contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import IntEnum, StrEnum
from typing import Any

import numpy as np

from owl.cadc.schema import OutcomeFamily
from owl.record.cadc_schema import CADCEventCode


class OutcomePerspective(StrEnum):
    """Permitted evidence perspective for an outcome definition."""

    AGENT = "agent"
    ORACLE_DIAGNOSTIC = "oracle_diagnostic"
    COLLECTIVE = "collective"


class CensorStatus(IntEnum):
    """Explicit observability state for one outcome endpoint."""

    OBSERVED = 0
    RIGHT_CENSORED = 1
    FOCAL_ABSENT_AMBIGUOUS = 2
    BRANCH_FAILED = 3


class DeathCause(IntEnum):
    """Cause registry aligned with the five-class competing-risk target."""

    NONE = 0
    STARVATION = 1
    TOXIN = 2
    OTHER_OBSERVED = 3
    AMBIGUOUS = 4


@dataclass(frozen=True)
class OutcomeDefinition:
    """One raw target with source lineage and missing/censor semantics."""

    name: str
    dtype: str
    unit: str
    direction: int
    family: OutcomeFamily
    horizons: tuple[int, ...]
    evidence: tuple[str, ...]
    derivation: str
    missing_rule: str
    perspective: OutcomePerspective
    loss: str
    calibration_metric: str


_BASE_HORIZONS = (1, 3, 5, 10)
_LONG_HORIZONS = (1, 3, 5, 10, 25)


def _raw(
    name: str,
    *,
    family: OutcomeFamily,
    unit: str = "state_unit",
    direction: int = 1,
    horizons: tuple[int, ...] = _BASE_HORIZONS,
    evidence: tuple[str, ...] | None = None,
    perspective: OutcomePerspective = OutcomePerspective.AGENT,
    loss: str = "huber",
    calibration: str = "mae_coverage",
    missing: str = "mask_when_branch_or_horizon_unavailable",
    derivation: str | None = None,
) -> OutcomeDefinition:
    source = evidence or (f"counterfactual_micro_rollouts.{name}",)
    return OutcomeDefinition(
        name=name,
        dtype="float32",
        unit=unit,
        direction=direction,
        family=family,
        horizons=horizons,
        evidence=source,
        derivation=derivation or f"direct:{source[0]}",
        missing_rule=missing,
        perspective=perspective,
        loss=loss,
        calibration_metric=calibration,
    )


def default_outcomes() -> tuple[OutcomeDefinition, ...]:
    """Return the raw outcome vector registry before scalarization."""
    values = [
        _raw("health_delta", family=OutcomeFamily.HOMEOSTASIS),
        _raw("resource_delta", family=OutcomeFamily.HOMEOSTASIS),
        _raw("boundary_delta", family=OutcomeFamily.HOMEOSTASIS),
        _raw("integration_delta", family=OutcomeFamily.HOMEOSTASIS),
        _raw("memory_delta", family=OutcomeFamily.HOMEOSTASIS),
        _raw("alive", family=OutcomeFamily.SURVIVAL, unit="probability", loss="bce"),
        _raw(
            "death_by_horizon",
            family=OutcomeFamily.SURVIVAL,
            direction=-1,
            unit="indicator",
            evidence=("counterfactual_micro_rollouts.alive",),
            loss="bce",
            calibration="brier_coverage",
        ),
        _raw(
            "first_death_tick",
            family=OutcomeFamily.SURVIVAL,
            unit="tick",
            direction=1,
            loss="discrete_time_hazard",
        ),
        _raw(
            "death_cause",
            family=OutcomeFamily.SURVIVAL,
            unit="cause_code",
            direction=0,
            evidence=("branch_events.event_code", "counterfactual_micro_rollouts.alive"),
            loss="cross_entropy",
            calibration="cause_brier",
        ),
        _raw("displacement_y", family=OutcomeFamily.ACTION_ENDPOINT, unit="cell"),
        _raw("displacement_x", family=OutcomeFamily.ACTION_ENDPOINT, unit="cell"),
        _raw("target_distance_delta", family=OutcomeFamily.ACTION_ENDPOINT, direction=-1),
        _raw("contact_opportunity", family=OutcomeFamily.ACTION_ENDPOINT, loss="bce"),
        _raw("known_hazard", family=OutcomeFamily.ACTION_ENDPOINT, direction=-1),
        _raw(
            "active_sense_new_cell_count",
            family=OutcomeFamily.INFORMATION,
            horizons=_LONG_HORIZONS,
        ),
        _raw(
            "active_sense_new_target_count",
            family=OutcomeFamily.INFORMATION,
            horizons=_LONG_HORIZONS,
        ),
        _raw(
            "information_control_value",
            family=OutcomeFamily.INFORMATION,
            horizons=_LONG_HORIZONS,
            evidence=(
                "information.*",
                "information_followups.*",
                "counterfactual_micro_rollouts.*",
            ),
            missing="right_censor_without_information_followup",
        ),
        _raw(
            "population_delta",
            family=OutcomeFamily.EXTERNALITY,
            horizons=_LONG_HORIZONS,
            perspective=OutcomePerspective.COLLECTIVE,
            evidence=("externality_targets.population_delta_vs_anchor",),
            missing="mask_without_selected_anchor",
        ),
        _raw(
            "world_food_delta",
            family=OutcomeFamily.EXTERNALITY,
            perspective=OutcomePerspective.COLLECTIVE,
            evidence=("externality_targets.world_food_delta_vs_anchor",),
            missing="mask_without_selected_anchor",
        ),
        _raw(
            "world_toxin_delta",
            family=OutcomeFamily.EXTERNALITY,
            direction=-1,
            perspective=OutcomePerspective.COLLECTIVE,
            evidence=("externality_targets.world_toxin_delta_vs_anchor",),
            missing="mask_without_selected_anchor",
        ),
        _raw(
            "world_waste_delta",
            family=OutcomeFamily.EXTERNALITY,
            direction=-1,
            perspective=OutcomePerspective.COLLECTIVE,
            evidence=("externality_targets.world_waste_delta_vs_anchor",),
            missing="mask_without_selected_anchor",
        ),
        _raw(
            "lineage_persistence",
            family=OutcomeFamily.LINEAGE,
            horizons=_LONG_HORIZONS,
            perspective=OutcomePerspective.COLLECTIVE,
            evidence=(
                "externality_targets.focal_lineage_persistence_delta_vs_anchor",
            ),
            derivation=(
                "alive and endpoint lineage equals factual source lineage, minus selected "
                "anchor; not descendant-lineage persistence"
            ),
            loss="huber",
        ),
    ]
    return tuple(values)


class OutcomeRegistry:
    """Immutable target registry that preserves raw targets before scalarization."""

    def __init__(self, definitions: Sequence[OutcomeDefinition] | None = None) -> None:
        values = tuple(definitions) if definitions is not None else default_outcomes()
        names = [value.name for value in values]
        if len(names) != len(set(names)):
            raise ValueError("outcome registry contains duplicate names")
        if any(value.direction not in {-1, 0, 1} for value in values):
            raise ValueError("outcome direction must be -1, 0, or 1")
        self._definitions = values

    @property
    def definitions(self) -> tuple[OutcomeDefinition, ...]:
        """Return immutable vector-outcome definitions."""
        return self._definitions

    @property
    def digest(self) -> str:
        """Return the canonical SHA-256 identity of the outcome registry."""
        payload = json.dumps(
            [asdict(value) for value in self._definitions],
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    def manifest(self) -> dict[str, Any]:
        """Return outcome definitions and their registry digest."""
        return {
            "schema_version": "owl.cadc.phase4-outcome-registry.v1",
            "outcome_registry_digest": self.digest,
            "definitions": [asdict(value) for value in self._definitions],
        }


def _xp(values: Any) -> Any:
    module = type(values).__module__.split(".", maxsplit=1)[0]
    if module == "cupy":
        import cupy as cp

        return cp
    return np


def reduce_events(
    event_code: Any,
    branch_tick: Any,
    *,
    horizons: Any,
) -> dict[str, Any]:
    """Vectorize cause evidence into one outcome per requested horizon."""
    xp = _xp(event_code)
    codes = xp.asarray(event_code)
    ticks = xp.asarray(branch_tick)
    ends = xp.asarray(horizons)
    if codes.ndim != 1 or ticks.shape != codes.shape or ends.ndim != 1:
        raise ValueError("event reduction expects one-dimensional arrays")
    within = ticks[:, None] <= ends[None, :]
    starvation = xp.any(
        within & (codes[:, None] == int(CADCEventCode.STARVATION_EVIDENCE)), axis=0
    )
    toxin = xp.any(
        within & (codes[:, None] == int(CADCEventCode.TOXIN_DAMAGE_EVIDENCE)), axis=0
    )
    death = xp.any(within & (codes[:, None] == int(CADCEventCode.DEATH)), axis=0)
    ambiguous = death & starvation & toxin
    cause = xp.where(
        ~death,
        int(DeathCause.NONE),
        xp.where(
            ambiguous,
            int(DeathCause.AMBIGUOUS),
            xp.where(
                starvation,
                int(DeathCause.STARVATION),
                xp.where(toxin, int(DeathCause.TOXIN), int(DeathCause.OTHER_OBSERVED)),
            ),
        ),
    ).astype(xp.int8)
    counts = xp.sum(within, axis=0, dtype=xp.int64)
    return {
        "event_count": counts,
        "starvation_evidence": starvation,
        "toxin_evidence": toxin,
        "death_event": death,
        "death_cause": cause,
    }


def reduce_contributions(
    fields: Sequence[str],
    branch_tick: Any,
    delta: Any,
    *,
    horizons: Any,
) -> dict[str, Any]:
    """Sum contribution deltas by field and horizon without changing evidence rows."""
    xp = _xp(delta)
    ticks = xp.asarray(branch_tick)
    values = xp.asarray(delta)
    ends = xp.asarray(horizons)
    field_values = np.asarray(fields, dtype=str)
    if values.ndim != 1 or ticks.shape != values.shape or field_values.size != values.size:
        raise ValueError("contribution reduction inputs have incompatible shapes")
    output: dict[str, Any] = {}
    for field in sorted(set(field_values.tolist())):
        mask = xp.asarray(field_values == field)
        output[field] = xp.sum(
            xp.where(mask[:, None] & (ticks[:, None] <= ends[None, :]), values[:, None], 0),
            axis=0,
            dtype=xp.float64,
        )
    return output


def build_branch_targets(
    horizon_rows: Mapping[str, Any],
    *,
    death_causes: Any | None = None,
) -> dict[str, Any]:
    """Derive direct per-branch targets while preserving source masks."""
    required = (
        "alive",
        "health_delta",
        "resource_delta",
        "boundary_delta",
        "integration_delta",
        "memory_delta",
    )
    missing = [name for name in required if name not in horizon_rows]
    if missing:
        raise KeyError(f"branch horizon columns missing: {missing}")
    xp = _xp(horizon_rows["alive"])
    alive = xp.asarray(horizon_rows["alive"], dtype=bool)
    status = xp.asarray(horizon_rows.get("horizon_status", xp.zeros_like(alive)))
    targets = {name: xp.asarray(value) for name, value in horizon_rows.items()}
    targets["death_by_horizon"] = ~alive
    targets["observed_mask"] = status != 3
    targets["censor_status"] = xp.where(
        status == 2,
        int(CensorStatus.FOCAL_ABSENT_AMBIGUOUS),
        xp.where(status == 3, int(CensorStatus.BRANCH_FAILED), int(CensorStatus.OBSERVED)),
    ).astype(xp.int8)
    if death_causes is not None:
        targets["death_cause"] = xp.asarray(death_causes, dtype=xp.int8)
    return targets


def build_survival_episodes(horizon_rows: Mapping[str, Any]) -> dict[str, Any]:
    """Build discrete-time survival labels with explicit censor state."""
    required = {"branch_id", "horizon", "alive", "first_death_tick", "horizon_status"}
    missing = sorted(required.difference(horizon_rows))
    if missing:
        raise KeyError(f"survival episode columns missing: {missing}")
    alive = np.asarray(horizon_rows["alive"], dtype=bool)
    horizon = np.asarray(horizon_rows["horizon"], dtype=np.int32)
    death_tick = np.asarray(horizon_rows["first_death_tick"], dtype=np.int64)
    status = np.asarray(horizon_rows["horizon_status"], dtype=np.int8)
    observed_death = (~alive) & (death_tick >= 0) & (death_tick <= horizon)
    censor = np.where(
        status == 2,
        int(CensorStatus.FOCAL_ABSENT_AMBIGUOUS),
        np.where(status == 3, int(CensorStatus.BRANCH_FAILED), int(CensorStatus.OBSERVED)),
    ).astype(np.int8)
    return {
        "branch_id": np.asarray(horizon_rows["branch_id"]),
        "horizon": horizon,
        "event_observed": observed_death,
        "event_time": np.where(observed_death, death_tick, horizon).astype(np.int64),
        "censor_status": censor,
        "valid_mask": censor == int(CensorStatus.OBSERVED),
    }
