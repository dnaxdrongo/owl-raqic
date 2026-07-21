"""Define versioned, perspective-safe CADC-MORE 2 features and transforms."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from owl.cadc.schema import FeaturePerspective, FeatureStage, ModelRole


@dataclass(frozen=True)
class FeatureDefinition:
    """One auditable feature with an exact source and availability boundary."""

    name: str
    source_table: str
    source_column: str
    dtype: str
    perspective: FeaturePerspective
    stage: FeatureStage
    missing_policy: str = "fail"
    normalization: str = "robust_standardize"
    model_roles: tuple[ModelRole, ...] = ()
    description: str = ""


_AGENT_SCALARS = (
    "agent_food_pressure",
    "agent_toxin_pressure",
    "agent_crowding",
    "agent_novelty",
    "agent_hunger",
    "agent_pain",
    "agent_boundary_stress",
    "agent_social_need",
    "agent_memory",
    "agent_health",
    "agent_resource",
    "agent_boundary",
    "agent_integration",
    "agent_threshold",
    "agent_activation",
    "agent_phase_coherence",
    "agent_sensed_food_mean",
    "agent_sensed_toxin_mean",
    "agent_sensed_alive_density",
    "agent_active_sense_food_memory",
    "agent_active_sense_toxin_memory",
    "agent_active_sense_alive_memory",
    "agent_active_sense_ttl",
)

_AGENT_TRAITS = (
    "mobility",
    "metabolism",
    "predation",
    "grazing",
    "cooperation",
    "aggression",
    "curiosity",
    "reproduction_rate",
    "toxin_resistance",
    "memory_capacity",
    "coupling_strength",
    "emit_strength",
    "emit_efficiency",
    "receive_sensitivity",
    "signal_precision",
    "honesty_bias",
    "deception_bias",
)

_ORACLE_COLUMNS = (
    "oracle_food",
    "oracle_toxin",
    "oracle_waste",
    "oracle_occupancy",
    "oracle_obstacle",
    "oracle_signal",
)

_CANDIDATE_COLUMNS = (
    "action_index",
    "target_kind",
    "proposed_y",
    "proposed_x",
    "resolved_y",
    "resolved_x",
    "target_ow_id",
    "destination_occupancy",
    "destination_obstacle",
    "destination_food",
    "destination_toxin",
    "opportunity_count",
    "policy_legal",
    "prechoice_executable",
    "prechoice_reason_code",
    "target_source",
    "target_distance",
    "target_confidence",
    "compiled_action",
)

_MECHANISM_COLUMNS = (
    ("agent_context", "agent_phase"),
    ("candidates", "utility"),
    ("agent_context", "agent_parent_intention"),
    ("agent_context", "agent_prior_probability"),
)

_EXECUTION_COLUMNS = (
    "attempted_action",
    "realized_action",
    "execution_success",
    "execution_reason_code",
    "compiled_execution_action",
    "realized_target_y",
    "realized_target_x",
    "realized_target_ow_id",
    "amount_consumed",
    "amount_transferred",
    "amount_repaired",
    "amount_damaged",
    "amount_emitted",
    "amount_received",
    "direct_cost",
)

_PRIMARY_FORBIDDEN_TABLES = frozenset(
    {
        "oracle_context",
        "dense_context",
        "execution",
        "events",
        "contributions",
        "information_followups",
        "branch_attempts",
        "counterfactual_micro_rollouts",
        "branch_events",
        "branch_event_summaries",
        "branch_contributions",
        "candidate_pairs",
    }
)

_PRIMARY_FORBIDDEN_COLUMNS = frozenset(
    {
        "selected_action",
        "selected_probability",
        "utility",
        "agent_parent_intention",
        "agent_prior_probability",
        "agent_phase",
        *_EXECUTION_COLUMNS,
    }
)


def _definition(
    source_table: str,
    source_column: str,
    perspective: FeaturePerspective,
    stage: FeatureStage,
    *,
    dtype: str = "float32",
    roles: tuple[ModelRole, ...] = (),
) -> FeatureDefinition:
    return FeatureDefinition(
        name=f"{source_table}.{source_column}",
        source_table=source_table,
        source_column=source_column,
        dtype=dtype,
        perspective=perspective,
        stage=stage,
        model_roles=roles,
    )


class FeatureRegistry:
    """Immutable feature registry with hard perspective and timing gates."""

    def __init__(self, definitions: Sequence[FeatureDefinition] | None = None) -> None:
        values = tuple(definitions) if definitions is not None else default_features()
        names = [value.name for value in values]
        if len(names) != len(set(names)):
            raise ValueError("feature registry contains duplicate names")
        for value in values:
            validate_feature_perspective(value)
        self._definitions = values

    @property
    def definitions(self) -> tuple[FeatureDefinition, ...]:
        """Return the immutable ordered feature definitions."""
        return self._definitions

    @property
    def digest(self) -> str:
        """Return the canonical SHA-256 identity of the registry."""
        payload = json.dumps(
            [asdict(value) for value in self._definitions],
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    def for_perspective(
        self, perspective: FeaturePerspective
    ) -> tuple[FeatureDefinition, ...]:
        """Return definitions belonging to one isolated evidence perspective."""
        return tuple(value for value in self._definitions if value.perspective is perspective)

    def names(self, perspective: FeaturePerspective) -> tuple[str, ...]:
        """Return ordered names for one isolated evidence perspective."""
        return tuple(value.name for value in self.for_perspective(perspective))

    def manifest(self) -> dict[str, Any]:
        """Return a machine-readable registry and nonfinite-input policy."""
        return {
            "schema_version": "owl.cadc.phase4-feature-registry.v1",
            "digest": self.digest,
            "definitions": [asdict(value) for value in self._definitions],
            "model_input_nonfinite_encoding": {
                "direction_context.direction_score": (
                    "finite_payload_zero_filled_plus_nan_posinf_neginf_masks"
                ),
                "all_other_features": "fail_closed",
                "outcomes": "fail_closed",
            },
        }


def validate_feature_perspective(definition: FeatureDefinition) -> None:
    """Reject leakage-prone feature/source combinations at registry construction."""
    if definition.perspective is FeaturePerspective.AGENT_PRIMARY:
        if definition.stage is not FeatureStage.PRE_CHOICE:
            raise ValueError(f"agent-primary feature is not pre-choice: {definition.name}")
        if definition.source_table in _PRIMARY_FORBIDDEN_TABLES:
            raise ValueError(f"agent-primary source table is forbidden: {definition.name}")
        if definition.source_column in _PRIMARY_FORBIDDEN_COLUMNS:
            raise ValueError(f"agent-primary source column is forbidden: {definition.name}")
        if definition.source_column.startswith(("oracle_", "dense_oracle_", "raqic_")):
            raise ValueError(f"agent-primary hidden source is forbidden: {definition.name}")
    if (
        definition.perspective is FeaturePerspective.ORACLE_DIAGNOSTIC
        and definition.stage is not FeatureStage.DIAGNOSTIC
    ):
        raise ValueError(f"oracle feature must remain diagnostic: {definition.name}")
    if (
        definition.perspective is FeaturePerspective.MECHANISM_MEDIATION
        and definition.stage is not FeatureStage.DIAGNOSTIC
    ):
        raise ValueError(f"mechanism feature must remain diagnostic: {definition.name}")
    if (
        definition.perspective is FeaturePerspective.EXECUTION_POSTCHOICE
        and definition.stage is not FeatureStage.POST_CHOICE
    ):
        raise ValueError(f"execution feature must be post-choice: {definition.name}")


def default_features() -> tuple[FeatureDefinition, ...]:
    """Return the source-grounded default four-view feature registry."""
    primary_roles = (
        ModelRole.VIABILITY_BASELINE,
        ModelRole.STRUCTURAL_TRANSITION,
        ModelRole.FAMILY_EXPERT,
        ModelRole.RANKER,
        ModelRole.SURVIVAL_RISK,
        ModelRole.EPISTEMIC_VALUE,
        ModelRole.EXTERNALITY,
    )
    values: list[FeatureDefinition] = []
    for column in _AGENT_SCALARS:
        values.append(
            _definition(
                "agent_context",
                column,
                FeaturePerspective.AGENT_PRIMARY,
                FeatureStage.PRE_CHOICE,
                roles=primary_roles,
            )
        )
    for trait in _AGENT_TRAITS:
        values.append(
            _definition(
                "agent_context",
                f"agent_trait_{trait}",
                FeaturePerspective.AGENT_PRIMARY,
                FeatureStage.PRE_CHOICE,
                roles=primary_roles,
            )
        )
    for column in ("agent_signal_reception", "agent_signal_memory"):
        values.append(
            _definition(
                "agent_context",
                column,
                FeaturePerspective.AGENT_PRIMARY,
                FeatureStage.PRE_CHOICE,
                dtype="fixed_size_list<float32>",
                roles=primary_roles,
            )
        )
    for column in _CANDIDATE_COLUMNS:
        dtype = (
            "bool"
            if column in {"destination_obstacle", "policy_legal", "prechoice_executable"}
            else "float32"
        )
        values.append(
            _definition(
                "candidates",
                column,
                FeaturePerspective.AGENT_PRIMARY,
                FeatureStage.PRE_CHOICE,
                dtype=dtype,
                roles=primary_roles[1:],
            )
        )
    for column in _ORACLE_COLUMNS:
        values.append(
            _definition(
                "oracle_context",
                column,
                FeaturePerspective.ORACLE_DIAGNOSTIC,
                FeatureStage.DIAGNOSTIC,
                dtype="fixed_size_list<float32>" if column == "oracle_signal" else "float32",
            )
        )
    for table, column in _MECHANISM_COLUMNS:
        values.append(
            _definition(
                table,
                column,
                FeaturePerspective.MECHANISM_MEDIATION,
                FeatureStage.DIAGNOSTIC,
            )
        )
    for column in _EXECUTION_COLUMNS:
        values.append(
            _definition(
                "execution",
                column,
                FeaturePerspective.EXECUTION_POSTCHOICE,
                FeatureStage.POST_CHOICE,
            )
        )
    return tuple(values)


def _build_view(
    tables: Mapping[str, Mapping[str, Any]], definitions: Sequence[FeatureDefinition]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for definition in definitions:
        table = tables.get(definition.source_table)
        if table is None or definition.source_column not in table:
            if definition.missing_policy == "fail":
                raise KeyError(f"missing required feature source: {definition.name}")
            continue
        result[definition.name] = table[definition.source_column]
    return result


def build_agent_features(
    tables: Mapping[str, Mapping[str, Any]], registry: FeatureRegistry | None = None
) -> dict[str, Any]:
    """Build the pre-choice agent-primary view only."""
    selected = registry or FeatureRegistry()
    return _build_view(tables, selected.for_perspective(FeaturePerspective.AGENT_PRIMARY))


def build_oracle_features(
    tables: Mapping[str, Mapping[str, Any]], registry: FeatureRegistry | None = None
) -> dict[str, Any]:
    """Build the explicitly diagnostic oracle view."""
    selected = registry or FeatureRegistry()
    return _build_view(
        tables, selected.for_perspective(FeaturePerspective.ORACLE_DIAGNOSTIC)
    )


def build_mechanism_features(
    tables: Mapping[str, Mapping[str, Any]], registry: FeatureRegistry | None = None
) -> dict[str, Any]:
    """Build the RAQIC mediator and moderator feature view for secondary analysis."""
    selected = registry or FeatureRegistry()
    return _build_view(
        tables, selected.for_perspective(FeaturePerspective.MECHANISM_MEDIATION)
    )


def build_execution_features(
    tables: Mapping[str, Mapping[str, Any]], registry: FeatureRegistry | None = None
) -> dict[str, Any]:
    """Build the post-choice execution analysis view."""
    selected = registry or FeatureRegistry()
    return _build_view(
        tables, selected.for_perspective(FeaturePerspective.EXECUTION_POSTCHOICE)
    )


def build_history(
    values: npt.NDArray[Any],
    group_ids: npt.NDArray[Any],
    ticks: npt.NDArray[Any],
    *,
    length: int,
    fill_value: float = 0.0,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Create deterministic causal histories without crossing OW/world groups."""
    data = np.asarray(values)
    groups = np.asarray(group_ids)
    time = np.asarray(ticks)
    if data.shape[0] != groups.size or groups.size != time.size:
        raise ValueError("history inputs must have equal row counts")
    if length < 0:
        raise ValueError("history length must be nonnegative")
    trailing = data.shape[1:]
    history = np.full((data.shape[0], length, *trailing), fill_value, dtype=data.dtype)
    mask = np.zeros((data.shape[0], length), dtype=bool)
    if length == 0:
        return history, mask
    order = np.lexsort((np.arange(groups.size), time, groups.astype(str)))
    positions: dict[str, list[int]] = {}
    for row in order:
        key = str(groups[row])
        prior = positions.setdefault(key, [])
        chosen = prior[-length:]
        offset = length - len(chosen)
        if chosen:
            history[row, offset:] = data[np.asarray(chosen, dtype=np.int64)]
            mask[row, offset:] = True
        prior.append(int(row))
    return history, mask
