"""Inject validated high-level forced actions at the action-transition seam."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from owl.core.actions import Action
from owl.record.cadc_schema import ReasonCode


@dataclass(frozen=True)
class ForcedActionBatch:
    branch_index: Any
    source_decision_index: Any
    ow_id: Any
    source_y: Any
    source_x: Any
    factual_action: Any
    forced_action: Any
    target_kind: Any
    target_source: Any
    target_y: Any
    target_x: Any
    target_ow_id: Any
    target_distance: Any
    target_confidence: Any
    compiled_candidate: Any
    policy_legal: Any
    prechoice_executable: Any
    reason: Any
    candidate_sequence: Any

    @property
    def size(self) -> int:
        return int(self.source_y.shape[0])


def build_forced_action_batch(
    decisions: Any,
    decision_indices: Any,
    actions: Any,
    *,
    branch_indices: Any | None = None,
) -> ForcedActionBatch:
    xp = decisions.source_y.__class__.__module__.startswith("cupy")
    if xp:
        import cupy as array_namespace  # pragma: no cover
    else:
        import numpy as array_namespace

    di = array_namespace.asarray(decision_indices, dtype=array_namespace.int32)
    action = array_namespace.asarray(actions, dtype=array_namespace.int16)
    if di.ndim != 1 or action.shape != di.shape:
        raise ValueError("decision indices and forced actions must be equal one-dimensional arrays")
    rows = array_namespace.arange(di.shape[0], dtype=array_namespace.int32)
    branches = rows if branch_indices is None else array_namespace.asarray(branch_indices)
    y = decisions.source_y[di]
    x = decisions.source_x[di]
    target_y = decisions.candidate_resolved_y[di, action]
    target_x = decisions.candidate_resolved_x[di, action]
    semantic = (action == int(Action.FLEE)) | (action == int(Action.PURSUE))
    family = array_namespace.where(action == int(Action.PURSUE), 1, 0).astype(array_namespace.int32)
    semantic_y = decisions.direction_fields["action_target_y"][di, family]
    semantic_x = decisions.direction_fields["action_target_x"][di, family]
    target_y = array_namespace.where(semantic, semantic_y, target_y)
    target_x = array_namespace.where(semantic, semantic_x, target_x)
    target_ow = array_namespace.where(
        semantic,
        decisions.direction_fields["action_target_ow_id"][di, family],
        decisions.candidate_target_ow_id[di, action],
    )
    target_kind = array_namespace.where(
        semantic,
        decisions.direction_fields["action_target_kind"][di, family],
        decisions.candidate_target_kind[di, action],
    )
    target_source = array_namespace.where(
        semantic,
        decisions.direction_fields["action_target_source"][di, family],
        decisions.candidate_target_source[di, action],
    )
    target_distance = array_namespace.where(
        semantic,
        decisions.direction_fields["action_target_distance"][di, family],
        decisions.candidate_target_distance[di, action],
    )
    target_confidence = array_namespace.where(
        semantic,
        decisions.direction_fields["action_target_confidence"][di, family],
        decisions.candidate_target_confidence[di, action],
    )
    return ForcedActionBatch(
        branch_index=branches,
        source_decision_index=di,
        ow_id=decisions.ow_id[di],
        source_y=y,
        source_x=x,
        factual_action=decisions.selected_action[di],
        forced_action=action,
        target_kind=target_kind,
        target_source=target_source,
        target_y=target_y,
        target_x=target_x,
        target_ow_id=target_ow,
        target_distance=target_distance,
        target_confidence=target_confidence,
        compiled_candidate=decisions.candidate_compiled_action[di, action],
        policy_legal=decisions.policy_legal[di, action],
        prechoice_executable=decisions.prechoice_executable[di, action],
        reason=decisions.candidate_reason[di, action],
        candidate_sequence=decisions.decision_sequence[di] * len(Action) + action,
    )


def validation_mask(ds: Any, batch: ForcedActionBatch) -> Any:
    """Return backend-native validation without synchronizing one row at a time."""
    xp = ds.xp
    y, x = batch.source_y, batch.source_x
    action = batch.forced_action.astype(xp.int32)
    in_axis = (action >= 0) & (action < len(Action))
    alive = (ds.health[y, x] > 0.0) & (~ds.obstacle[y, x])
    expected_id = ds.occupancy[y, x]
    valid = (
        in_axis
        & alive
        & (expected_id == batch.ow_id)
        & batch.policy_legal
        & batch.prechoice_executable
        & (batch.reason == int(ReasonCode.NONE))
    )
    family = xp.where(action == int(Action.PURSUE), 1, 0).astype(xp.int32)
    high_level = (action == int(Action.FLEE)) | (action == int(Action.PURSUE))
    target_y = ds.action_target_y[y, x, family]
    target_x = ds.action_target_x[y, x, family]
    target_id = ds.action_target_ow_id[y, x, family]
    target_kind = ds.action_target_kind[y, x, family]
    target_source = ds.action_target_source[y, x, family]
    target_distance = ds.action_target_distance[y, x, family]
    target_confidence = ds.action_target_confidence[y, x, family]
    compiled = xp.where(
        action == int(Action.FLEE),
        ds.flee_compiled_action[y, x],
        ds.pursue_compiled_action[y, x],
    )
    target_match = (
        (target_y == batch.target_y)
        & (target_x == batch.target_x)
        & (target_id == batch.target_ow_id)
        & (target_kind == batch.target_kind)
        & (target_source == batch.target_source)
        & (target_distance == batch.target_distance)
        & (target_confidence == batch.target_confidence)
        & (compiled == batch.compiled_candidate)
    )
    valid &= (~high_level) | target_match
    sense = action == int(Action.SENSE)
    valid &= (~sense) | (
        ds.resource[y, x] >= float(ds.metadata["cfg"].action_transitions.active_sense_cost)
    )
    return valid


def inject_forced_actions(ds: Any, batch: ForcedActionBatch) -> Any:
    """Overwrite only committed high-level action fields in one indexed write."""
    valid = validation_mask(ds, batch)
    y = batch.source_y[valid]
    x = batch.source_x[valid]
    action = batch.forced_action[valid].astype(ds.readout.dtype, copy=False)
    ds.readout[y, x] = action
    if "raqic_readout" in ds.arrays:
        ds.raqic_readout[y, x] = action.astype(ds.raqic_readout.dtype, copy=False)
    return valid
