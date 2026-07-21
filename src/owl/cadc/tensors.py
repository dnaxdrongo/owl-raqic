"""Fixed-action tensor assembly for vectorized CADC-MORE 2 training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Phase4TensorBatch:
    """Fixed-axis training tensors plus decision and split provenance."""

    decision_ids: Any
    context: Any
    oracle_context: Any
    candidates: Any
    candidate_mask: Any
    directions: Any
    direction_mask: Any
    horizons: Any
    outcomes: Any
    outcome_variance: Any
    scalar_targets: Any
    scalar_quantiles: Any
    scalar_cvar: Any
    outcome_mask: Any
    repeat_count: Any
    selected_actions: Any
    seeds: Any
    split_roles: Any
    outer_folds: Any


def _xp(value: Any) -> Any:
    if type(value).__module__.split(".", maxsplit=1)[0] == "cupy":
        import cupy as cp

        return cp
    return np


def assemble_fixed_action_tensors(
    *,
    decision_ids: Any,
    seeds: Any,
    split_roles: Any,
    outer_folds: Any,
    context_columns: Mapping[str, Any],
    oracle_context_columns: Mapping[str, Any],
    candidate_decision_index: Any,
    candidate_action_index: Any,
    candidate_columns: Mapping[str, Any],
    candidate_executable: Any,
    direction_decision_index: Any,
    direction_family_index: Any,
    direction_index: Any,
    direction_columns: Mapping[str, Any],
    direction_executable: Any,
    branch_decision_index: Any,
    branch_action_index: Any,
    branch_horizon: Any,
    branch_repeat_index: Any,
    outcome_columns: Mapping[str, Any],
    branch_scalar_target: Any,
    registered_horizons: Sequence[int],
    quantile_levels: Sequence[float],
    cvar_alpha: float,
    selected_actions: Any,
) -> Phase4TensorBatch:
    """Assemble ``[D,22]`` and ``[D,2,8]`` tensors with repeat means/masks."""
    xp = _xp(next(iter(context_columns.values())))
    ids = np.asarray(decision_ids).astype(str)
    decisions = int(ids.shape[0])
    context = _stack_columns(context_columns, xp=xp, rows=decisions)
    oracle_context = _stack_columns(
        oracle_context_columns, xp=xp, rows=decisions
    )
    candidate_row = xp.asarray(candidate_decision_index, dtype=xp.int64)
    action = xp.asarray(candidate_action_index, dtype=xp.int64)
    if int(candidate_row.size) != decisions * 22:
        raise ValueError("tensor assembly requires exactly 22 candidate rows per decision")
    candidate_values = _stack_columns(
        candidate_columns, xp=xp, rows=int(candidate_row.size)
    )
    candidates = xp.zeros((decisions, 22, candidate_values.shape[-1]), dtype=xp.float32)
    candidate_mask = xp.zeros((decisions, 22), dtype=bool)
    candidates[candidate_row, action] = candidate_values
    candidate_mask[candidate_row, action] = xp.asarray(candidate_executable, dtype=bool)
    if not bool(xp.all(xp.bincount(candidate_row, minlength=decisions) == 22)):
        raise ValueError("candidate decision index violates the 22-action contract")

    direction_row = xp.asarray(direction_decision_index, dtype=xp.int64)
    family = xp.asarray(direction_family_index, dtype=xp.int64)
    local_direction = xp.asarray(direction_index, dtype=xp.int64)
    direction_values = _stack_columns(
        direction_columns,
        xp=xp,
        rows=int(direction_row.size),
        nonfinite_mask_columns=frozenset({"direction_score"}),
    )
    directions = xp.zeros(
        (decisions, 2, 8, direction_values.shape[-1]), dtype=xp.float32
    )
    direction_mask = xp.zeros((decisions, 2, 8), dtype=bool)
    directions[direction_row, family, local_direction] = direction_values
    direction_mask[direction_row, family, local_direction] = xp.asarray(
        direction_executable, dtype=bool
    )
    if not bool(xp.all(xp.bincount(direction_row, minlength=decisions) == 16)):
        raise ValueError("direction decision index violates the 16-direction contract")

    horizons = xp.asarray(tuple(registered_horizons), dtype=xp.int32)
    horizon_values = xp.asarray(branch_horizon, dtype=xp.int32)
    horizon_slot = xp.searchsorted(horizons, horizon_values)
    valid_horizon = (horizon_slot < horizons.size) & (
        horizons[xp.minimum(horizon_slot, horizons.size - 1)] == horizon_values
    )
    if not bool(xp.all(valid_horizon)):
        raise ValueError("branch horizon lies outside the registered horizon axis")
    branch_decision = xp.asarray(branch_decision_index, dtype=xp.int64)
    branch_action = xp.asarray(branch_action_index, dtype=xp.int64)
    outcome_values = _stack_columns(
        outcome_columns, xp=xp, rows=int(branch_decision.size)
    )
    outcome_sum = xp.zeros(
        (decisions, horizons.size, 22, outcome_values.shape[-1]), dtype=xp.float64
    )
    count = xp.zeros((decisions, horizons.size, 22), dtype=xp.int32)
    outcome_square_sum = xp.zeros_like(outcome_sum)
    xp.add.at(
        outcome_sum,
        (branch_decision, horizon_slot, branch_action),
        outcome_values.astype(xp.float64),
    )
    xp.add.at(
        outcome_square_sum,
        (branch_decision, horizon_slot, branch_action),
        outcome_values.astype(xp.float64) ** 2,
    )
    xp.add.at(count, (branch_decision, horizon_slot, branch_action), 1)
    outcomes = (
        outcome_sum / xp.maximum(count[..., None], 1)
    ).astype(xp.float32)
    outcome_variance = xp.maximum(
        outcome_square_sum / xp.maximum(count[..., None], 1)
        - outcomes.astype(xp.float64) ** 2,
        0.0,
    ).astype(xp.float32)
    scalar_sum = xp.zeros((decisions, horizons.size, 22), dtype=xp.float64)
    xp.add.at(
        scalar_sum,
        (branch_decision, horizon_slot, branch_action),
        xp.asarray(branch_scalar_target, dtype=xp.float64),
    )
    scalar_targets = (scalar_sum / xp.maximum(count, 1)).astype(xp.float32)
    repeats = xp.asarray(branch_repeat_index, dtype=xp.int64)
    maximum_repeats = max(1, int(repeats.max()) + 1)
    samples = xp.full(
        (decisions, horizons.size, 22, maximum_repeats),
        xp.inf,
        dtype=xp.float32,
    )
    samples[branch_decision, horizon_slot, branch_action, repeats] = xp.asarray(
        branch_scalar_target, dtype=xp.float32
    )
    ordered = xp.sort(samples, axis=-1)
    quantile_parts = []
    count_float = count.astype(xp.float32)
    for level in quantile_levels:
        position = float(level) * xp.maximum(count_float - 1.0, 0.0)
        lower_index = xp.floor(position).astype(xp.int64)
        upper_index = xp.ceil(position).astype(xp.int64)
        lower = xp.take_along_axis(ordered, lower_index[..., None], axis=-1)[..., 0]
        upper = xp.take_along_axis(ordered, upper_index[..., None], axis=-1)[..., 0]
        lower = xp.where(count > 0, lower, 0.0)
        upper = xp.where(count > 0, upper, 0.0)
        value = lower + (upper - lower) * (position - lower_index)
        quantile_parts.append(xp.where(count > 0, value, 0.0))
    scalar_quantiles = xp.stack(quantile_parts, axis=-1).astype(xp.float32)
    tail_count = xp.maximum(xp.ceil(float(cvar_alpha) * count_float), 1).astype(
        xp.int32
    )
    rank = xp.arange(maximum_repeats, dtype=xp.int32)
    tail_mask = rank[None, None, None, :] < tail_count[..., None]
    scalar_cvar = xp.where(
        count > 0,
        xp.sum(xp.where(tail_mask, ordered, 0.0), axis=-1)
        / tail_count.astype(xp.float32),
        0.0,
    ).astype(xp.float32)
    mask = count > 0
    return Phase4TensorBatch(
        decision_ids=ids,
        context=context,
        oracle_context=oracle_context,
        candidates=candidates,
        candidate_mask=candidate_mask,
        directions=directions,
        direction_mask=direction_mask,
        horizons=horizons,
        outcomes=outcomes,
        outcome_variance=outcome_variance,
        scalar_targets=scalar_targets,
        scalar_quantiles=scalar_quantiles,
        scalar_cvar=scalar_cvar,
        outcome_mask=mask,
        repeat_count=count,
        selected_actions=xp.asarray(selected_actions, dtype=xp.int16),
        seeds=xp.asarray(seeds),
        split_roles=np.asarray(split_roles).astype(str),
        outer_folds=np.asarray(outer_folds, dtype=np.int16),
    )


def _stack_columns(
    columns: Mapping[str, Any],
    *,
    xp: Any,
    rows: int,
    nonfinite_mask_columns: frozenset[str] = frozenset(),
) -> Any:
    if not columns:
        raise ValueError("tensor feature mapping cannot be empty")
    pieces = []
    for name in sorted(columns):
        values = xp.asarray(columns[name])
        if values.shape[0] != rows:
            raise ValueError(f"tensor column has wrong row count: {name}")
        flattened = values.reshape(rows, -1).astype(xp.float32, copy=False)
        finite = xp.isfinite(flattened)
        if name in nonfinite_mask_columns:
            pieces.append(xp.where(finite, flattened, 0.0).astype(xp.float32))
            pieces.append(xp.isnan(flattened).astype(xp.float32))
            pieces.append(xp.isposinf(flattened).astype(xp.float32))
            pieces.append(xp.isneginf(flattened).astype(xp.float32))
            continue
        if not bool(xp.all(finite)):
            raise ValueError(f"tensor column contains nonfinite values: {name}")
        pieces.append(flattened)
    return xp.concatenate(pieces, axis=1)
