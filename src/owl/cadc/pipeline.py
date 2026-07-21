"""Canonical Parquet-to-fixed-tensor training pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from owl.cadc.features import FeatureRegistry
from owl.cadc.schema import FeaturePerspective
from owl.cadc.tensors import Phase4TensorBatch, assemble_fixed_action_tensors

_DIRECTION_FEATURES = (
    "target_distance",
    "target_confidence",
    "direction_y",
    "direction_x",
    "direction_score",
    "distance_delta",
    "known_hazard",
    "opportunity",
)

_OUTCOME_FEATURES = (
    "health_delta",
    "resource_delta",
    "boundary_delta",
    "integration_delta",
    "memory_delta",
    "alive",
    "target_distance_delta",
    "contact_opportunity",
    "active_sense_new_cell_count",
    "active_sense_new_target_count",
    "population_delta_vs_anchor",
    "world_food_delta_vs_anchor",
    "world_toxin_delta_vs_anchor",
    "world_waste_delta_vs_anchor",
    "focal_lineage_persistence_delta_vs_anchor",
    "death_cause_0",
    "death_cause_1",
    "death_cause_2",
    "death_cause_3",
    "death_cause_4",
)

_EXTERNALITY_FEATURES = _OUTCOME_FEATURES[10:15]


def load_phase4_tensors(
    root: str | Path,
    *,
    backend: str,
    feature_registry: FeatureRegistry | None = None,
    history_length: int = 8,
    quantile_levels: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95),
    cvar_alpha: float = 0.1,
) -> Phase4TensorBatch:
    """Read canonical partitions and assemble fixed action/direction tensors."""
    dataset = Path(root)
    registry = feature_registry or FeatureRegistry()
    if backend == "cupy":
        return _load_gpu(
            dataset, registry, history_length, quantile_levels, cvar_alpha
        )
    if backend == "numpy":
        return _load_cpu(
            dataset, registry, history_length, quantile_levels, cvar_alpha
        )
    raise ValueError(f"unsupported tensor backend: {backend}")


def _load_gpu(
    root: Path,
    registry: FeatureRegistry,
    history_length: int,
    quantile_levels: tuple[float, ...],
    cvar_alpha: float,
) -> Phase4TensorBatch:
    import cudf
    import cupy as cp

    decision = cudf.read_parquet(root / "decision_context")
    candidate = cudf.read_parquet(root / "candidate_context")
    direction = cudf.read_parquet(root / "direction_context")
    history = cudf.read_parquet(root / "history_context")
    targets = cudf.read_parquet(root / "branch_targets")
    externality = cudf.read_parquet(root / "externality_targets")
    decision = decision.sort_values("source_decision_id").reset_index(drop=True)
    if not bool(decision["source_decision_id"].is_unique):
        raise ValueError("decision context is not unique by source_decision_id")
    decision["decision_index"] = cp.arange(len(decision), dtype=cp.int64)
    index = decision[["source_decision_id", "decision_index"]]
    candidate = candidate.merge(index, on="source_decision_id", how="inner").sort_values(
        ["decision_index", "action_index"]
    )
    direction = direction.merge(index, on="source_decision_id", how="inner").sort_values(
        ["decision_index", "action_family", "direction_index"]
    )
    history = history.merge(index, on="source_decision_id", how="inner")
    target_keys = [
        "branch_id",
        "source_decision_id",
        "forced_action",
        "repeat_index",
        "horizon",
    ]
    if bool(targets.duplicated(subset=target_keys).any()):
        raise ValueError("branch targets are not unique by the one-to-one join key")
    if bool(externality.duplicated(subset=target_keys).any()):
        raise ValueError("externality targets are not unique by the one-to-one join key")
    target_rows = len(targets)
    targets = targets.merge(
        externality,
        on=target_keys,
        how="left",
    )
    if len(targets) != target_rows:
        raise ValueError("externality one-to-one join changed branch target cardinality")
    targets = targets.merge(index, on="source_decision_id", how="inner")
    if targets[list(_EXTERNALITY_FEATURES)].isnull().any().any():
        raise ValueError("externality target join is incomplete")
    context_names = [
        value.source_column
        for value in registry.for_perspective(FeaturePerspective.AGENT_PRIMARY)
        if value.source_table == "agent_context" and value.source_column in decision.columns
    ]
    oracle_names = [
        value.source_column
        for value in registry.for_perspective(FeaturePerspective.ORACLE_DIAGNOSTIC)
        if value.source_table == "oracle_context" and value.source_column in decision.columns
    ]
    candidate_names = [
        value.source_column
        for value in registry.for_perspective(FeaturePerspective.AGENT_PRIMARY)
        if value.source_table == "candidates"
        and value.source_column in candidate.columns
        and value.source_column not in {"policy_legal", "prechoice_executable"}
    ]
    context_columns = _cudf_numeric_columns(decision, context_names)
    context_columns.update(
        _cudf_history_columns(
            history,
            decision,
            context_names,
            decisions=len(decision),
            history_length=history_length,
        )
    )
    return assemble_fixed_action_tensors(
        decision_ids=decision["source_decision_id"].to_arrow().to_pylist(),
        seeds=decision["seed"].values,
        split_roles=decision["split_role"].to_arrow().to_pylist(),
        outer_folds=decision["outer_fold"].to_arrow().to_pylist(),
        context_columns=context_columns,
        oracle_context_columns=_cudf_numeric_columns(decision, oracle_names),
        candidate_decision_index=candidate["decision_index"].values,
        candidate_action_index=candidate["action_index"].values,
        candidate_columns=_cudf_numeric_columns(candidate, candidate_names),
        candidate_executable=(
            candidate["policy_legal"].astype("bool")
            & candidate["prechoice_executable"].astype("bool")
        ).values,
        direction_decision_index=direction["decision_index"].values,
        direction_family_index=direction["action_family"].values,
        direction_index=direction["direction_index"].values,
        direction_columns=_cudf_numeric_columns(direction, list(_DIRECTION_FEATURES)),
        direction_executable=direction["direction_executable"].values,
        branch_decision_index=targets["decision_index"].values,
        branch_action_index=targets["forced_action"].values,
        branch_horizon=targets["horizon"].values,
        branch_repeat_index=targets["repeat_index"].values,
        outcome_columns=_cudf_numeric_columns(targets, list(_OUTCOME_FEATURES)),
        branch_scalar_target=targets["agent_risk_averse"].values,
        registered_horizons=sorted(targets["horizon"].unique().to_arrow().to_pylist()),
        quantile_levels=quantile_levels,
        cvar_alpha=cvar_alpha,
        selected_actions=decision["selected_action"].values,
    )


def _cudf_numeric_columns(frame: Any, names: list[str]) -> dict[str, Any]:
    output = {}
    rows = len(frame)
    for name in names:
        series = frame[name]
        if str(series.dtype).startswith("list"):
            leaves = series.list.leaves.values
            width = int(leaves.size) // max(rows, 1)
            output[name] = leaves.reshape(rows, width)
        else:
            output[name] = series.fillna(0).values
    return output


def _cudf_history_columns(
    history: Any,
    decision: Any,
    names: list[str],
    *,
    decisions: int,
    history_length: int,
) -> dict[str, Any]:
    if history_length == 0:
        return {}
    import cupy as cp

    templates = _cudf_numeric_columns(decision, names)
    values = _cudf_numeric_columns(history, names) if len(history) else {}
    output: dict[str, Any] = {}
    row = history["decision_index"].values.astype(cp.int64) if len(history) else None
    lag = history["history_lag"].values.astype(cp.int64) if len(history) else None
    valid = (lag < history_length) if lag is not None else None
    for name, template in templates.items():
        shape = (decisions, history_length, *template.shape[1:])
        array = cp.zeros(shape, dtype=template.dtype)
        if row is not None and lag is not None and valid is not None:
            array[row[valid], lag[valid]] = values[name][valid]
        output[f"history_{name}"] = array
    present = cp.zeros((decisions, history_length), dtype=cp.float32)
    if row is not None and lag is not None and valid is not None:
        present[row[valid], lag[valid]] = 1.0
    output["history_present"] = present
    return output


def _load_cpu(
    root: Path,
    registry: FeatureRegistry,
    history_length: int,
    quantile_levels: tuple[float, ...],
    cvar_alpha: float,
) -> Phase4TensorBatch:
    try:
        import polars as pl
    except ImportError as exc:
        raise RuntimeError("CPU tensor loading requires Polars") from exc
    decision = pl.read_parquet(root / "decision_context").sort("source_decision_id")
    if decision["source_decision_id"].n_unique() != decision.height:
        raise ValueError("decision context is not unique by source_decision_id")
    decision = decision.with_row_index("decision_index")
    index = decision.select("source_decision_id", "decision_index")
    candidate = pl.read_parquet(root / "candidate_context").join(
        index, on="source_decision_id", how="inner"
    ).sort("decision_index", "action_index")
    direction = pl.read_parquet(root / "direction_context").join(
        index, on="source_decision_id", how="inner"
    ).sort("decision_index", "action_family", "direction_index")
    history = pl.read_parquet(root / "history_context").join(
        index, on="source_decision_id", how="inner"
    )
    targets = pl.read_parquet(root / "branch_targets").join(
        pl.read_parquet(root / "externality_targets"),
        on=[
            "branch_id",
            "source_decision_id",
            "forced_action",
            "repeat_index",
            "horizon",
        ],
        how="left",
        validate="1:1",
    ).join(index, on="source_decision_id", how="inner")
    if targets.select(
        [pl.col(name).is_null().any() for name in _EXTERNALITY_FEATURES]
    ).row(0).count(True):
        raise ValueError("externality target join is incomplete")
    context_names = [
        value.source_column
        for value in registry.for_perspective(FeaturePerspective.AGENT_PRIMARY)
        if value.source_table == "agent_context" and value.source_column in decision.columns
    ]
    oracle_names = [
        value.source_column
        for value in registry.for_perspective(FeaturePerspective.ORACLE_DIAGNOSTIC)
        if value.source_table == "oracle_context" and value.source_column in decision.columns
    ]
    candidate_names = [
        value.source_column
        for value in registry.for_perspective(FeaturePerspective.AGENT_PRIMARY)
        if value.source_table == "candidates"
        and value.source_column in candidate.columns
        and value.source_column not in {"policy_legal", "prechoice_executable"}
    ]
    context_columns = _polars_numeric_columns(decision, context_names)
    context_columns.update(
        _polars_history_columns(
            history,
            decision,
            context_names,
            decisions=decision.height,
            history_length=history_length,
        )
    )
    return assemble_fixed_action_tensors(
        decision_ids=decision["source_decision_id"].to_numpy(),
        seeds=decision["seed"].to_numpy(),
        split_roles=decision["split_role"].to_numpy(),
        outer_folds=decision["outer_fold"].to_numpy(),
        context_columns=context_columns,
        oracle_context_columns=_polars_numeric_columns(decision, oracle_names),
        candidate_decision_index=candidate["decision_index"].to_numpy(),
        candidate_action_index=candidate["action_index"].to_numpy(),
        candidate_columns=_polars_numeric_columns(candidate, candidate_names),
        candidate_executable=(
            candidate["policy_legal"] & candidate["prechoice_executable"]
        ).to_numpy(),
        direction_decision_index=direction["decision_index"].to_numpy(),
        direction_family_index=direction["action_family"].to_numpy(),
        direction_index=direction["direction_index"].to_numpy(),
        direction_columns=_polars_numeric_columns(direction, list(_DIRECTION_FEATURES)),
        direction_executable=direction["direction_executable"].to_numpy(),
        branch_decision_index=targets["decision_index"].to_numpy(),
        branch_action_index=targets["forced_action"].to_numpy(),
        branch_horizon=targets["horizon"].to_numpy(),
        branch_repeat_index=targets["repeat_index"].to_numpy(),
        outcome_columns=_polars_numeric_columns(targets, list(_OUTCOME_FEATURES)),
        branch_scalar_target=targets["agent_risk_averse"].to_numpy(),
        registered_horizons=sorted(targets["horizon"].unique().to_list()),
        quantile_levels=quantile_levels,
        cvar_alpha=cvar_alpha,
        selected_actions=decision["selected_action"].to_numpy(),
    )


def _polars_numeric_columns(frame: Any, names: list[str]) -> dict[str, npt.NDArray[Any]]:
    output = {}
    for name in names:
        series = frame[name]
        if series.dtype.is_nested():
            output[name] = np.asarray(series.to_list())
        else:
            output[name] = series.fill_null(0).to_numpy()
    return output


def _polars_history_columns(
    history: Any,
    decision: Any,
    names: list[str],
    *,
    decisions: int,
    history_length: int,
) -> dict[str, npt.NDArray[Any]]:
    if history_length == 0:
        return {}
    templates = _polars_numeric_columns(decision, names)
    values = _polars_numeric_columns(history, names) if history.height else {}
    output: dict[str, npt.NDArray[Any]] = {}
    row = history["decision_index"].to_numpy().astype(np.int64) if history.height else None
    lag = history["history_lag"].to_numpy().astype(np.int64) if history.height else None
    valid = (lag < history_length) if lag is not None else None
    for name, template in templates.items():
        shape = (decisions, history_length, *template.shape[1:])
        array = np.zeros(shape, dtype=template.dtype)
        if row is not None and lag is not None and valid is not None:
            array[row[valid], lag[valid]] = values[name][valid]
        output[f"history_{name}"] = array
    present = np.zeros((decisions, history_length), dtype=np.float32)
    if row is not None and lag is not None and valid is not None:
        present[row[valid], lag[valid]] = 1.0
    output["history_present"] = present
    return output
