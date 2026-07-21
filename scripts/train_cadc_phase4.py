#!/usr/bin/env python3
"""Train the complete CADC-MORE 2 model suite with grouped cross-fitting."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from owl.cadc.artifacts import atomic_json, sha256_file, write_model_card  # noqa: E402
from owl.cadc.config import load_phase4_config  # noqa: E402
from owl.cadc.features import FeatureRegistry  # noqa: E402
from owl.cadc.gpu_io import to_torch_dlpack  # noqa: E402
from owl.cadc.models import ActionAgnosticBaseline, CADCMore2Suite  # noqa: E402
from owl.cadc.models.ranker import listwise_loss  # noqa: E402
from owl.cadc.models.transition import StructuralTransitionModel  # noqa: E402
from owl.cadc.outcomes import OutcomeRegistry  # noqa: E402
from owl.cadc.pipeline import load_phase4_tensors  # noqa: E402
from owl.cadc.scalarization import quantile_cvar_weights  # noqa: E402
from owl.cadc.schema import EXPECTED_PHASE3_SOURCE_SHA256  # noqa: E402
from owl.experiments.controller import _release_hash  # noqa: E402


def _tensor(value: Any, *, device: str) -> Any:

    tensor = to_torch_dlpack(value)
    return tensor.to(device=device, non_blocking=True)


def _precision_context(precision: str, device: str) -> Any:
    """Return the declared arithmetic context; never silently downgrade FP8."""

    if precision == "fp32" or device == "cpu":
        return nullcontext()
    if precision == "bf16":
        import torch

        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    raise RuntimeError(
        "B200 FP8 is experimental and requires a separately implemented and "
        "certified Transformer Engine parity path; use the B200 BF16 profile"
    )


def _unwrapped(model: Any) -> Any:
    """Return the source module behind ``torch.compile`` for stable artifacts."""

    return getattr(model, "_orig_mod", model)


def _write_training_history(
    fold_root: Path, *, outer_fold: int, member_seeds: tuple[int, ...]
) -> dict[str, Any]:
    """Materialize one compact, typed history table after all members finish."""

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("training history requires PyArrow") from exc
    rows: list[dict[str, Any]] = []
    for member_seed in member_seeds:
        ledger_path = fold_root / f"member-{member_seed}-ledger.json"
        if not ledger_path.is_file():
            raise FileNotFoundError(f"member training ledger missing: {ledger_path}")
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        for value in ledger:
            rows.append(
                {
                    "outer_fold": int(outer_fold),
                    "member_seed": int(member_seed),
                    "epoch": int(value["epoch"]),
                    "train_loss": float(value["train_loss"]),
                    "validation_loss": float(value["validation_loss"]),
                    "learning_rate": float(value["learning_rate"]),
                    "gradient_norm_max": float(value["gradient_norm_max"]),
                    "nonfinite_count": int(value["nonfinite_count"]),
                    "examples_per_second": float(value["examples_per_second"]),
                    "gpu_memory_bytes": int(value["gpu_memory_bytes"]),
                }
            )
    if not rows:
        raise RuntimeError("training history is empty")
    schema = pa.schema(
        [
            ("outer_fold", pa.int16()),
            ("member_seed", pa.int64()),
            ("epoch", pa.int32()),
            ("train_loss", pa.float64()),
            ("validation_loss", pa.float64()),
            ("learning_rate", pa.float64()),
            ("gradient_norm_max", pa.float64()),
            ("nonfinite_count", pa.int64()),
            ("examples_per_second", pa.float64()),
            ("gpu_memory_bytes", pa.int64()),
        ]
    )
    table = pa.Table.from_pylist(rows, schema=schema)
    destination = fold_root / "training_history.parquet"
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    pq.write_table(table, temporary, compression="zstd", use_dictionary=True)
    os.replace(temporary, destination)
    return {
        "path": str(destination),
        "sha256": sha256_file(destination),
        "bytes": destination.stat().st_size,
        "rows": table.num_rows,
        "members": len(member_seeds),
    }


def _new_suite(config: Any, data: dict[str, Any], *, device: str) -> Any:
    structural = StructuralTransitionModel(
        context_dim=data["context"].shape[-1],
        candidate_dim=data["candidates"].shape[-1],
        direction_dim=data["directions"].shape[-1],
        hidden_dim=config.models.hidden_width,
        outcome_dim=data["outcomes"].shape[-1],
        quantile_count=len(config.scalarization.quantiles),
        time_bins=data["outcomes"].shape[1],
        death_causes=4,
        depth=config.models.depth,
        dropout=config.models.dropout,
    )
    return CADCMore2Suite(structural, config.models.hidden_width).to(device)


def _reload_identity(
    source_model: Any,
    checkpoint: Path,
    config: Any,
    data: dict[str, Any],
    rows: Any,
    *,
    device: str,
) -> float:
    """Reload one member in a fresh module and compare deterministic predictions."""

    import torch

    restored = _new_suite(config, data, device=device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    restored.load_state_dict(state, strict=True)
    source = _unwrapped(source_model)
    source.eval()
    restored.eval()
    index = rows[: min(4, rows.numel())]
    horizon = torch.zeros(index.numel(), dtype=torch.long, device=device)
    inputs = (
        data["context"][index],
        data["candidates"][index],
        data["directions"][index],
        data["direction_mask"][index],
        horizon,
    )
    with torch.no_grad(), _precision_context(config.runtime.precision, device):
        expected = source(*inputs)
        actual = restored(*inputs)
    errors = []
    tolerance = 1e-6 if config.runtime.precision == "fp32" else 5e-3
    for name in ("outcome_mean", "rank_score", "competing_risk_logits"):
        torch.testing.assert_close(
            actual[name], expected[name], rtol=tolerance, atol=tolerance
        )
        errors.append(float((actual[name] - expected[name]).abs().max().cpu()))
    return max(errors, default=0.0)


def _train_xgboost_baseline(
    context: Any,
    target: Any,
    train_rows: Any,
    validation_rows: Any,
    *,
    device: str,
    seed: int,
    epochs: int,
    patience: int,
    output: Path,
) -> tuple[dict[str, Any], np.ndarray]:
    """Train a context-only tree baseline through XGBoost's array interface."""

    import xgboost as xgb

    train = xgb.QuantileDMatrix(context[train_rows], label=target[train_rows])
    validation = xgb.QuantileDMatrix(
        context[validation_rows], label=target[validation_rows], ref=train
    )
    parameters = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "device": "cuda" if device == "cuda" else "cpu",
        "max_depth": 6,
        "eta": 0.05,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "seed": int(seed),
        "nthread": 1,
    }
    booster = xgb.train(
        parameters,
        train,
        num_boost_round=max(20, int(epochs)),
        evals=[(validation, "validation")],
        early_stopping_rounds=int(patience),
        verbose_eval=False,
    )
    booster.save_model(output)
    prediction = _to_host(booster.inplace_predict(context)).astype(np.float32)
    restored = xgb.Booster()
    restored.load_model(output)
    restored_prediction = _to_host(restored.inplace_predict(context)).astype(np.float32)
    reload_error = float(np.max(np.abs(prediction - restored_prediction)))
    if reload_error > 1e-6:
        raise AssertionError("XGBoost baseline save/reload identity failed")
    return (
        {
            "role": "viability_baseline_xgboost",
            "path": str(output),
            "sha256": sha256_file(output),
            "best_iteration": int(booster.best_iteration),
            "context_only": True,
            "device": parameters["device"],
            "xgboost_version": xgb.__version__,
            "reload_max_abs_error": reload_error,
        },
        prediction,
    )


def _train_xgboost_survival_baseline(
    context: Any,
    alive_target: Any,
    horizon_values: Any,
    train_rows: Any,
    validation_rows: Any,
    *,
    device: str,
    seed: int,
    epochs: int,
    patience: int,
    output: Path,
) -> tuple[dict[str, Any], np.ndarray]:
    """Train an action-agnostic XGBoost survival-probability comparator."""
    import torch
    import xgboost as xgb

    decisions, horizons = alive_target.shape
    context_slot = context[:, None, :].expand(-1, horizons, -1)
    horizon_scale = horizon_values.to(dtype=context.dtype, device=context.device)
    horizon_scale = horizon_scale / horizon_scale.max().clamp_min(1.0)
    horizon_slot = horizon_scale[None, :, None].expand(decisions, -1, -1)
    design = torch.cat((context_slot, horizon_slot), dim=-1)

    def partition(rows: Any) -> tuple[Any, Any]:
        return design[rows].reshape(-1, design.shape[-1]), alive_target[rows].reshape(-1)

    train_x, train_y = partition(train_rows)
    valid_x, valid_y = partition(validation_rows)
    train = xgb.QuantileDMatrix(train_x, label=train_y)
    validation = xgb.QuantileDMatrix(valid_x, label=valid_y, ref=train)
    parameters = {
        "objective": "reg:logistic",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "device": "cuda" if device == "cuda" else "cpu",
        "max_depth": 6,
        "eta": 0.05,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "seed": int(seed),
        "nthread": 1,
    }
    booster = xgb.train(
        parameters,
        train,
        num_boost_round=max(20, int(epochs)),
        evals=[(validation, "validation")],
        early_stopping_rounds=int(patience),
        verbose_eval=False,
    )
    booster.save_model(output)
    flat_design = design.reshape(-1, design.shape[-1])
    prediction = _to_host(booster.inplace_predict(flat_design)).reshape(
        decisions, horizons
    )
    restored = xgb.Booster()
    restored.load_model(output)
    restored_prediction = _to_host(restored.inplace_predict(flat_design)).reshape(
        decisions, horizons
    )
    reload_error = float(np.max(np.abs(prediction - restored_prediction)))
    if reload_error > 1e-6:
        raise AssertionError("XGBoost survival baseline save/reload identity failed")
    return (
        {
            "role": "survival_baseline_xgboost_action_agnostic",
            "path": str(output),
            "sha256": sha256_file(output),
            "best_iteration": int(booster.best_iteration),
            "context_only_plus_horizon": True,
            "device": parameters["device"],
            "xgboost_version": xgb.__version__,
            "reload_max_abs_error": reload_error,
        },
        prediction.astype(np.float32),
    )


def _to_host(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if type(value).__module__.split(".", maxsplit=1)[0] == "cupy":
        import cupy as cp

        return cp.asnumpy(value)
    return np.asarray(value)


def _train_xgboost_ranker(
    context: Any,
    candidates: Any,
    target: Any,
    mask: Any,
    horizon_values: Any,
    train_rows: Any,
    validation_rows: Any,
    *,
    device: str,
    seed: int,
    epochs: int,
    patience: int,
    output: Path,
    role: str,
) -> tuple[dict[str, Any], np.ndarray]:
    """Fit a fixed-action GPU pairwise rank baseline with sorted query IDs."""

    import torch
    import xgboost as xgb

    decisions, horizons, actions = target.shape
    context_slot = context[:, None, None, :].expand(-1, horizons, actions, -1)
    candidate_slot = candidates[:, None, :, :].expand(-1, horizons, -1, -1)
    horizon_scale = horizon_values.to(dtype=context.dtype, device=context.device)
    horizon_scale = horizon_scale / horizon_scale.max().clamp_min(1.0)
    horizon_slot = horizon_scale[None, :, None, None].expand(
        decisions, -1, actions, -1
    )
    design = torch.cat((context_slot, candidate_slot, horizon_slot), dim=-1)
    query = torch.arange(
        decisions * horizons, device=context.device, dtype=torch.int64
    ).reshape(decisions, horizons, 1).expand(-1, -1, actions)

    def partition(rows: Any) -> tuple[Any, Any, Any]:
        selected_rows = torch.zeros(decisions, dtype=torch.bool, device=context.device)
        selected_rows[rows] = True
        selected = mask & selected_rows[:, None, None]
        selected &= selected.sum(dim=-1, keepdim=True) >= 2
        if not bool(selected.any()):
            raise ValueError(f"{role} has no rank groups in a gated partition")
        return design[selected], target[selected], query[selected]

    train_x, train_y, train_qid = partition(train_rows)
    valid_x, valid_y, valid_qid = partition(validation_rows)
    train = xgb.DMatrix(train_x, label=train_y, qid=train_qid)
    validation = xgb.DMatrix(valid_x, label=valid_y, qid=valid_qid)
    parameters = {
        "objective": "rank:pairwise",
        "eval_metric": "ndcg",
        "tree_method": "hist",
        "device": "cuda" if device == "cuda" else "cpu",
        "max_depth": 6,
        "eta": 0.05,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "seed": int(seed),
        "nthread": 1,
    }
    booster = xgb.train(
        parameters,
        train,
        num_boost_round=max(20, int(epochs)),
        evals=[(validation, "validation")],
        early_stopping_rounds=int(patience),
        verbose_eval=False,
    )
    booster.save_model(output)
    prediction = booster.inplace_predict(design.reshape(-1, design.shape[-1]))
    prediction = _to_host(prediction).reshape(decisions, horizons, actions)
    restored = xgb.Booster()
    restored.load_model(output)
    restored_prediction = restored.inplace_predict(
        design.reshape(-1, design.shape[-1])
    )
    restored_prediction = _to_host(restored_prediction).reshape(
        decisions, horizons, actions
    )
    reload_error = float(np.max(np.abs(prediction - restored_prediction)))
    if reload_error > 1e-6:
        raise AssertionError(f"{role} save/reload identity failed")
    return (
        {
            "role": role,
            "path": str(output),
            "sha256": sha256_file(output),
            "best_iteration": int(booster.best_iteration),
            "device": parameters["device"],
            "primary_grader": role == "candidate_ranker_xgboost_agent",
            "oracle_diagnostic_only": role == "candidate_ranker_xgboost_oracle",
            "xgboost_version": xgb.__version__,
            "reload_max_abs_error": reload_error,
        },
        prediction.astype(np.float32),
    )


def _standardize(value: Any, rows: Any) -> tuple[Any, dict[str, Any]]:
    import torch

    selected = value[rows]
    center = torch.median(selected.reshape(-1, selected.shape[-1]), dim=0).values
    q25 = torch.quantile(selected.reshape(-1, selected.shape[-1]), 0.25, dim=0)
    q75 = torch.quantile(selected.reshape(-1, selected.shape[-1]), 0.75, dim=0)
    scale = (q75 - q25).clamp_min(1e-6)
    transformed = (value - center) / scale
    if not bool(torch.isfinite(transformed).all()):
        raise FloatingPointError("fold transform produced nonfinite values")
    return transformed, {
        "center": center.detach().cpu().tolist(),
        "scale": scale.detach().cpu().tolist(),
    }


def _fixed_batches(rows: Any, batch_size: int) -> Any:
    """Yield fixed-size indices and a padding mask for compile-stable execution."""

    import torch

    if rows.numel() < 1:
        return
    for start in range(0, rows.numel(), batch_size):
        selected = rows[start : start + batch_size]
        actual = selected.numel()
        if actual < batch_size:
            padding = selected[-1:].expand(batch_size - actual)
            selected = torch.cat((selected, padding))
        valid = torch.arange(batch_size, device=rows.device) < actual
        yield selected, valid, actual


def _decision_horizon_batches(
    rows: Any,
    horizon_count: int,
    batch_size: int,
    *,
    mode: str,
) -> Any:
    """Yield fixed-shape decision/horizon indices for GPU-dense training."""

    import torch

    if mode == "sequential":
        for horizon in range(horizon_count):
            for decision, valid, actual in _fixed_batches(rows, batch_size):
                yield (
                    decision,
                    torch.full(
                        (decision.numel(),),
                        horizon,
                        dtype=torch.long,
                        device=rows.device,
                    ),
                    valid,
                    actual,
                )
        return
    if mode != "flattened":
        raise ValueError(f"unknown horizon batching mode: {mode}")
    decisions = rows.repeat(horizon_count)
    horizons = torch.arange(
        horizon_count, dtype=torch.long, device=rows.device
    ).repeat_interleave(rows.numel())
    slots = torch.arange(decisions.numel(), dtype=torch.long, device=rows.device)
    for selected, valid, actual in _fixed_batches(slots, batch_size):
        yield decisions[selected], horizons[selected], valid, actual


def _suite_loss(
    output: dict[str, Any],
    target: Any,
    mask: Any,
    horizon: Any,
    scalar_target: Any,
    outcome_variance: Any,
    scalar_quantiles: Any,
    scalar_cvar: Any,
    quantile_levels: Any,
    cvar_weights: Any,
    decision_valid: Any | None = None,
) -> Any:
    import torch

    if decision_valid is not None:
        mask = mask & decision_valid[:, None]
    valid = mask.to(dtype=target.dtype)
    denominator = valid.sum().clamp_min(1.0)
    squared_error = (output["outcome_mean"] - target) ** 2
    log_scale = output["outcome_log_scale"]
    outcome_nll = 0.5 * (
        (squared_error + outcome_variance) * torch.exp(-2.0 * log_scale)
        + 2.0 * log_scale
    )
    outcome_loss = (outcome_nll * valid[..., None]).sum() / (
        denominator * target.shape[-1]
    )
    scalar = scalar_target
    rank_loss = listwise_loss(output["rank_score"], scalar, mask)
    left, right = torch.triu_indices(22, 22, offset=1, device=mask.device)
    pair_mask = mask[:, left] & mask[:, right]
    pair_delta = output["rank_score"][:, left] - output["rank_score"][:, right]
    target_delta = scalar[:, left] - scalar[:, right]
    pair_signed = torch.where(target_delta >= 0.0, 1.0, -1.0)
    pair_error = torch.nn.functional.softplus(-pair_signed * pair_delta)
    tie = target_delta.abs() <= 1e-6
    pair_error = torch.where(tie, pair_delta.abs(), pair_error)
    pair_weight = target_delta.abs().clamp_min(1e-3) * pair_mask
    pairwise_rank_loss = (pair_error * pair_weight).sum() / pair_weight.sum().clamp_min(1.0)
    family_loss = (((output["family_value"] - scalar) ** 2) * valid).sum() / denominator
    all_cause_logits = output["competing_risk_logits"]
    if hasattr(horizon, "ndim") and horizon.ndim:
        gather_index = horizon[:, None, None, None].expand(
            -1,
            all_cause_logits.shape[1],
            1,
            all_cause_logits.shape[-1],
        )
        cause_logits = all_cause_logits.gather(2, gather_index).squeeze(2)
    else:
        cause_logits = all_cause_logits[..., int(horizon), :]
    cause_target = target[..., 15:20].clamp_min(0.0)
    cause_target = cause_target / cause_target.sum(dim=-1, keepdim=True).clamp_min(1.0)
    survival_loss = (
        -(cause_target * torch.log_softmax(cause_logits, dim=-1)).sum(dim=-1) * valid
    ).sum() / denominator
    quantiles = output["return_quantiles"]
    quantile_residual = scalar_quantiles - quantiles
    quantile_error = torch.maximum(
        quantile_levels * quantile_residual,
        (quantile_levels - 1.0) * quantile_residual,
    )
    quantile_loss = (quantile_error * valid[..., None]).sum() / (
        denominator * quantiles.shape[-1]
    )
    predicted_cvar = (quantiles * cvar_weights).sum(dim=-1)
    cvar_loss = (
        torch.nn.functional.smooth_l1_loss(
            predicted_cvar, scalar_cvar, reduction="none"
        )
        * valid
    ).sum() / denominator
    information_mask = mask.clone()
    action = torch.arange(22, device=mask.device)[None, :]
    information_mask &= (action == 1) | (action == 11)
    information_target = target[..., 8] + target[..., 9]
    information_outputs = output["epistemic_head"]
    information_error = (
        (information_outputs["new_information"] - information_target) ** 2
        + (information_outputs["later_value_improvement"] - scalar) ** 2
        + (information_outputs["cost_adjusted_control_value"] - scalar) ** 2
    ) / 3.0
    information_denominator = information_mask.sum().clamp_min(1)
    information_loss = (
        information_error * information_mask
    ).sum() / information_denominator
    external_target = target[..., 10:15]
    external_prediction = torch.stack(
        tuple(output["externality_head"].values()), dim=-1
    )
    externality_loss = (
        ((external_prediction - external_target) ** 2) * valid[..., None]
    ).sum() / (denominator * external_target.shape[-1])
    return (
        outcome_loss
        + 0.15 * rank_loss
        + 0.15 * pairwise_rank_loss
        + 0.15 * family_loss
        + 0.2 * survival_loss
        + 0.1 * quantile_loss
        + 0.05 * cvar_loss
        + 0.1 * information_loss
        + 0.05 * externality_loss
    )


def _evaluate(
    model: Any,
    data: dict[str, Any],
    rows: Any,
    batch_size: int,
    *,
    precision: str,
    device: str,
    quantile_levels: Any,
    cvar_weights: Any,
    horizon_batching: str,
) -> float:
    import torch

    model.eval()
    loss_sum = None
    batch_count = 0
    with torch.no_grad():
        for index, horizon, batch_valid, _ in _decision_horizon_batches(
            rows,
            data["outcomes"].shape[1],
            batch_size,
            mode=horizon_batching,
        ):
            with _precision_context(precision, device):
                output = model(
                    data["context"][index],
                    data["candidates"][index],
                    data["directions"][index],
                    data["direction_mask"][index],
                    horizon,
                )
                loss = _suite_loss(
                    output,
                    data["outcomes"][index, horizon],
                    data["outcome_mask"][index, horizon]
                    & data["candidate_mask"][index],
                    horizon,
                    data["scalar_targets"][index, horizon],
                    data["outcome_variance"][index, horizon],
                    data["scalar_quantiles"][index, horizon],
                    data["scalar_cvar"][index, horizon],
                    quantile_levels,
                    cvar_weights,
                    batch_valid,
                )
            detached = loss.detach().float()
            loss_sum = detached if loss_sum is None else loss_sum + detached
            batch_count += 1
    if loss_sum is None or batch_count == 0:
        raise ValueError("validation split produced no batches")
    return float((loss_sum / batch_count).cpu())


def _predict_summary(
    model: Any,
    data: dict[str, Any],
    rows: Any,
    batch_size: int,
    *,
    precision: str,
    device: str,
) -> dict[str, np.ndarray]:
    """Run all decision/horizon predictions densely and cross the host boundary once."""

    import torch

    model.eval()
    parts: dict[str, list[Any]] = {
        "rank_score": [],
        "survival_probability": [],
        "cause_probability": [],
        "information_value": [],
        "information_components": [],
        "embedding": [],
        "outcome_mean": [],
        "outcome_log_scale": [],
        "externality": [],
        "return_quantiles": [],
    }
    horizon_count = int(data["outcomes"].shape[1])
    with torch.no_grad():
        for index, horizon, _, actual in _decision_horizon_batches(
            rows,
            horizon_count,
            batch_size,
            mode="flattened",
        ):
            with _precision_context(precision, device):
                output = model(
                    data["context"][index],
                    data["candidates"][index],
                    data["directions"][index],
                    data["direction_mask"][index],
                    horizon,
                )
            cause_index = horizon[:, None, None, None].expand(
                -1,
                output["competing_risk_logits"].shape[1],
                1,
                output["competing_risk_logits"].shape[-1],
            )
            cause_probability = torch.softmax(
                output["competing_risk_logits"]
                .gather(2, cause_index)
                .squeeze(2),
                dim=-1,
            )
            parts["rank_score"].append(output["rank_score"][:actual].detach())
            parts["survival_probability"].append(
                cause_probability[:actual, ..., 0].detach()
            )
            parts["cause_probability"].append(cause_probability[:actual].detach())
            parts["information_value"].append(
                output["epistemic_head"]["cost_adjusted_control_value"][:actual].detach()
            )
            parts["information_components"].append(
                torch.stack(tuple(output["epistemic_head"].values()), dim=-1)[
                    :actual
                ].detach()
            )
            parts["embedding"].append(output["embedding"][:actual].detach())
            parts["outcome_mean"].append(output["outcome_mean"][:actual].detach())
            parts["outcome_log_scale"].append(
                output["outcome_log_scale"][:actual].detach()
            )
            parts["externality"].append(
                torch.stack(tuple(output["externality_head"].values()), dim=-1)[
                    :actual
                ].detach()
            )
            parts["return_quantiles"].append(
                output["return_quantiles"][:actual].detach()
            )

    decision_count = int(rows.numel())
    result: dict[str, np.ndarray] = {}
    for name, chunks in parts.items():
        flat = torch.cat(chunks, dim=0)
        shaped = flat.reshape(horizon_count, decision_count, *flat.shape[1:])
        result[name] = shaped.transpose(0, 1).contiguous().cpu().numpy()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_phase4_config(args.config)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / "training_receipt.json"
    live_progress_path = output / "training_progress.json"
    if args.resume and receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("passed") is True:
            return 0
    import torch

    if config.runtime.target.value != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("target-GPU Phase 4 training requires positive Torch CUDA")
    device = "cuda" if config.runtime.target.value != "cpu" else "cpu"
    _precision_context(config.runtime.precision, device)
    if config.runtime.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cuda.matmul.allow_tf32 = False
    batch = load_phase4_tensors(
        Path(args.dataset) / "canonical_data",
        backend=config.runtime.backend,
        history_length=config.features.history_length,
        quantile_levels=config.scalarization.quantiles,
        cvar_alpha=config.scalarization.cvar_alpha,
    )
    data = {
        "context": _tensor(batch.context, device=device),
        "oracle_context": _tensor(batch.oracle_context, device=device),
        "candidates": _tensor(batch.candidates, device=device),
        "candidate_mask": _tensor(batch.candidate_mask, device=device).bool(),
        "directions": _tensor(batch.directions, device=device),
        "direction_mask": _tensor(batch.direction_mask, device=device).bool(),
        "outcomes": _tensor(batch.outcomes, device=device),
        "outcome_variance": _tensor(batch.outcome_variance, device=device),
        "scalar_targets": _tensor(batch.scalar_targets, device=device),
        "scalar_quantiles": _tensor(batch.scalar_quantiles, device=device),
        "scalar_cvar": _tensor(batch.scalar_cvar, device=device),
        "outcome_mask": _tensor(batch.outcome_mask, device=device).bool(),
        "selected_actions": _tensor(batch.selected_actions, device=device).long(),
        "horizons": _tensor(batch.horizons, device=device).long(),
    }
    data["quantile_levels"] = torch.as_tensor(
        config.scalarization.quantiles,
        dtype=data["scalar_quantiles"].dtype,
        device=device,
    )
    data["cvar_weights"] = torch.as_tensor(
        quantile_cvar_weights(
            config.scalarization.quantiles, alpha=config.scalarization.cvar_alpha
        ),
        dtype=data["scalar_quantiles"].dtype,
        device=device,
    )
    roles = np.asarray(batch.split_roles).astype(str)
    folds = np.asarray(batch.outer_folds, dtype=np.int16)
    model_receipts = []
    history_receipts: list[dict[str, Any]] = []
    fold_metrics: dict[str, Any] = {}
    dataset_manifest_path = (
        Path(args.dataset) / "canonical_data" / "manifests" / "dataset_manifest.json"
    )
    dataset_manifest_sha256 = sha256_file(dataset_manifest_path)
    outer_fold_values = sorted(np.unique(folds[roles == "train"]).tolist())
    total_member_trials = len(outer_fold_values) * len(config.training.member_seeds)
    for outer_fold in outer_fold_values:
        train_rows_np = np.flatnonzero((roles == "train") & (folds != outer_fold))
        test_rows_np = np.flatnonzero((roles == "train") & (folds == outer_fold))
        validation_rows_np = np.flatnonzero(roles == "validation")
        if not train_rows_np.size or not validation_rows_np.size or not test_rows_np.size:
            raise ValueError(f"outer fold {outer_fold} has an empty gated partition")
        train_rows = torch.as_tensor(train_rows_np, device=device, dtype=torch.long)
        validation_rows = torch.as_tensor(
            validation_rows_np, device=device, dtype=torch.long
        )
        fold_root = output / f"outer-{outer_fold}"
        fold_root.mkdir(parents=True, exist_ok=True)
        fold_receipt_path = fold_root / "fold_training_receipt.json"
        if args.resume and fold_receipt_path.is_file():
            completed_fold = json.loads(fold_receipt_path.read_text(encoding="utf-8"))
            if (
                completed_fold.get("passed") is True
                and completed_fold.get("model_spec_sha256")
                == config.model_spec_digest()
                and completed_fold.get("dataset_manifest_sha256")
                == dataset_manifest_sha256
            ):
                for model_receipt in completed_fold.get("models", []):
                    model_path = Path(str(model_receipt["path"]))
                    if not model_path.is_file() or sha256_file(model_path) != model_receipt[
                        "sha256"
                    ]:
                        raise RuntimeError(
                            f"completed fold artifact changed before resume: {model_path}"
                        )
                prediction_path = fold_root / "heldout_predictions.npz"
                if (
                    not prediction_path.is_file()
                    or sha256_file(prediction_path)
                    != completed_fold.get("prediction_sha256")
                ):
                    raise RuntimeError("completed fold prediction changed before resume")
                history = completed_fold.get("training_history", {})
                history_path = Path(str(history.get("path", "")))
                if (
                    not history_path.is_file()
                    or sha256_file(history_path) != history.get("sha256")
                ):
                    raise RuntimeError("completed fold training history changed before resume")
                model_receipts.extend(completed_fold["models"])
                history_receipts.append(history)
                fold_metrics[str(outer_fold)] = completed_fold["metrics"]
                continue
        context, context_transform = _standardize(data["context"], train_rows)
        oracle_context, oracle_transform = _standardize(
            data["oracle_context"], train_rows
        )
        candidates, candidate_transform = _standardize(data["candidates"], train_rows)
        directions, direction_transform = _standardize(data["directions"], train_rows)
        fold_data = {
            **data,
            "context": context,
            "candidates": candidates,
            "directions": directions,
        }
        atomic_json(
            fold_root / "fold_transform.json",
            {
                "context": context_transform,
                "oracle_context": oracle_transform,
                "candidates": candidate_transform,
                "directions": direction_transform,
                "fit_rows": train_rows_np.tolist(),
            },
        )
        baseline_count = data["outcome_mask"].sum(dim=2).clamp_min(1)
        baseline_target = (
            data["outcomes"] * data["outcome_mask"][..., None]
        ).sum(dim=2) / baseline_count[..., None]
        baseline_flat = baseline_target.reshape(baseline_target.shape[0], -1)
        baseline = ActionAgnosticBaseline(
            data["context"].shape[-1], baseline_flat.shape[-1]
        ).to(device)
        baseline_optimizer = torch.optim.AdamW(
            baseline.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        for _ in range(min(20, config.training.epochs)):
            baseline_optimizer.zero_grad(set_to_none=True)
            with _precision_context(config.runtime.precision, device):
                prediction = baseline(context[train_rows])
                loss = torch.nn.functional.huber_loss(
                    prediction, baseline_flat[train_rows]
                )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("action-agnostic baseline loss is nonfinite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                baseline.parameters(), config.training.gradient_clip
            )
            baseline_optimizer.step()
        baseline_path = fold_root / "action_agnostic_baseline.pt"
        torch.save(baseline.state_dict(), baseline_path)
        restored_baseline = ActionAgnosticBaseline(
            data["context"].shape[-1], baseline_flat.shape[-1]
        ).to(device)
        restored_baseline.load_state_dict(
            torch.load(baseline_path, map_location=device, weights_only=True),
            strict=True,
        )
        baseline.eval()
        restored_baseline.eval()
        with torch.no_grad(), _precision_context(config.runtime.precision, device):
            neural_baseline_prediction = _to_host(baseline(context)).astype(np.float32)
            restored_baseline_prediction = _to_host(
                restored_baseline(context)
            ).astype(np.float32)
        baseline_reload_error = float(
            np.max(np.abs(neural_baseline_prediction - restored_baseline_prediction))
        )
        baseline_tolerance = (
            1e-6 if config.runtime.precision == "fp32" else 5e-3
        )
        if baseline_reload_error > baseline_tolerance:
            raise AssertionError("neural viability baseline save/reload identity failed")
        model_receipts.append(
            {
                "role": "viability_baseline",
                "outer_fold": outer_fold,
                "path": str(baseline_path),
                "sha256": sha256_file(baseline_path),
                "reload_max_abs_error": baseline_reload_error,
            }
        )
        if config.models.xgboost_enabled:
            xgboost_path = fold_root / "action_agnostic_xgboost.json"
            xgboost_receipt, xgboost_baseline_prediction = _train_xgboost_baseline(
                context,
                baseline_flat,
                train_rows,
                validation_rows,
                device=device,
                seed=config.training.member_seeds[0] + int(outer_fold),
                epochs=config.training.epochs,
                patience=config.training.early_stopping_patience,
                output=xgboost_path,
            )
            xgboost_receipt["outer_fold"] = int(outer_fold)
            model_receipts.append(xgboost_receipt)
            survival_count = data["outcome_mask"].sum(dim=2).clamp_min(1)
            survival_target = (
                data["outcomes"][..., 5] * data["outcome_mask"]
            ).sum(dim=2) / survival_count
            survival_receipt, xgboost_survival_prediction = (
                _train_xgboost_survival_baseline(
                    context,
                    survival_target,
                    data["horizons"],
                    train_rows,
                    validation_rows,
                    device=device,
                    seed=config.training.member_seeds[0] + 50 + int(outer_fold),
                    epochs=config.training.epochs,
                    patience=config.training.early_stopping_patience,
                    output=fold_root / "survival_baseline_xgboost.json",
                )
            )
            survival_receipt["outer_fold"] = int(outer_fold)
            model_receipts.append(survival_receipt)
            target_scalar_device = data["scalar_targets"]
            rank_mask = data["outcome_mask"] & data["candidate_mask"][:, None, :]
            agent_rank_receipt, xgboost_agent_rank = _train_xgboost_ranker(
                context,
                candidates,
                target_scalar_device,
                rank_mask,
                data["horizons"],
                train_rows,
                validation_rows,
                device=device,
                seed=config.training.member_seeds[0] + 100 + int(outer_fold),
                epochs=config.training.epochs,
                patience=config.training.early_stopping_patience,
                output=fold_root / "candidate_ranker_xgboost_agent.json",
                role="candidate_ranker_xgboost_agent",
            )
            model_receipts.append({**agent_rank_receipt, "outer_fold": int(outer_fold)})
            oracle_rank_receipt, xgboost_oracle_rank = _train_xgboost_ranker(
                torch.cat((context, oracle_context), dim=-1),
                candidates,
                target_scalar_device,
                rank_mask,
                data["horizons"],
                train_rows,
                validation_rows,
                device=device,
                seed=config.training.member_seeds[0] + 200 + int(outer_fold),
                epochs=config.training.epochs,
                patience=config.training.early_stopping_patience,
                output=fold_root / "candidate_ranker_xgboost_oracle.json",
                role="candidate_ranker_xgboost_oracle",
            )
            model_receipts.append({**oracle_rank_receipt, "outer_fold": int(outer_fold)})
        else:
            xgboost_baseline_prediction = np.full_like(
                neural_baseline_prediction, np.nan
            )
            xgboost_survival_prediction = np.full(
                tuple(data["outcomes"].shape[:2]), np.nan, dtype=np.float32
            )
            xgboost_agent_rank = np.full(
                tuple(data["outcomes"].shape[:3]), np.nan, dtype=np.float32
            )
            xgboost_oracle_rank = np.full_like(xgboost_agent_rank, np.nan)
        member_metrics = []
        member_predictions = []
        for member_seed in config.training.member_seeds:
            trial_path = fold_root / f"member-{member_seed}-trial.json"
            atomic_json(
                trial_path,
                {
                    "schema_version": "owl.cadc.phase4-training-trial.v1",
                    "trial_id": f"outer-{outer_fold}-member-{member_seed}",
                    "parent_experiment_id": config.model_spec_digest(),
                    "config_sha256": config.canonical_digest(),
                    "split_sha256": dataset_manifest_sha256,
                    "model_role": "cadc_more2_suite",
                    "status": "incomplete_or_interrupted",
                    "failure_class": None,
                    "outer_fold": int(outer_fold),
                    "member_seed": int(member_seed),
                    "phase5_locked": True,
                },
            )
            torch.manual_seed(member_seed)
            torch.cuda.manual_seed_all(member_seed)
            model = _new_suite(config, fold_data, device=device)
            if config.runtime.compile:
                model = torch.compile(model, dynamic=False)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.training.learning_rate,
                weight_decay=config.training.weight_decay,
            )
            best_loss = float("inf")
            best_state = None
            patience = 0
            ledger = []
            start_epoch = 0
            progress_path = fold_root / f"member-{member_seed}-progress.pt"
            if args.resume and progress_path.is_file():
                progress = torch.load(
                    progress_path, map_location=device, weights_only=False
                )
                expected_progress = (
                    progress.get("model_spec_sha256") == config.model_spec_digest()
                    and progress.get("dataset_manifest_sha256")
                    == dataset_manifest_sha256
                    and int(progress.get("outer_fold", -1)) == int(outer_fold)
                    and int(progress.get("member_seed", -1)) == int(member_seed)
                    and progress.get("precision") == config.runtime.precision
                )
                if not expected_progress:
                    raise RuntimeError("member resume checkpoint scope mismatch")
                _unwrapped(model).load_state_dict(progress["model_state"], strict=True)
                optimizer.load_state_dict(progress["optimizer_state"])
                best_state = progress["best_state"]
                best_loss = float(progress["best_loss"])
                patience = int(progress["patience"])
                ledger = list(progress["ledger"])
                start_epoch = int(progress["next_epoch"])
                if patience >= config.training.early_stopping_patience:
                    start_epoch = config.training.epochs
                torch.set_rng_state(progress["torch_rng_state"].cpu())
                if device == "cuda":
                    torch.cuda.set_rng_state_all(progress["cuda_rng_state"])
            for epoch in range(start_epoch, config.training.epochs):
                model.train()
                generator = torch.Generator(device=device).manual_seed(
                    member_seed + epoch
                )
                permutation = torch.randperm(
                    train_rows.numel(), generator=generator, device=device
                )
                order = train_rows[permutation]
                started = time.perf_counter()
                epoch_loss_device = torch.zeros((), device=device, dtype=torch.float32)
                epoch_gradient_max_device = torch.zeros(
                    (), device=device, dtype=torch.float32
                )
                steps = 0
                for index, horizon, batch_valid, _ in _decision_horizon_batches(
                    order,
                    data["outcomes"].shape[1],
                    config.runtime.batch_size,
                    mode=config.runtime.training_horizon_batching,
                ):
                    optimizer.zero_grad(set_to_none=True)
                    with _precision_context(config.runtime.precision, device):
                        prediction = model(
                            context[index],
                            candidates[index],
                            directions[index],
                            data["direction_mask"][index],
                            horizon,
                        )
                        loss = _suite_loss(
                            prediction,
                            data["outcomes"][index, horizon],
                            data["outcome_mask"][index, horizon]
                            & data["candidate_mask"][index],
                            horizon,
                            data["scalar_targets"][index, horizon],
                            data["outcome_variance"][index, horizon],
                            data["scalar_quantiles"][index, horizon],
                            data["scalar_cvar"][index, horizon],
                            data["quantile_levels"],
                            data["cvar_weights"],
                            batch_valid,
                        )
                    loss_finite = torch.isfinite(loss.detach())
                    if device == "cuda":
                        torch._assert_async(loss_finite, "CADC-MORE 2 loss is nonfinite")
                    elif not bool(loss_finite):
                        raise FloatingPointError("CADC-MORE 2 loss is nonfinite")
                    loss.backward()
                    gradient = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.training.gradient_clip
                    )
                    gradient_finite = torch.isfinite(gradient.detach())
                    if device == "cuda":
                        torch._assert_async(
                            gradient_finite,
                            "CADC-MORE 2 gradient is nonfinite",
                        )
                    elif not bool(gradient_finite):
                        raise FloatingPointError("CADC-MORE 2 gradient is nonfinite")
                    epoch_gradient_max_device = torch.maximum(
                        epoch_gradient_max_device,
                        gradient.detach().float(),
                    )
                    optimizer.step()
                    epoch_loss_device += loss.detach().float()
                    steps += 1
                validation_loss = _evaluate(
                    model,
                    fold_data,
                    validation_rows,
                    config.runtime.batch_size,
                    precision=config.runtime.precision,
                    device=device,
                    quantile_levels=data["quantile_levels"],
                    cvar_weights=data["cvar_weights"],
                    horizon_batching=config.runtime.training_horizon_batching,
                )
                elapsed = max(time.perf_counter() - started, 1e-12)
                epoch_loss = float((epoch_loss_device / max(1, steps)).cpu())
                epoch_gradient_max = float(epoch_gradient_max_device.cpu())
                ledger.append(
                    {
                        "epoch": epoch,
                        "train_loss": epoch_loss,
                        "validation_loss": validation_loss,
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                        "gradient_norm_max": epoch_gradient_max,
                        "nonfinite_count": 0,
                        "examples_per_second": (
                            train_rows.numel() * data["outcomes"].shape[1] / elapsed
                        ),
                        "elapsed_seconds": elapsed,
                        "gpu_memory_bytes": (
                            int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0
                        ),
                    }
                )
                completed_member_trials = sum(
                    value.get("role") == "cadc_more2_suite"
                    for value in model_receipts
                )
                atomic_json(
                    live_progress_path,
                    {
                        "schema_version": "owl.cadc.phase4-training-progress.v1",
                        "passed": False,
                        "stage": "training_member",
                        "outer_fold": int(outer_fold),
                        "member_seed": int(member_seed),
                        "epoch_completed": int(epoch + 1),
                        "epochs_configured": int(config.training.epochs),
                        "completed_member_trials": int(completed_member_trials),
                        "total_member_trials": int(total_member_trials),
                        "latest_epoch_seconds": float(elapsed),
                        "mean_epoch_seconds_current_member": float(
                            np.mean(
                                [value["elapsed_seconds"] for value in ledger],
                                dtype=np.float64,
                            )
                        ),
                        "phase5_locked": True,
                    },
                )
                if validation_loss < best_loss:
                    best_loss = validation_loss
                    best_state = copy.deepcopy(_unwrapped(model).state_dict())
                    patience = 0
                else:
                    patience += 1
                progress_payload = {
                    "schema_version": "owl.cadc.phase4-member-progress.v1",
                    "model_spec_sha256": config.model_spec_digest(),
                    "dataset_manifest_sha256": dataset_manifest_sha256,
                    "outer_fold": int(outer_fold),
                    "member_seed": int(member_seed),
                    "precision": config.runtime.precision,
                    "next_epoch": epoch + 1,
                    "model_state": _unwrapped(model).state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_state": best_state,
                    "best_loss": best_loss,
                    "patience": patience,
                    "ledger": ledger,
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": (
                        torch.cuda.get_rng_state_all() if device == "cuda" else []
                    ),
                }
                should_checkpoint = (
                    (epoch + 1) % config.runtime.training_checkpoint_interval == 0
                    or epoch + 1 == config.training.epochs
                    or patience >= config.training.early_stopping_patience
                )
                if should_checkpoint:
                    temporary_progress = progress_path.with_name(
                        f".{progress_path.name}.tmp.{os.getpid()}"
                    )
                    torch.save(progress_payload, temporary_progress)
                    os.replace(temporary_progress, progress_path)
                if patience >= config.training.early_stopping_patience:
                    break
            if best_state is None:
                raise RuntimeError("member training produced no valid checkpoint")
            _unwrapped(model).load_state_dict(best_state)
            member_path = fold_root / f"cadc_more2_member-{member_seed}.pt"
            temporary = fold_root / f".{member_path.name}.tmp.{os.getpid()}"
            torch.save(_unwrapped(model).state_dict(), temporary)
            os.replace(temporary, member_path)
            atomic_json(fold_root / f"member-{member_seed}-ledger.json", ledger)
            reload_error = _reload_identity(
                model,
                member_path,
                config,
                fold_data,
                train_rows,
                device=device,
            )
            model_receipts.append(
                {
                    "role": "cadc_more2_suite",
                    "outer_fold": outer_fold,
                    "member_seed": member_seed,
                    "path": str(member_path),
                    "sha256": sha256_file(member_path),
                    "best_validation_loss": best_loss,
                    "parameter_count": int(
                        sum(value.numel() for value in _unwrapped(model).parameters())
                    ),
                    "checkpoint_bytes": member_path.stat().st_size,
                    "reload_max_abs_error": reload_error,
                }
            )
            atomic_json(
                trial_path,
                {
                    "schema_version": "owl.cadc.phase4-training-trial.v1",
                    "trial_id": f"outer-{outer_fold}-member-{member_seed}",
                    "parent_experiment_id": config.model_spec_digest(),
                    "config_sha256": config.canonical_digest(),
                    "split_sha256": dataset_manifest_sha256,
                    "model_role": "cadc_more2_suite",
                    "status": "completed",
                    "failure_class": None,
                    "outer_fold": int(outer_fold),
                    "member_seed": int(member_seed),
                    "best_step": int(
                        min(
                            range(len(ledger)),
                            key=lambda index: ledger[index]["validation_loss"],
                        )
                    ),
                    "best_validation_loss": best_loss,
                    "peak_memory_bytes": max(
                        (value["gpu_memory_bytes"] for value in ledger), default=0
                    ),
                    "checkpoint_sha256": sha256_file(member_path),
                    "phase5_locked": True,
                },
            )
            member_metrics.append(best_loss)
            evaluation_rows_np = np.arange(roles.size, dtype=np.int64)
            evaluation_rows = torch.as_tensor(
                evaluation_rows_np, device=device, dtype=torch.long
            )
            member_predictions.append(
                _predict_summary(
                    model,
                    fold_data,
                    evaluation_rows,
                    config.runtime.batch_size,
                    precision=config.runtime.precision,
                    device=device,
                )
            )
            atomic_json(
                live_progress_path,
                {
                    "schema_version": "owl.cadc.phase4-training-progress.v1",
                    "passed": False,
                    "stage": "member_completed",
                    "outer_fold": int(outer_fold),
                    "member_seed": int(member_seed),
                    "epoch_completed": int(len(ledger)),
                    "epochs_configured": int(config.training.epochs),
                    "completed_member_trials": int(
                        sum(
                            value.get("role") == "cadc_more2_suite"
                            for value in model_receipts
                        )
                    ),
                    "total_member_trials": int(total_member_trials),
                    "phase5_locked": True,
                },
            )
        evaluation_rows_np = np.arange(roles.size, dtype=np.int64)
        target = batch.outcomes[evaluation_rows_np]
        target_mask = batch.outcome_mask[evaluation_rows_np]
        repeat_count = batch.repeat_count[evaluation_rows_np]
        seeds_host = batch.seeds
        if type(target).__module__.split(".", maxsplit=1)[0] == "cupy":
            import cupy as cp

            target = cp.asnumpy(target)
            target_mask = cp.asnumpy(target_mask)
            repeat_count = cp.asnumpy(repeat_count)
            seeds_host = cp.asnumpy(seeds_host)
        target = np.asarray(target)
        target_scalar = _to_host(data["scalar_targets"])
        np.savez_compressed(
            fold_root / "heldout_predictions.npz",
            decision_ids=np.asarray(batch.decision_ids)[evaluation_rows_np],
            horizons=_to_host(batch.horizons),
            row_indices=evaluation_rows_np,
            split_roles=roles[evaluation_rows_np],
            outer_folds=folds[evaluation_rows_np],
            seeds=np.asarray(seeds_host)[evaluation_rows_np],
            rank_score=np.stack(
                [value["rank_score"] for value in member_predictions], axis=0
            ),
            survival_probability=np.stack(
                [value["survival_probability"] for value in member_predictions], axis=0
            ),
            cause_probability=np.stack(
                [value["cause_probability"] for value in member_predictions], axis=0
            ),
            information_value=np.stack(
                [value["information_value"] for value in member_predictions], axis=0
            ),
            information_components=np.stack(
                [value["information_components"] for value in member_predictions],
                axis=0,
            ),
            outcome_mean=np.stack(
                [value["outcome_mean"] for value in member_predictions], axis=0
            ),
            outcome_log_scale=np.stack(
                [value["outcome_log_scale"] for value in member_predictions], axis=0
            ),
            externality_prediction=np.stack(
                [value["externality"] for value in member_predictions], axis=0
            ),
            return_quantiles=np.stack(
                [value["return_quantiles"] for value in member_predictions], axis=0
            ),
            embedding=np.mean(
                np.stack([value["embedding"] for value in member_predictions], axis=0),
                axis=0,
                dtype=np.float64,
            ).astype(np.float32),
            target_scalar=target_scalar,
            target_outcomes=target,
            target_outcome_variance=_to_host(data["outcome_variance"]),
            target_scalar_quantiles=_to_host(data["scalar_quantiles"]),
            target_scalar_cvar=_to_host(data["scalar_cvar"]),
            target_alive=target[..., 5],
            target_death_cause_probability=target[..., 15:20],
            target_mask=np.asarray(target_mask),
            repeat_count=np.asarray(repeat_count),
            selected_actions=_to_host(data["selected_actions"]),
            neural_viability_baseline=neural_baseline_prediction,
            xgboost_viability_baseline=xgboost_baseline_prediction,
            xgboost_survival_baseline=xgboost_survival_prediction,
            action_agnostic_target=_to_host(baseline_target),
            xgboost_agent_rank=xgboost_agent_rank,
            xgboost_oracle_rank=xgboost_oracle_rank,
        )
        fold_metrics[str(outer_fold)] = {
            "members": len(member_metrics),
            "validation_loss_mean": float(np.mean(member_metrics, dtype=np.float64)),
            "outer_test_rows_sealed_during_fit": int(test_rows_np.size),
        }
        atomic_json(
            fold_root / "model_config.json",
            {
                "schema_version": "owl.cadc.phase4-model-config.v1",
                "model_spec_sha256": config.model_spec_digest(),
                "outer_fold": int(outer_fold),
                "context_dim": int(context.shape[-1]),
                "candidate_dim": int(candidates.shape[-1]),
                "direction_dim": int(directions.shape[-1]),
                "outcome_dim": int(data["outcomes"].shape[-1]),
                "actions": 22,
                "precision": config.runtime.precision,
            },
        )
        atomic_json(fold_root / "feature_schema.json", FeatureRegistry().manifest())
        atomic_json(fold_root / "outcome_registry.json", OutcomeRegistry().manifest())
        training_history = _write_training_history(
            fold_root,
            outer_fold=int(outer_fold),
            member_seeds=tuple(config.training.member_seeds),
        )
        history_receipts.append(training_history)
        fold_models = [
            value
            for value in model_receipts
            if int(value.get("outer_fold", -1)) == int(outer_fold)
        ]
        atomic_json(
            fold_root / "ensemble_manifest.json",
            {
                "schema_version": "owl.cadc.phase4-ensemble-manifest.v1",
                "passed": True,
                "outer_fold": int(outer_fold),
                "configured_members": config.models.ensemble_members,
                "completed_members": sum(
                    value.get("role") == "cadc_more2_suite" for value in fold_models
                ),
                "models": fold_models,
                "model_spec_sha256": config.model_spec_digest(),
                "phase5_locked": True,
            },
        )
        write_model_card(
            fold_root / "model_card.md",
            {
                "model_name": f"CADC-MORE 2 development outer fold {outer_fold}",
                "role": "context-sensitive simulated action grader",
                "source_sha256": EXPECTED_PHASE3_SOURCE_SHA256,
                "dataset_sha256": dataset_manifest_sha256,
                "feature_schema_digest": FeatureRegistry().digest,
                "outcome_registry_digest": OutcomeRegistry().digest,
                "split_registry_digest": dataset_manifest_sha256,
                "model_spec_sha256": config.model_spec_digest(),
                "intended_use": (
                    "Development-only grading of simulated context-sensitive action "
                    "choice on the immutable 22-action OWL/RAQIC axis."
                ),
                "forbidden_use": [
                    "real-world decisions",
                    "consciousness claims",
                    "Phase 5 or Phase 6 confirmatory inference",
                    "feeding predictions into factual simulation policy or world state",
                ],
                "feature_perspective": {
                    "primary": "agent-visible pre-choice only",
                    "oracle": "diagnostic only",
                    "raqic_mechanism": "mediation/moderation analysis only",
                    "execution": "post-choice analysis only",
                },
                "supported_actions": [
                    "all 22 immutable actions, including SENSE, FLEE, and PURSUE"
                ],
                "supported_horizons": list(config.corpus.horizons),
                "outcomes": [value.name for value in OutcomeRegistry().definitions],
                "split_design": {
                    "outer_fold": int(outer_fold),
                    "unit": list(config.splits.group_fields),
                    "calibration": "separate seed role; not used for fitting",
                    "phase5_phase6_seeds": "cryptographically registered and sealed",
                },
                "metrics": {
                    "training": fold_metrics[str(outer_fold)],
                    "held_out_family_context_metrics": (
                        "materialized by evaluate_cadc_phase4.py after calibration"
                    ),
                },
                "calibration": (
                    "separate Mondrian conformal and isotonic artifacts are required "
                    "before a scored development candidate is valid"
                ),
                "support": (
                    "abstain with insufficient_counterfactual_support or OOD outside "
                    "the separately fitted support index"
                ),
                "negative_controls": (
                    "action/target/temporal/repeat controls must collapse in the "
                    "independent Phase 4 acceptance run"
                ),
                "hardware_environment": {
                    "target": config.runtime.target.value,
                    "precision": config.runtime.precision,
                    "portable_common_path": "H100/H200/B200 BF16",
                    "fp8": "unsupported until separate B200 parity certificate",
                },
                "known_failure_cases": [
                    "insufficient action-family or repeat support",
                    "feature out of distribution",
                    "wide conformal interval or high ensemble disagreement",
                    "later-action-change epistemic label unavailable in Phase 3 evidence",
                ],
                "reproducibility_limits": (
                    "Exact source/data/model/environment hashes are required; Phase 4 "
                    "is development evidence and Phase 5 remains locked."
                ),
                "limitations": [
                    "simulated evidence only",
                    "later-action-change epistemic component marked unsupported_evidence",
                    "untouched seeds require a separate Phase 5 freeze decision",
                ],
                "claims_boundary": (
                    "This model grades behavior inside a simulation and does not "
                    "establish consciousness."
                ),
            },
        )
        atomic_json(
            fold_receipt_path,
            {
                "schema_version": "owl.cadc.phase4-fold-training-receipt.v1",
                "passed": True,
                "outer_fold": int(outer_fold),
                "model_spec_sha256": config.model_spec_digest(),
                "dataset_manifest_sha256": dataset_manifest_sha256,
                "models": fold_models,
                "training_history": training_history,
                "metrics": fold_metrics[str(outer_fold)],
                "prediction_sha256": sha256_file(
                    fold_root / "heldout_predictions.npz"
                ),
                "phase5_locked": True,
            },
        )
    receipt = {
        "schema_version": "owl.cadc.phase4-training-receipt.v1",
        "passed": True,
        "classification": "PHASE4_MODELS_TRAINED_DEVELOPMENT_ONLY",
        "phase3_source_sha256": EXPECTED_PHASE3_SOURCE_SHA256,
        "phase4_source_sha256": _release_hash(ROOT),
        "config_sha256": config.canonical_digest(),
        "corpus_contract_sha256": config.corpus_digest(),
        "model_spec_sha256": config.model_spec_digest(),
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "target": config.runtime.target.value,
        "precision": config.runtime.precision,
        "fold_metrics": fold_metrics,
        "models": model_receipts,
        "training_histories": history_receipts,
        "component_status": {
            "action_agnostic_viability": "trained",
            "structural_transition_ensemble": "trained",
            "action_family_experts": "trained",
            "pairwise_listwise_ranker": "trained",
            "survival_competing_risk": "trained",
            "survival_action_agnostic_xgboost": "trained",
            "epistemic_new_information": "trained",
            "epistemic_later_value": "trained",
            "epistemic_cost_adjusted_control_value": "trained",
            "epistemic_later_action_change": "unsupported_evidence",
            "externality": "trained",
            "oracle_ranker": "diagnostic_only",
            "doubly_robust_factual_augmentation": "implemented_optional_secondary",
        },
        "unsupported_evidence": {
            "epistemic_later_action_change": (
                "factual v2 information_followups records linkage status but not the "
                "later selected action inside each counterfactual branch"
            )
        },
        "phase5_locked": True,
    }
    atomic_json(receipt_path, receipt)
    atomic_json(
        live_progress_path,
        {
            "schema_version": "owl.cadc.phase4-training-progress.v1",
            "passed": True,
            "stage": "completed",
            "completed_member_trials": int(total_member_trials),
            "total_member_trials": int(total_member_trials),
            "phase5_locked": True,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
