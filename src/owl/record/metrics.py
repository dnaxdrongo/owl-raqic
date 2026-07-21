"""Metrics collection and tabular recording interfaces.

The metrics layer is read-only with respect to ``WorldState``. It extracts
scalar diagnostics from the dense physical, possibility, communication, and
fractal/mosaic layers so headless runs can be inspected without opening a GUI.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np

from owl.core.actions import DIAGONAL_MOVES, MOVE_DELTAS, Action, SignalChannel
from owl.core.config import SimulationConfig
from owl.core.state import WorldState, action_shape, channel_shape, field_shape


def _channel_sum(state: WorldState, channel: SignalChannel, cfg: SimulationConfig) -> float:
    """Return total signal mass for a configured channel, or zero if absent."""
    _, _, channels = channel_shape(state)
    idx = int(channel)
    if idx >= min(channels, cfg.communication.num_channels):
        return 0.0
    return float(np.sum(state.signal[..., idx], dtype=np.float64))


def _safe_alive_mean(values: np.ndarray, alive: np.ndarray) -> float:
    """Return finite mean over alive cells, or zero for an empty population."""
    if not np.any(alive):
        return 0.0
    return float(np.mean(values[alive], dtype=np.float64))


def _action_fraction(state: WorldState, action: Action, alive: np.ndarray) -> float:
    """Return the fraction of living cells currently actualizing ``action``."""
    alive_count = int(np.count_nonzero(alive))
    if alive_count <= 0:
        return 0.0
    return float(np.count_nonzero((state.readout == int(action)) & alive) / alive_count)


def _base_collect_metrics(state: WorldState, cfg: SimulationConfig) -> dict[str, Any]:
    """Collect one tick of scalar diagnostics.

    Parameters
    ----------
    state:
        Runtime dense state. This function does not mutate state.
    cfg:
        Simulation coefficients used to interpret channel count and resource
        scale.

    Returns
    -------
    dict
        JSON-serializable scalar diagnostics for one tick. The metrics keep the
        three model layers visible: physical survival, communication/possibility,
        and patch/global integration.
    """
    h, w = field_shape(state)
    ah, aw, actions = action_shape(state)
    ch, cw, channels = channel_shape(state)
    if (ah, aw) != (h, w):
        raise ValueError("state.possibility spatial shape must match health shape")
    if (ch, cw) != (h, w):
        raise ValueError("state.signal spatial shape must match health shape")
    if actions != len(Action):
        raise ValueError(f"state.possibility action axis must equal len(Action)={len(Action)}")
    if channels != cfg.communication.num_channels:
        raise ValueError("state.signal channel axis must match cfg.communication.num_channels")

    alive = (state.health > 0.0) & (~state.obstacle)
    alive_count = int(np.count_nonzero(alive))
    total_cells = int(h * w)
    patch_integration = np.asarray(state.patches.integration, dtype=np.float32)

    possibility_entropy = 0.0
    if alive_count:
        P = np.clip(state.possibility[alive], 0.0, 1.0)
        sums = np.sum(P, axis=-1, keepdims=True)
        P = np.divide(P, sums, out=np.zeros_like(P), where=sums > cfg.actions.epsilon)
        positive = P > 0.0
        entropy = -np.sum(
            np.where(positive, P * np.log(P + cfg.actions.epsilon), 0.0),
            axis=-1,
        ) / np.log(float(len(Action)))
        possibility_entropy = float(np.mean(np.clip(entropy, 0.0, 1.0), dtype=np.float64))

    move_actions = tuple(MOVE_DELTAS.keys())
    diagonal_actions = tuple(DIAGONAL_MOVES)
    movement_fraction = (
        float(
            sum(np.count_nonzero((state.readout == int(action)) & alive) for action in move_actions)
            / max(alive_count, 1)
        )
        if alive_count
        else 0.0
    )
    diagonal_movement_fraction = (
        float(
            sum(
                np.count_nonzero((state.readout == int(action)) & alive)
                for action in diagonal_actions
            )
            / max(alive_count, 1)
        )
        if alive_count
        else 0.0
    )
    mean_starvation_debt = (
        _safe_alive_mean(state.starvation_debt, alive)
        if isinstance(state.starvation_debt, np.ndarray)
        else 0.0
    )
    mean_last_intake = (
        _safe_alive_mean(state.last_intake, alive)
        if isinstance(state.last_intake, np.ndarray)
        else 0.0
    )
    mean_movement_loop_score = (
        _safe_alive_mean(state.movement_loop_score, alive)
        if isinstance(state.movement_loop_score, np.ndarray)
        else 0.0
    )

    identity_duplicate_count = 0
    if alive_count:
        ids = state.occupancy[alive]
        ids = ids[ids >= 0]
        if ids.size:
            _, counts = np.unique(ids, return_counts=True)
            identity_duplicate_count = int(np.sum(np.maximum(counts - 1, 0)))

    def _safe_patch_mean(name: str) -> float:
        arr = getattr(state.patches, name, None)
        if isinstance(arr, np.ndarray) and arr.size:
            return float(np.mean(np.clip(arr, 0.0, 1.0), dtype=np.float64))
        return 0.0

    def _safe_state_mean(name: str) -> float:
        arr = getattr(state, name, None)
        if isinstance(arr, np.ndarray):
            return _safe_alive_mean(arr if arr.ndim == 2 else np.mean(arr, axis=-1), alive)
        return 0.0

    starving_food_feed_probability = 0.0
    if (
        alive_count
        and isinstance(state.pre_starvation_debt, np.ndarray)
        and isinstance(state.pre_food, np.ndarray)
    ):
        hungry_food = alive & (state.pre_starvation_debt > 0.35) & (state.pre_food > 0.02)
        if np.any(hungry_food):
            starving_food_feed_probability = float(
                np.mean(state.possibility[hungry_food, int(Action.FEED)], dtype=np.float64)
            )

    low_boundary_repair_probability = 0.0
    if alive_count:
        low_boundary = alive & (state.boundary < 0.45)
        if np.any(low_boundary):
            low_boundary_repair_probability = float(
                np.mean(state.possibility[low_boundary, int(Action.REPAIR)], dtype=np.float64)
            )

    crowded_reproduction_probability = 0.0
    crowd = getattr(state.patches, "patch_carrying_pressure", None)
    if alive_count and isinstance(crowd, np.ndarray):
        from owl.engine.aggregation import upsample_patch_field

        crowd_cells = (
            upsample_patch_field(crowd, cfg.world.patch_size)
            if hasattr(cfg, "world")
            else np.zeros_like(state.health)
        )
        crowded = alive & (crowd_cells > 0.5)
        if np.any(crowded):
            crowded_reproduction_probability = float(
                np.mean(state.possibility[crowded, int(Action.REPRODUCE)], dtype=np.float64)
            )

    metrics = {
        "tick": int(state.tick),
        "alive_count": alive_count,
        "alive_fraction": float(alive_count / max(total_cells, 1)),
        "dead_count": int(total_cells - alive_count),
        "event_count": int(len(state.event_queue)),
        "mean_activation": _safe_alive_mean(state.activation, alive),
        "mean_memory": _safe_alive_mean(state.memory, alive),
        "mean_integration": _safe_alive_mean(state.integration, alive),
        "mean_resource": _safe_alive_mean(state.resource, alive),
        "mean_health": _safe_alive_mean(state.health, alive),
        "mean_boundary": _safe_alive_mean(state.boundary, alive),
        "mean_threshold": _safe_alive_mean(state.threshold, alive),
        "mean_possibility_entropy": possibility_entropy,
        "global_integration": float(state.global_state.integration),
        "global_fragmentation": float(state.global_state.fragmentation),
        "global_diversity": float(state.global_state.diversity),
        "global_complexity": float(state.global_state.complexity),
        "global_crisis": float(getattr(state.global_state, "crisis", 0.0)),
        "global_carrying_pressure": float(getattr(state.global_state, "carrying_pressure", 0.0)),
        "global_starvation_pressure": float(
            getattr(state.global_state, "starvation_pressure", 0.0)
        ),
        "global_food_deficit": float(getattr(state.global_state, "food_deficit", 0.0)),
        "global_readout": int(state.global_state.readout),
        "global_intention": int(state.global_state.intention),
        "patch_mean_integration": float(np.mean(patch_integration, dtype=np.float64))
        if patch_integration.size
        else 0.0,
        "patch_var_integration": float(np.var(patch_integration, dtype=np.float64))
        if patch_integration.size
        else 0.0,
        "food_total": float(np.sum(state.food, dtype=np.float64)),
        "toxin_total": float(np.sum(state.toxin, dtype=np.float64)),
        "signal_total": float(np.sum(state.signal, dtype=np.float64)),
        "signal_food_total": _channel_sum(state, SignalChannel.FOOD, cfg),
        "signal_danger_total": _channel_sum(state, SignalChannel.DANGER, cfg),
        "signal_coordination_total": _channel_sum(state, SignalChannel.COORDINATION, cfg),
        "signal_integration_total": _channel_sum(state, SignalChannel.INTEGRATION, cfg),
        "carnivore_fraction": float(
            np.count_nonzero((state.predation > 0.6) & alive) / max(alive_count, 1)
        ),
        "grazer_fraction": float(
            np.count_nonzero((state.grazing > 0.6) & alive) / max(alive_count, 1)
        ),
        "identity_duplicate_count": identity_duplicate_count,
        "mean_decision_urgency": _safe_state_mean("last_decision_urgency"),
        "mean_homeostatic_error": _safe_state_mean("last_homeostatic_error"),
        "mean_noetic_B": _safe_state_mean("noetic_B"),
        "mean_noetic_M": _safe_state_mean("noetic_M"),
        "mean_noetic_P": _safe_state_mean("noetic_P"),
        "mean_noetic_C": _safe_state_mean("noetic_C"),
        "mean_noetic_K": _safe_state_mean("noetic_K"),
        "mean_noetic_Theta": _safe_state_mean("noetic_Theta"),
        "mean_noetic_N": _safe_state_mean("noetic_N"),
        "patch_mean_crisis": _safe_patch_mean("patch_crisis"),
        "patch_mean_carrying_pressure": _safe_patch_mean("patch_carrying_pressure"),
        "starving_food_feed_probability": starving_food_feed_probability,
        "low_boundary_repair_probability": low_boundary_repair_probability,
        "crowded_reproduction_probability": crowded_reproduction_probability,
        "rest_fraction": _action_fraction(state, Action.REST, alive),
        "feed_fraction": _action_fraction(state, Action.FEED, alive),
        "communicate_fraction": _action_fraction(state, Action.COMMUNICATE, alive),
        "integrate_fraction": _action_fraction(state, Action.INTEGRATE, alive),
        "repair_fraction": _action_fraction(state, Action.REPAIR, alive),
        "reproduce_fraction": _action_fraction(state, Action.REPRODUCE, alive),
        "ingest_fraction": _action_fraction(state, Action.INGEST, alive),
        "movement_fraction": movement_fraction,
        "diagonal_movement_fraction": diagonal_movement_fraction,
        "mean_starvation_debt": mean_starvation_debt,
        "mean_last_intake": mean_last_intake,
        "mean_movement_loop_score": mean_movement_loop_score,
        "mean_action_entropy": possibility_entropy,
    }

    # RAQIC diagnostics are scalar audit metadata, not ontological evidence.
    if getattr(getattr(cfg, "raqic", None), "enabled", False) and isinstance(
        state.raqic_probabilities, np.ndarray
    ):
        rq = np.clip(state.raqic_probabilities, 0.0, 1.0)
        if alive_count:
            rq_alive = rq[alive]
            rq_sums = np.sum(rq_alive, axis=-1, keepdims=True)
            rq_alive = np.divide(
                rq_alive, rq_sums, out=np.zeros_like(rq_alive), where=rq_sums > cfg.actions.epsilon
            )
            rq_pos = rq_alive > 0.0
            rq_entropy = -np.sum(
                np.where(rq_pos, rq_alive * np.log(rq_alive + cfg.actions.epsilon), 0.0), axis=-1
            ) / np.log(float(len(Action)))
            metrics["mean_raqic_entropy"] = float(
                np.mean(np.clip(rq_entropy, 0.0, 1.0), dtype=np.float64)
            )
            metrics["mean_raqic_confidence"] = _safe_state_mean("raqic_record_confidence")
            metrics["mean_raqic_trace_error"] = _safe_state_mean("raqic_trace_error")
            metrics["min_raqic_eigenvalue"] = (
                float(np.min(state.raqic_min_eigenvalue[alive]))
                if isinstance(state.raqic_min_eigenvalue, np.ndarray)
                else 0.0
            )
            metrics["mean_raqic_compare_l1"] = _safe_state_mean("raqic_compare_l1")
            metrics["mean_raqic_compare_kl"] = _safe_state_mean("raqic_compare_kl")
            innovation = getattr(state, "raqic_utility_innovation", None)
            metrics["mean_raqic_utility_innovation_l1"] = (
                float(np.mean(np.sum(np.abs(innovation[alive]), axis=-1), dtype=np.float64))
                if isinstance(innovation, np.ndarray)
                else 0.0
            )
            alignment = getattr(state, "raqic_phase_alignment", None)
            metrics["mean_raqic_phase_alignment"] = (
                float(np.mean(alignment[alive], dtype=np.float64))
                if isinstance(alignment, np.ndarray)
                else 0.0
            )
            metrics["mean_raqic_interference_delta_l1"] = _safe_state_mean(
                "raqic_interference_delta_l1"
            )
            metrics["mean_raqic_policy_kl"] = _safe_state_mean("raqic_policy_kl")
            metrics["mean_raqic_utility_projection_fraction"] = _safe_state_mean(
                "raqic_utility_projection_fraction"
            )
            metrics["mean_raqic_utility_score_cosine"] = _safe_state_mean(
                "raqic_utility_score_cosine"
            )
            metrics["max_raqic_utility_orthogonality_residual"] = (
                float(np.max(np.abs(state.raqic_utility_orthogonality_residual[alive])))
                if isinstance(state.raqic_utility_orthogonality_residual, np.ndarray)
                else 0.0
            )
            metrics["mean_raqic_utility_innovation_norm"] = _safe_state_mean(
                "raqic_utility_innovation_norm"
            )
            metrics["max_raqic_interference_norm_error"] = (
                float(np.max(np.abs(state.raqic_interference_norm_error[alive])))
                if isinstance(state.raqic_interference_norm_error, np.ndarray)
                else 0.0
            )
            metrics["max_raqic_interference_illegal_mass"] = (
                float(np.max(np.abs(state.raqic_interference_illegal_mass[alive])))
                if isinstance(state.raqic_interference_illegal_mass, np.ndarray)
                else 0.0
            )
            shadow_readout = getattr(state, "raqic_shadow_readout", None)
            metrics["raqic_shadow_readout_change_fraction"] = (
                float(np.mean(shadow_readout[alive] != state.raqic_readout[alive]))
                if isinstance(shadow_readout, np.ndarray)
                and isinstance(state.raqic_readout, np.ndarray)
                else 0.0
            )
            patch_phase_coherence = getattr(state, "raqic_patch_action_coherence", None)
            metrics["mean_raqic_patch_action_coherence"] = (
                float(np.mean(patch_phase_coherence, dtype=np.float64))
                if isinstance(patch_phase_coherence, np.ndarray)
                else 0.0
            )
            global_phase_coherence = getattr(state, "raqic_global_action_coherence", None)
            metrics["mean_raqic_global_action_coherence"] = (
                float(np.mean(global_phase_coherence, dtype=np.float64))
                if isinstance(global_phase_coherence, np.ndarray)
                else 0.0
            )
            flags = getattr(state, "raqic_audit_flags", None)
            if isinstance(flags, np.ndarray) and flags.ndim == 3:
                metrics["raqic_fallback_fraction"] = float(
                    np.mean(flags[alive, 0] > 0, dtype=np.float64)
                )
            else:
                metrics["raqic_fallback_fraction"] = 0.0
        else:
            metrics.update(
                {
                    "mean_raqic_entropy": 0.0,
                    "mean_raqic_confidence": 0.0,
                    "mean_raqic_trace_error": 0.0,
                    "min_raqic_eigenvalue": 0.0,
                    "mean_raqic_compare_l1": 0.0,
                    "mean_raqic_compare_kl": 0.0,
                    "mean_raqic_utility_innovation_l1": 0.0,
                    "mean_raqic_phase_alignment": 0.0,
                    "mean_raqic_interference_delta_l1": 0.0,
                    "mean_raqic_policy_kl": 0.0,
                    "mean_raqic_utility_projection_fraction": 0.0,
                    "mean_raqic_utility_score_cosine": 0.0,
                    "max_raqic_utility_orthogonality_residual": 0.0,
                    "mean_raqic_utility_innovation_norm": 0.0,
                    "max_raqic_interference_norm_error": 0.0,
                    "max_raqic_interference_illegal_mass": 0.0,
                    "raqic_shadow_readout_change_fraction": 0.0,
                    "mean_raqic_patch_action_coherence": 0.0,
                    "mean_raqic_global_action_coherence": 0.0,
                    "raqic_fallback_fraction": 0.0,
                }
            )
        pi = getattr(state, "raqic_parent_intention", None)
        if isinstance(pi, np.ndarray) and alive_count:
            P = np.clip(pi[alive], 0.0, 1.0)
            sums = np.sum(P, axis=-1, keepdims=True)
            P = np.divide(P, sums, out=np.zeros_like(P), where=sums > cfg.actions.epsilon)
            positive = P > 0
            H = -np.sum(
                np.where(positive, P * np.log(P + cfg.actions.epsilon), 0.0), axis=-1
            ) / np.log(float(len(Action)))
            metrics["mean_raqic_parent_intention_entropy"] = float(
                np.mean(np.clip(H, 0.0, 1.0), dtype=np.float64)
            )
        else:
            metrics["mean_raqic_parent_intention_entropy"] = 0.0

    return metrics


def _json_ready(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert metric scalar values into JSON/CSV friendly primitives."""
    rows: list[dict[str, Any]] = []
    for row in metrics:
        clean: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, np.generic):
                clean[key] = value.item()
            elif isinstance(value, (np.ndarray, list, tuple, dict)):
                raise TypeError(
                    f"metric value for {key!r} must be scalar, got {type(value).__name__}"
                )
            else:
                clean[key] = value
        rows.append(clean)
    return rows


def save_metrics(metrics: list[dict[str, Any]], path: str) -> None:
    """Write scalar metrics to a tabular file.

    Parameters
    ----------
    metrics:
        List of scalar dictionaries returned by :func:`collect_metrics`.
    path:
        Destination path. ``.json``/``.jsonl`` and ``.csv`` are always
        supported. ``.parquet`` uses Polars when available, then pandas if a
        Parquet engine is installed; otherwise a clear JSON-lines fallback is
        written to the requested path so headless runs remain observable in
        minimal environments.
    """
    if not isinstance(metrics, list):
        raise TypeError(f"metrics must be a list[dict], got {type(metrics).__name__}")

    out_path = Path(path)
    if out_path.exists() and out_path.is_dir():
        raise ValueError(f"metrics path points to a directory: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = _json_ready(metrics)
    suffix = out_path.suffix.lower()

    if suffix == ".json":
        out_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return

    if suffix == ".jsonl" or suffix == ".ndjson":
        with out_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        return

    if suffix == ".csv":
        fieldnames = sorted({key for row in rows for key in row})
        with out_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return

    if suffix == ".parquet":
        try:
            import polars as pl

            pl.DataFrame(rows).write_parquet(str(out_path))
            return
        except Exception:
            pass

        try:
            import pandas as pd

            pd.DataFrame(rows).to_parquet(out_path)
            return
        except Exception:
            # Minimal-runtime fallback. The file extension is preserved because
            # the configured path may be *.parquet; the content is documented by
            # the first metadata row.
            with out_path.open("w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps({"format": "jsonl-fallback", "reason": "parquet engine unavailable"})
                    + "\n"
                )
                for row in rows:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
            return

    raise ValueError(
        f"unsupported metrics file extension {suffix!r}; use .json, .jsonl, .csv, or .parquet"
    )


def summarize_metrics(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute summary diagnostics from a run.

    Parameters
    ----------
    metrics:
        List of per-tick scalar diagnostics.

    Returns
    -------
    dict
        JSON-serializable summary including tick span, population extrema,
        final integration, and averages of core survival/integration metrics.
    """
    if not metrics:
        return {
            "num_records": 0,
            "first_tick": None,
            "last_tick": None,
            "initial_alive": 0,
            "final_alive": 0,
            "max_alive": 0,
            "min_alive": 0,
            "mean_alive": 0.0,
            "final_global_integration": 0.0,
            "max_mean_integration": 0.0,
            "mean_food_total": 0.0,
            "mean_signal_total": 0.0,
        }

    rows = _json_ready(metrics)

    def values(key: str) -> list[float]:
        return [float(row[key]) for row in rows if key in row and row[key] is not None]

    alive_values = [int(row.get("alive_count", 0)) for row in rows]
    integration_values = values("mean_integration")
    food_values = values("food_total")
    signal_values = values("signal_total")
    global_values = values("global_integration")

    return {
        "num_records": len(rows),
        "first_tick": int(rows[0].get("tick", 0)),
        "last_tick": int(rows[-1].get("tick", 0)),
        "initial_alive": int(alive_values[0]) if alive_values else 0,
        "final_alive": int(alive_values[-1]) if alive_values else 0,
        "max_alive": int(max(alive_values)) if alive_values else 0,
        "min_alive": int(min(alive_values)) if alive_values else 0,
        "mean_alive": float(fmean(alive_values)) if alive_values else 0.0,
        "final_global_integration": float(global_values[-1]) if global_values else 0.0,
        "max_mean_integration": float(max(integration_values)) if integration_values else 0.0,
        "mean_food_total": float(fmean(food_values)) if food_values else 0.0,
        "mean_signal_total": float(fmean(signal_values)) if signal_values else 0.0,
    }


# --- Advanced build overrides ------------------------------------------------
_mvp_collect_metrics = _base_collect_metrics


def collect_metrics(state: WorldState, cfg: SimulationConfig) -> dict[str, float | int]:
    """Collect baseline metrics plus advanced-build diagnostics when present."""
    out = dict(_mvp_collect_metrics(state, cfg))
    for name in (
        "digestion",
        "waste",
        "age_stress",
        "prediction_error",
        "development_stage",
        "symbiosis",
    ):
        arr = getattr(state, name, None)
        if isinstance(arr, np.ndarray):
            out[f"{name}_mean"] = float(np.mean(arr))
            out[f"{name}_sum"] = float(np.sum(arr, dtype=np.float64))
    if isinstance(state.genome, np.ndarray):
        living = (state.health > 0.0) & (~state.obstacle)
        if np.any(living):
            genome = state.genome[living]
            out["genome_diversity"] = float(np.mean(np.var(genome, axis=0)))
            out["genome_mean"] = float(np.mean(genome))
        else:
            out["genome_diversity"] = 0.0
            out["genome_mean"] = 0.0
    if isinstance(state.deception_memory, np.ndarray):
        out["deception_mean"] = float(np.mean(state.deception_memory))
    if isinstance(state.neighbor_trust, np.ndarray):
        out["neighbor_trust_mean"] = float(np.mean(state.neighbor_trust))
    if isinstance(state.last_death_mask, np.ndarray):
        out["death_count_last_tick"] = int(np.count_nonzero(state.last_death_mask))
    out["mobile_ow_count"] = int(len(state.mobile_ows))
    return out
