from __future__ import annotations

from typing import cast

import numpy as np

from owl.core.actions import Action, SignalChannel
from owl.core.config import SimulationConfig
from owl.core.state import WorldState, field_shape
from owl.raqic.feature_extraction import FEATURE_NAMES
from owl_raqic.gpu.dense_types import RAQICDenseBatch


def _channel_pressure(
    state: WorldState, channel: SignalChannel, cfg: SimulationConfig
) -> np.ndarray:
    idx = int(channel)
    if state.signal_reception.ndim != 3 or idx >= min(
        cfg.communication.num_channels, state.signal_reception.shape[-1]
    ):
        return np.zeros_like(state.health, dtype=np.float32)
    return np.asarray(state.signal_reception[..., idx], dtype=np.float32)


def _entropy_concentration(parent_intention: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    intention = np.asarray(parent_intention, dtype=np.float64)
    intention = np.clip(intention, 0.0, None)
    sums = np.sum(intention, axis=-1, keepdims=True)
    n_actions = intention.shape[-1]
    norm = np.divide(intention, sums, out=np.zeros_like(intention), where=sums > eps)
    ent = -np.sum(np.where(norm > 0, norm * np.log(norm + eps), 0.0), axis=-1) / np.log(
        float(n_actions)
    )
    conc = np.clip(1.0 - ent, 0.0, 1.0)
    conc[sums[..., 0] <= eps] = 0.0
    return cast(np.ndarray, conc.astype(np.float32))


def build_dense_feature_batch_numpy(
    state: WorldState,
    cfg: SimulationConfig,
    authority: np.ndarray,
    parent_intention: np.ndarray,
    utilities: np.ndarray | None = None,
    parent_action_phase: np.ndarray | None = None,
    parent_action_coherence: np.ndarray | None = None,
    *,
    all_cells_required: bool | None = None,
) -> RAQICDenseBatch:
    """Build the dense all-cell RAQIC batch from OWL arrays.

    This is the dense counterpart of build_feature_packets. It does not mutate state.
    """
    h, w = field_shape(state)
    actions = len(Action)
    if authority.shape != (h, w, actions):
        raise ValueError(f"authority must have shape {(h, w, actions)}, got {authority.shape}")
    if parent_intention.shape != (h, w, actions):
        raise ValueError(
            f"parent_intention must have shape {(h, w, actions)}, got {parent_intention.shape}"
        )

    if utilities is not None and utilities.shape != (h, w, actions):
        raise ValueError(f"utilities must have shape {(h, w, actions)}, got {utilities.shape}")
    if parent_action_phase is not None and parent_action_phase.shape != (h, w, actions):
        raise ValueError(
            "parent_action_phase must have shape "
            f"{(h, w, actions)}, got {parent_action_phase.shape}"
        )
    if parent_action_coherence is not None and parent_action_coherence.shape != (h, w, actions):
        raise ValueError(
            "parent_action_coherence must have shape "
            f"{(h, w, actions)}, got {parent_action_coherence.shape}"
        )

    eligible = (state.health > 0.0) & (~state.obstacle)
    positions = np.argwhere(eligible)
    n_eligible = int(positions.shape[0])
    all_required = (
        cfg.raqic.gpu_all_cells_required if all_cells_required is None else bool(all_cells_required)
    )
    if cfg.raqic.max_cells_per_tick is not None and n_eligible > int(cfg.raqic.max_cells_per_tick):
        if all_required:
            raise RuntimeError(
                f"gpu all-cell mode refuses to cap cells: eligible={n_eligible}, "
                f"max_cells_per_tick={cfg.raqic.max_cells_per_tick}"
            )
        positions = positions[: int(cfg.raqic.max_cells_per_tick)]

    if positions.size == 0:
        return RAQICDenseBatch(
            ow_id=np.zeros((0,), dtype=np.int64),
            yx=np.zeros((0, 2), dtype=np.int32),
            features=np.zeros((0, len(FEATURE_NAMES)), dtype=np.float64),
            feature_bins=np.zeros((0, len(FEATURE_NAMES)), dtype=np.int32),
            adelic_codes=np.zeros((0, len(FEATURE_NAMES)), dtype=np.int32),
            authority_mask=np.zeros((0, actions), dtype=bool),
            parent_intention=np.zeros((0, actions), dtype=np.float64),
            alive_mask=np.zeros((0,), dtype=bool),
            scale_id=np.zeros((0,), dtype=np.int32),
            tick=int(state.tick),
            feature_names=tuple(FEATURE_NAMES),
            action_names=tuple(action.name for action in Action),
            active_primes=tuple(cfg.raqic.active_primes),
            action_utilities=(
                None if utilities is None else np.zeros((0, actions), dtype=np.float64)
            ),
            parent_action_phase=(
                None if parent_action_phase is None else np.zeros((0, actions), dtype=np.float64)
            ),
            parent_action_coherence=(
                None
                if parent_action_coherence is None
                else np.zeros((0, actions), dtype=np.float64)
            ),
            metadata={
                "eligible_cells": 0,
                "processed_cells": 0,
                "all_cells_required": all_required,
            },
        )

    y = positions[:, 0].astype(np.int32)
    x = positions[:, 1].astype(np.int32)

    resource = np.clip(
        state.resource / max(cfg.resources.max_resource, cfg.actions.epsilon), 0.0, 1.0
    )
    toxin = np.clip(state.toxin, 0.0, 1.0)
    food = np.clip(state.food, 0.0, 1.0)
    starvation = np.clip(getattr(state, "starvation_debt", np.zeros_like(state.health)), 0.0, 1.0)
    danger = np.clip(_channel_pressure(state, SignalChannel.DANGER, cfg), 0.0, 1.0)
    threat = np.clip(_channel_pressure(state, SignalChannel.THREAT, cfg), 0.0, 1.0)
    signal = np.clip(np.mean(state.signal_reception, axis=-1), 0.0, 1.0)
    coherence_src = getattr(state, "noetic_C", state.integration)
    coherence = np.clip(coherence_src, 0.0, 1.0)
    pred = np.clip(getattr(state, "prediction_error", np.zeros_like(state.health)), 0.0, 1.0)
    phase = np.clip((state.phase % (2.0 * np.pi)) / (2.0 * np.pi), 0.0, 1.0)
    parent_context = _entropy_concentration(parent_intention)
    risk = np.clip(0.45 * toxin + 0.25 * starvation + 0.15 * danger + 0.15 * threat, 0.0, 1.0)

    fields = {
        "resource": resource,
        "risk": risk,
        "memory": np.clip(state.memory, 0.0, 1.0),
        "coherence": coherence,
        "phase": phase,
        "boundary": np.clip(state.boundary, 0.0, 1.0),
        "signal": signal,
        "prediction_error": pred,
        "parent_context": parent_context,
        "food": food,
        "toxin": toxin,
    }
    features = np.stack(
        [np.asarray(fields[name], dtype=np.float64)[y, x] for name in FEATURE_NAMES], axis=1
    )
    feature_bins = np.floor(np.clip(features, 0.0, 1.0) * 255.0).astype(np.int32)
    feature_bins = np.clip(feature_bins, 0, 255).astype(np.int32)

    occupancy = np.asarray(state.occupancy)
    ow_id = np.where(
        occupancy[y, x] >= 0, occupancy[y, x], y.astype(np.int64) * int(w) + x.astype(np.int64)
    ).astype(np.int64)

    return RAQICDenseBatch(
        ow_id=ow_id,
        yx=positions.astype(np.int32),
        features=features,
        feature_bins=feature_bins,
        adelic_codes=feature_bins.copy(),
        authority_mask=np.asarray(authority[y, x, :] > 0, dtype=bool),
        parent_intention=np.asarray(parent_intention[y, x, :], dtype=np.float64),
        alive_mask=np.ones((positions.shape[0],), dtype=bool),
        scale_id=np.zeros((positions.shape[0],), dtype=np.int32),
        tick=int(state.tick),
        feature_names=tuple(FEATURE_NAMES),
        action_names=tuple(action.name for action in Action),
        active_primes=tuple(cfg.raqic.active_primes),
        action_utilities=(
            None if utilities is None else np.asarray(utilities[y, x, :], dtype=np.float64)
        ),
        parent_action_phase=(
            None
            if parent_action_phase is None
            else np.asarray(parent_action_phase[y, x, :], dtype=np.float64)
        ),
        parent_action_coherence=(
            None
            if parent_action_coherence is None
            else np.asarray(parent_action_coherence[y, x, :], dtype=np.float64)
        ),
        metadata={
            "eligible_cells": n_eligible,
            "processed_cells": int(positions.shape[0]),
            "all_cells_required": all_required,
            "feature_source": "owl_dense_numpy",
        },
    )


def build_dense_feature_batch_gpu(
    state: WorldState,
    cfg: SimulationConfig,
    authority: np.ndarray,
    parent_intention: np.ndarray,
    utilities: np.ndarray | None = None,
    parent_action_phase: np.ndarray | None = None,
    parent_action_coherence: np.ndarray | None = None,
) -> RAQICDenseBatch:
    """Build a dense batch and move it to CuPy.

    The first accepted GPU implementation computes OWL feature extraction on CPU
    arrays then stages the dense slab once to device. The RAQIC decision math is
    GPU-native; future work can mirror OWL fields persistently on device.
    """
    from owl_raqic.gpu.backend import require_cupy

    cp = require_cupy()
    batch = build_dense_feature_batch_numpy(
        state,
        cfg,
        authority,
        parent_intention,
        utilities=utilities,
        parent_action_phase=parent_action_phase,
        parent_action_coherence=parent_action_coherence,
    )
    return RAQICDenseBatch(
        ow_id=cp.asarray(batch.ow_id),
        yx=cp.asarray(batch.yx),
        features=cp.asarray(batch.features),
        feature_bins=cp.asarray(batch.feature_bins),
        adelic_codes=cp.asarray(batch.adelic_codes),
        authority_mask=cp.asarray(batch.authority_mask),
        parent_intention=cp.asarray(batch.parent_intention),
        alive_mask=cp.asarray(batch.alive_mask),
        scale_id=cp.asarray(batch.scale_id),
        tick=batch.tick,
        feature_names=batch.feature_names,
        action_names=batch.action_names,
        active_primes=batch.active_primes,
        action_utilities=(
            None if batch.action_utilities is None else cp.asarray(batch.action_utilities)
        ),
        parent_action_phase=(
            None if batch.parent_action_phase is None else cp.asarray(batch.parent_action_phase)
        ),
        parent_action_coherence=(
            None
            if batch.parent_action_coherence is None
            else cp.asarray(batch.parent_action_coherence)
        ),
        metadata={**batch.metadata, "feature_source": "owl_dense_staged_to_gpu"},
    )
