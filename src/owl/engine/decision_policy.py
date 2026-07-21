from __future__ import annotations

import numpy as np

from owl.core.actions import Action
from owl.core.config import SimulationConfig
from owl.core.constants import DEFAULT_FLOAT_DTYPE, DEFAULT_READOUT_DTYPE
from owl.core.state import WorldState
from owl.engine.actualization import actualize_actions, compute_action_logits, sample_actions
from owl.kernels.numpy_kernels import normalize_last_axis, softmax_stable


def _legacy_shadow(
    state: WorldState,
    utilities: np.ndarray,
    authority: np.ndarray,
    parent_bias: np.ndarray,
    rng: np.random.Generator,
    cfg: SimulationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    logits = compute_action_logits(state, utilities, authority, parent_bias, cfg)
    probabilities = softmax_stable(logits, axis=-1, epsilon=cfg.actions.epsilon)
    dead = (state.health <= 0.0) | state.obstacle
    if np.any(dead):
        probabilities[dead, :] = 0
        probabilities[dead, int(Action.REST)] = 1
    probabilities = normalize_last_axis(probabilities, epsilon=cfg.actions.epsilon)
    readout = (
        sample_actions(probabilities, rng)
        if cfg.actions.stochastic
        else np.argmax(probabilities, axis=-1).astype(DEFAULT_READOUT_DTYPE)
    )
    readout[dead] = int(Action.REST)
    return probabilities.astype(DEFAULT_FLOAT_DTYPE), readout.astype(DEFAULT_READOUT_DTYPE)


def _record_hybrid_compare(
    state: WorldState, legacy_probs: np.ndarray, legacy_readout: np.ndarray
) -> None:
    if isinstance(state.raqic_legacy_shadow_possibility, np.ndarray):
        state.raqic_legacy_shadow_possibility[...] = legacy_probs
    if isinstance(state.raqic_legacy_shadow_readout, np.ndarray):
        state.raqic_legacy_shadow_readout[...] = legacy_readout
    if isinstance(state.raqic_compare_l1, np.ndarray) and isinstance(
        state.raqic_probabilities, np.ndarray
    ):
        state.raqic_compare_l1[...] = np.sum(
            np.abs(np.clip(state.raqic_probabilities, 0, 1) - legacy_probs), axis=-1
        ).astype(DEFAULT_FLOAT_DTYPE)
    if isinstance(state.raqic_compare_kl, np.ndarray) and isinstance(
        state.raqic_probabilities, np.ndarray
    ):
        p = np.clip(state.raqic_probabilities, 1e-8, 1.0)
        q = np.clip(legacy_probs, 1e-8, 1.0)
        state.raqic_compare_kl[...] = np.sum(p * np.log(p / q), axis=-1).astype(DEFAULT_FLOAT_DTYPE)


def apply_decision_policy(
    state: WorldState,
    cfg: SimulationConfig,
    rng: np.random.Generator,
    utilities: np.ndarray,
    authority: np.ndarray,
    parent_bias: np.ndarray,
    parent_phase: np.ndarray | None = None,
    synchrony: np.ndarray | None = None,
    coherence: np.ndarray | None = None,
    cross_scale: np.ndarray | None = None,
) -> None:
    rq = getattr(cfg, "raqic", None)
    if (
        rq is None
        or (not rq.enabled)
        or rq.decision_policy == "legacy"
        or float(rq.epsilon_raqic) == 0.0
    ):
        actualize_actions(state, utilities, authority, parent_bias, rng, cfg)
        return
    from owl.raqic.engine import apply_raqic_decisions
    from owl.raqic.state import ensure_raqic_fields

    ensure_raqic_fields(state, cfg)
    if rq.decision_policy == "hybrid_compare":
        legacy_probs, legacy_readout = _legacy_shadow(
            state, utilities, authority, parent_bias, rng, cfg
        )
        apply_raqic_decisions(
            state,
            cfg,
            authority,
            rng,
            utilities=np.asarray(
                state.pre_utilities if state.pre_utilities is not None else utilities
            ),
        )
        _record_hybrid_compare(state, legacy_probs, legacy_readout)
        actualize_actions(state, utilities, authority, parent_bias, rng, cfg)
        return
    if rq.decision_policy == "raqic":
        apply_raqic_decisions(
            state,
            cfg,
            authority,
            rng,
            utilities=np.asarray(
                state.pre_utilities if state.pre_utilities is not None else utilities
            ),
        )
        return
    raise ValueError(f"unknown RAQIC decision_policy {rq.decision_policy!r}")
