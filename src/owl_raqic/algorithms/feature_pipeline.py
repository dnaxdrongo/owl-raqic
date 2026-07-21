from __future__ import annotations

from typing import cast

import numpy as np

from owl_raqic.adelic.projection import action_phase_vector, finite_feature_projection
from owl_raqic.config import RAQICAlgorithmConfig
from owl_raqic.types import RAQICFeaturePacket

FEATURE_ORDER = (
    "resource",
    "risk",
    "memory",
    "coherence",
    "phase",
    "boundary",
    "signal",
    "prediction_error",
    "parent_context",
    "food",
    "toxin",
)

DEFAULT_SCORE_WEIGHTS = np.array(
    [
        [0.20, -0.25, 0.10, 0.10, 0.00, 0.20, 0.00, -0.10, 0.00, 0.00, -0.10],
        [0.00, 0.30, 0.10, 0.10, 0.05, 0.00, 0.20, 0.60, 0.05, 0.00, 0.10],
        [0.05, 0.35, 0.05, 0.00, 0.10, 0.05, 0.20, 0.20, 0.10, 0.15, 0.25],
        [-0.30, -0.05, 0.05, 0.00, 0.00, 0.10, 0.05, 0.05, 0.00, 0.80, -0.05],
        [0.00, 0.05, 0.25, 0.35, 0.15, 0.10, 0.60, 0.05, 0.30, 0.00, 0.00],
        [0.10, 0.65, 0.10, 0.10, 0.00, 0.20, 0.30, 0.10, 0.10, 0.00, 0.55],
        [0.20, -0.05, 0.60, 0.55, 0.20, 0.35, 0.05, 0.10, 0.20, 0.00, 0.00],
        [-0.20, 0.25, 0.25, 0.10, 0.00, -0.55, 0.05, 0.15, 0.05, 0.00, 0.30],
        [0.75, -0.45, 0.35, 0.35, 0.05, 0.40, 0.05, -0.05, 0.10, 0.10, -0.40],
        [-0.20, 0.10, 0.00, 0.00, 0.00, 0.05, 0.10, 0.00, 0.00, 0.75, 0.05],
    ],
    dtype=float,
)

_DEFAULT_BY_NAME = {
    "REST": DEFAULT_SCORE_WEIGHTS[0],
    "SENSE": DEFAULT_SCORE_WEIGHTS[1],
    "MOVE": DEFAULT_SCORE_WEIGHTS[2],
    "FEED": DEFAULT_SCORE_WEIGHTS[3],
    "COMMUNICATE": DEFAULT_SCORE_WEIGHTS[4],
    "INHIBIT": DEFAULT_SCORE_WEIGHTS[5],
    "INTEGRATE": DEFAULT_SCORE_WEIGHTS[6],
    "REPAIR": DEFAULT_SCORE_WEIGHTS[7],
    "REPRODUCE": DEFAULT_SCORE_WEIGHTS[8],
    "INGEST": DEFAULT_SCORE_WEIGHTS[9],
    "EXPEL": np.array(
        [-0.10, 0.25, 0.00, 0.00, 0.00, 0.05, 0.15, 0.05, 0.00, -0.05, 0.40], dtype=float
    ),
    "SPLIT": np.array(
        [0.35, -0.10, 0.20, 0.20, 0.05, 0.25, 0.10, 0.00, 0.10, 0.05, -0.20], dtype=float
    ),
    "MERGE": np.array(
        [0.05, 0.15, 0.20, 0.35, 0.15, 0.20, 0.25, 0.05, 0.20, 0.00, 0.05], dtype=float
    ),
    "FLEE": np.array(
        [0.05, 0.85, 0.05, 0.00, 0.05, 0.10, 0.25, 0.10, 0.10, 0.00, 0.50], dtype=float
    ),
    "PURSUE": np.array(
        [0.20, 0.40, 0.05, 0.00, 0.05, 0.10, 0.15, 0.05, 0.10, 0.25, 0.05], dtype=float
    ),
}
for _move_name in (
    "MOVE_N",
    "MOVE_S",
    "MOVE_E",
    "MOVE_W",
    "MOVE_NE",
    "MOVE_NW",
    "MOVE_SE",
    "MOVE_SW",
):
    _DEFAULT_BY_NAME[_move_name] = DEFAULT_SCORE_WEIGHTS[2] + np.array(
        [0.0, 0.03 if "_" in _move_name else 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        dtype=float,
    )


def score_weights_for_actions(
    n_actions: int, action_names: tuple[str, ...] | None = None
) -> np.ndarray:
    """Return a deterministic action-feature score matrix for any finite action basis."""
    if n_actions <= 0:
        raise ValueError("n_actions must be positive")
    if action_names is None:
        if n_actions <= DEFAULT_SCORE_WEIGHTS.shape[0]:
            return DEFAULT_SCORE_WEIGHTS[:n_actions].copy()
        action_names = tuple(f"ACTION_{i}" for i in range(n_actions))
    out = np.zeros((n_actions, len(FEATURE_ORDER)), dtype=float)
    for i in range(n_actions):
        name = str(action_names[i]).upper() if i < len(action_names) else f"ACTION_{i}"
        if name in _DEFAULT_BY_NAME:
            out[i] = _DEFAULT_BY_NAME[name]
        elif name.startswith("MOVE"):
            out[i] = DEFAULT_SCORE_WEIGHTS[2]
        else:
            out[i] = np.array(
                [0.05, 0.00, 0.05, 0.05, 0.00, 0.05, 0.00, 0.00, 0.00, 0.00, 0.00], dtype=float
            )
    return out


def packet_to_feature_array(packet: RAQICFeaturePacket) -> np.ndarray:
    return np.array([float(packet.feature_bins.get(k, 0.0)) for k in FEATURE_ORDER], dtype=float)


def compute_scores(
    packet: RAQICFeaturePacket,
    config: RAQICAlgorithmConfig,
    action_names: tuple[str, ...] | None = None,
) -> np.ndarray:
    x = packet_to_feature_array(packet)
    places = config.active_places
    code_list = []
    for k in FEATURE_ORDER:
        val = packet.adelic_codes.get(k, 0)
        code_list.append({p: int(val) for p in places.primes})
    projected = finite_feature_projection(
        x, code_list, places.primes, places.prime_weights, epsilon_adelic=config.epsilon_adelic
    )
    weights = score_weights_for_actions(config.registers.n_actions, action_names)
    return cast(np.ndarray, weights @ projected)


def compute_action_phases(packet: RAQICFeaturePacket, config: RAQICAlgorithmConfig) -> np.ndarray:
    if getattr(config, "phase_mode", "scalar_reference") == "canonical_device":
        from owl_raqic.gpu.phase_kernels import build_phase_coefficients, compute_canonical_phases

        bins = np.array(
            [[int(packet.adelic_codes.get(k, 0)) for k in FEATURE_ORDER]], dtype=np.int32
        )
        table = build_phase_coefficients(
            FEATURE_ORDER,
            config.registers.n_actions,
            config.active_places.primes,
            xp=np,
            modulus_power=max(1, min(int(config.active_places.modulus_power), 4)),
        )
        return np.asarray(
            compute_canonical_phases(bins, table, xp=np, epsilon_adelic=config.epsilon_adelic)[0],
            dtype=float,
        )
    pairs = []
    for a in range(config.registers.n_actions):
        n = int(packet.adelic_codes.get(f"phase_num_{a}", a + 1))
        d = int(packet.adelic_codes.get(f"phase_den_{a}", a + 2))
        pairs.append((n, d))
    return action_phase_vector(pairs, config.active_places.primes, diagonal_test=False)
