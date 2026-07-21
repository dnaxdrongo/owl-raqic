from __future__ import annotations

import numpy as np

from owl.core.actions import Action, SignalChannel
from owl.core.config import SimulationConfig
from owl.core.state import WorldState, field_shape
from owl_raqic.types import RAQICFeatureBatch, RAQICFeaturePacket

FEATURE_NAMES = (
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


def quantize_unit_interval(x: float, bins: int = 256) -> int:
    return int(np.clip(np.floor(float(np.clip(x, 0.0, 1.0)) * (bins - 1)), 0, bins - 1))


def _channel_pressure(
    state: WorldState, channel: SignalChannel, cfg: SimulationConfig
) -> np.ndarray:
    idx = int(channel)
    if state.signal_reception.ndim != 3 or idx >= min(
        cfg.communication.num_channels, state.signal_reception.shape[-1]
    ):
        return np.zeros_like(state.health, dtype=np.float32)
    return np.asarray(state.signal_reception[..., idx], dtype=np.float32)


def _feature_values(
    state: WorldState, cfg: SimulationConfig, y: int, x: int, intention: np.ndarray
) -> dict[str, float]:
    resource = float(
        np.clip(state.resource[y, x] / max(cfg.resources.max_resource, cfg.actions.epsilon), 0, 1)
    )
    toxin = float(np.clip(state.toxin[y, x], 0, 1))
    food = float(np.clip(state.food[y, x], 0, 1))
    starvation = float(
        np.clip(getattr(state, "starvation_debt", np.zeros_like(state.health))[y, x], 0, 1)
    )
    danger = float(np.clip(_channel_pressure(state, SignalChannel.DANGER, cfg)[y, x], 0, 1))
    threat = float(np.clip(_channel_pressure(state, SignalChannel.THREAT, cfg)[y, x], 0, 1))
    signal = float(np.clip(np.mean(state.signal_reception[y, x, :]), 0, 1))
    coherence_src = getattr(state, "noetic_C", state.integration)
    coherence = float(np.clip(coherence_src[y, x], 0, 1))
    pred = float(
        np.clip(getattr(state, "prediction_error", np.zeros_like(state.health))[y, x], 0, 1)
    )
    phase = float(np.clip((state.phase[y, x] % (2 * np.pi)) / (2 * np.pi), 0, 1))
    intention = np.clip(np.asarray(intention, dtype=float), 0, 1)
    if intention.size > 1 and intention.sum() > 0:
        intention = intention / intention.sum()
        entropy = -np.sum(
            np.where(intention > 0, intention * np.log(intention + 1e-8), 0)
        ) / np.log(float(intention.size))
        parent = float(np.clip(1 - entropy, 0, 1))
    else:
        parent = 0.0
    risk = float(np.clip(0.45 * toxin + 0.25 * starvation + 0.15 * danger + 0.15 * threat, 0, 1))
    return {
        "resource": resource,
        "risk": risk,
        "memory": float(np.clip(state.memory[y, x], 0, 1)),
        "coherence": coherence,
        "phase": phase,
        "boundary": float(np.clip(state.boundary[y, x], 0, 1)),
        "signal": signal,
        "prediction_error": pred,
        "parent_context": parent,
        "food": food,
        "toxin": toxin,
    }


def build_feature_packets(
    state: WorldState, cfg: SimulationConfig, authority: np.ndarray, parent_intention: np.ndarray
) -> list[RAQICFeaturePacket]:
    h, w = field_shape(state)
    actions = len(Action)
    if authority.shape != (h, w, actions):
        raise ValueError(f"authority must have shape {(h, w, actions)}, got {authority.shape}")
    if parent_intention.shape != (h, w, actions):
        raise ValueError(
            f"parent_intention must have shape {(h, w, actions)}, got {parent_intention.shape}"
        )
    positions = np.argwhere((state.health > 0.0) & (~state.obstacle))
    if cfg.raqic.max_cells_per_tick is not None:
        positions = positions[: int(cfg.raqic.max_cells_per_tick)]
    packets = []
    for yi, xi in positions:
        y = int(yi)
        x = int(xi)
        features = _feature_values(state, cfg, y, x, parent_intention[y, x, :])
        codes = {k: quantize_unit_interval(v) for k, v in features.items()}
        codes.update(
            {
                "position_x": x,
                "position_y": y,
                "patch_id": int(
                    (y // cfg.world.patch_size) * (w // cfg.world.patch_size)
                    + (x // cfg.world.patch_size)
                ),
                "ow_id": int(state.occupancy[y, x])
                if int(state.occupancy[y, x]) >= 0
                else int(y * w + x),
            }
        )
        phase_code = max(codes["phase"], 1)
        for a in range(actions):
            codes[f"phase_num_{a}"] = int((phase_code + a + 1) % 251 + 1)
            codes[f"phase_den_{a}"] = int((a + 2) * 257)
        ow_id = int(state.occupancy[y, x]) if int(state.occupancy[y, x]) >= 0 else int(y * w + x)
        packets.append(
            RAQICFeaturePacket(
                ow_id=ow_id,
                scale_id=0,
                tick=int(state.tick),
                feature_bins=features,
                adelic_codes=codes,
                parent_intention=np.asarray(parent_intention[y, x, :], dtype=float),
                authority_mask=np.asarray(authority[y, x, :] > 0, dtype=bool),
                metadata={"y": y, "x": x, "owl_action_names": tuple(a.name for a in Action)},
            )
        )
    return packets


def build_feature_batch(
    state: WorldState, cfg: SimulationConfig, authority: np.ndarray, parent_intention: np.ndarray
) -> RAQICFeatureBatch:
    packets = build_feature_packets(state, cfg, authority, parent_intention)
    features = np.array(
        [[float(p.feature_bins[k]) for k in FEATURE_NAMES] for p in packets], dtype=np.float32
    )
    masks = (
        np.array([p.authority_mask for p in packets], dtype=bool)
        if packets
        else np.zeros((0, len(Action)), dtype=bool)
    )
    return RAQICFeatureBatch(
        features=features,
        action_mask=masks,
        scale_ids=np.array([p.scale_id for p in packets], dtype=np.int32),
        ow_ids=np.array([p.ow_id for p in packets], dtype=np.int64),
    )
