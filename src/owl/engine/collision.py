"""Collision, inhibition, attack, and ingestion.

This module resolves sparse collision events produced by movement and applies
direct cell-cell interaction effects. It mutates only dense simulation state and
the sparse event queue; it does not choose actions.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from owl.core.actions import Action, EventKind, SignalChannel
from owl.core.config import SimulationConfig
from owl.core.state import EventRecord, WorldState, field_shape
from owl.engine.death import clear_cell
from owl.engine.events import dequeue_events, enqueue_event
from owl.engine.health import clip_life_fields
from owl.kernels.numba_kernels import ingestion_attempt_kernel
from owl.kernels.numpy_kernels import neighbor_mean_wrap, sigmoid
from owl_raqic.random_contract import RNGStream, uniform01


class RandomDrawSource(Protocol):
    def random(self) -> float: ...


def _validate_position(state: WorldState, position: tuple[int, int], label: str) -> tuple[int, int]:
    """Validate a cell coordinate."""
    y, x = map(int, position)
    height, width = field_shape(state)
    if not (0 <= y < height and 0 <= x < width):
        raise ValueError(f"{label} position {(y, x)} is outside field shape {(height, width)}")
    return y, x


def _alive_at(state: WorldState, position: tuple[int, int]) -> bool:
    """Return whether a position holds a living non-obstacle cell."""
    y, x = position
    return bool(
        (not state.obstacle[y, x]) and state.health[y, x] > 0.0 and state.boundary[y, x] > 0.0
    )


def compute_ingestion_probability(
    state: WorldState,
    predator: tuple[int, int],
    target: tuple[int, int],
    cfg: SimulationConfig,
) -> float:
    """Compute bounded ingestion success probability for one predator-target pair.

    The score uses the planned baseline formula: predator advantage from predation,
    integration, resource, and aggression competes against target health,
    boundary, and integration. ``cfg.predation.resistance_weight`` scales target
    resistance; disabled predation or sub-threshold predation returns zero.
    """
    py, px = _validate_position(state, predator, "predator")
    ty, tx = _validate_position(state, target, "target")

    if not cfg.predation.enabled:
        return 0.0
    if not _alive_at(state, (py, px)) or not _alive_at(state, (ty, tx)):
        return 0.0
    if float(state.predation[py, px]) < cfg.predation.min_predation_trait:
        return 0.0
    if float(state.resource[py, px]) <= cfg.resources.movement_cost:
        return 0.0

    probability = ingestion_attempt_kernel(
        state.predation,
        state.integration,
        state.resource,
        state.aggression,
        state.health,
        state.boundary,
        predator_y=np.array([py], dtype=np.int64),
        predator_x=np.array([px], dtype=np.int64),
        target_y=np.array([ty], dtype=np.int64),
        target_x=np.array([tx], dtype=np.int64),
        offset=0.3,
    )[0]

    # Apply additional configured resistance as a conservative adjustment. The
    # kernel implements the base formula; this lets config tune difficulty
    # without duplicating the whole compiled kernel.
    if cfg.predation.resistance_weight != 1.0:
        predator_score = (
            1.5 * float(state.predation[py, px])
            + 0.8 * float(state.integration[py, px])
            + 0.5 * float(state.resource[py, px])
            + 0.3 * float(state.aggression[py, px])
        )
        target_resistance = cfg.predation.resistance_weight * (
            0.8 * float(state.health[ty, tx])
            + 0.8 * float(state.boundary[ty, tx])
            + 0.4 * float(state.integration[ty, tx])
        )
        probability = float(sigmoid(predator_score - target_resistance - 0.3))

    return float(np.clip(probability, 0.0, 1.0))


def attempt_ingestion(
    state: WorldState,
    predator: tuple[int, int],
    target: tuple[int, int],
    cfg: SimulationConfig,
    rng: RandomDrawSource,
) -> bool:
    """Attempt predatory ingestion and mutate resources/health accordingly.

    On success, a fraction of target resource is transferred to the predator, a
    small memory trace is inherited, a residue/distress trace is deposited at the
    target, and the target cell is cleared. On failure, the predator and target
    take small bounded damage.
    """
    py, px = _validate_position(state, predator, "predator")
    ty, tx = _validate_position(state, target, "target")

    probability = compute_ingestion_probability(state, (py, px), (ty, tx), cfg)
    if probability <= 0.0:
        return False

    success = bool(rng.random() < probability)
    if success:
        target_resource = float(np.clip(state.resource[ty, tx], 0.0, cfg.resources.max_resource))
        transferred = cfg.predation.resource_transfer * target_resource
        state.resource[py, px] = min(
            cfg.resources.max_resource,
            float(state.resource[py, px]) + transferred,
        )
        state.memory[py, px] = float(
            np.clip(
                float(state.memory[py, px])
                + cfg.predation.memory_transfer * float(state.memory[ty, tx]),
                0.0,
                1.0,
            )
        )

        # Return a small non-transferred residue to the environment.
        state.food[ty, tx] = min(1.0, float(state.food[ty, tx]) + 0.20 * target_resource)
        distress_idx = int(SignalChannel.DISTRESS)
        if cfg.communication.enabled and distress_idx < state.signal_emission.shape[-1]:
            state.signal_emission[ty, tx, distress_idx] = min(
                1.0,
                float(state.signal_emission[ty, tx, distress_idx]) + 0.10,
            )

        clear_cell(state, (ty, tx))
        enqueue_event(
            state,
            EventRecord(
                kind=str(EventKind.INGESTION),
                tick=int(state.tick),
                source=(py, px),
                target=(ty, tx),
                payload={
                    "success": True,
                    "probability": probability,
                    "resource_transfer": transferred,
                },
            ),
        )
        clip_life_fields(state, cfg)
        return True

    # Failed attack: bounded costs/damage.
    state.resource[py, px] -= 0.5 * cfg.resources.movement_cost
    state.health[py, px] -= 0.03
    state.boundary[py, px] -= 0.02
    state.health[ty, tx] -= 0.01

    enqueue_event(
        state,
        EventRecord(
            kind=str(EventKind.INGESTION),
            tick=int(state.tick),
            source=(py, px),
            target=(ty, tx),
            payload={"success": False, "probability": probability},
        ),
    )
    clip_life_fields(state, cfg)
    return False


def _apply_collision_damage(
    state: WorldState, source: tuple[int, int], target: tuple[int, int], cfg: SimulationConfig
) -> None:
    """Apply non-ingestion collision damage to source and target."""
    sy, sx = source
    ty, tx = target
    damage = 0.02 * (float(state.aggression[sy, sx]) + float(state.aggression[ty, tx]))
    if damage <= 0.0:
        damage = 0.005

    state.health[sy, sx] -= damage
    state.boundary[sy, sx] -= 0.5 * damage
    state.health[ty, tx] -= damage
    state.boundary[ty, tx] -= 0.5 * damage
    clip_life_fields(state, cfg)


def _legacy_resolve_collisions_impl(
    state: WorldState, cfg: SimulationConfig, rng: np.random.Generator
) -> None:
    """Resolve queued collision events.

    Removes ``COLLISION`` events from the queue while preserving unrelated
    events. Ingestion readouts trigger predatory ingestion attempts; ordinary
    collisions apply small aggression-scaled damage.
    """
    collisions = dequeue_events(state, str(EventKind.COLLISION))
    if not collisions:
        return

    for event in collisions:
        if event.source is None or event.target is None:
            continue
        source = _validate_position(state, event.source, "collision source")
        target = _validate_position(state, event.target, "collision target")

        if not _alive_at(state, source) or not _alive_at(state, target):
            continue

        source_action = Action(int(state.readout[source]))
        target_action = Action(int(state.readout[target]))

        if source_action == Action.INGEST:
            attempt_ingestion(state, source, target, cfg, rng)
        elif target_action == Action.INGEST:
            attempt_ingestion(state, target, source, cfg, rng)
        else:
            _apply_collision_damage(state, source, target, cfg)


def apply_inhibition(state: WorldState, cfg: SimulationConfig) -> None:
    """Apply neighbor inhibition from cells whose readout is ``INHIBIT``.

    Mutates neighboring activation, integration, and a small amount of boundary
    pressure. Inhibiting cells pay a modest resource cost. A THREAT signal pulse
    is emitted when communication is enabled.
    """
    shape = field_shape(state)
    if state.readout.shape != shape:
        raise ValueError(f"state.readout must have shape {shape}, got {state.readout.shape}")

    alive = (state.health > 0.0) & (~state.obstacle)
    inhibiting = (state.readout == int(Action.INHIBIT)) & alive
    if not np.any(inhibiting):
        return

    inhibitor_strength = (
        inhibiting.astype(np.float32)
        * np.clip(state.aggression + state.integration + state.cooperation, 0.0, 3.0)
        / 3.0
    )
    pressure = neighbor_mean_wrap(inhibitor_strength)

    state.activation -= (0.08 * pressure).astype(state.activation.dtype, copy=False)
    state.integration -= (0.04 * pressure).astype(state.integration.dtype, copy=False)
    state.boundary -= (0.01 * pressure).astype(state.boundary.dtype, copy=False)

    state.resource[inhibiting] -= 0.5 * cfg.resources.movement_cost

    threat_idx = int(SignalChannel.THREAT)
    if cfg.communication.enabled and threat_idx < state.signal_emission.shape[-1]:
        state.signal_emission[..., threat_idx] += 0.10 * inhibitor_strength
        np.clip(
            state.signal_emission[..., threat_idx],
            0.0,
            1.0,
            out=state.signal_emission[..., threat_idx],
        )

    clip_life_fields(state, cfg)


# --- Deterministic simultaneous collision handling -------------------------
_legacy_resolve_collisions = _legacy_resolve_collisions_impl


class _FixedScientificDraw:
    """Tiny RNG adapter used to keep ``attempt_ingestion``'s public API stable."""

    def __init__(self, value: float):
        self.value = float(value)

    def random(self) -> float:
        return self.value


def _ingestion_counter_draw(
    state: WorldState, source: tuple[int, int], target: tuple[int, int], cfg: SimulationConfig
) -> float:
    sy, sx = source
    ty, tx = target
    predator_id = (
        int(state.occupancy[sy, sx])
        if int(state.occupancy[sy, sx]) >= 0
        else sy * state.health.shape[1] + sx
    )
    target_id = (
        int(state.occupancy[ty, tx])
        if int(state.occupancy[ty, tx]) >= 0
        else ty * state.health.shape[1] + tx
    )
    return float(
        uniform01(
            int(cfg.world.seed),
            int(state.tick),
            predator_id,
            RNGStream.INGESTION_OUTCOME,
            target_id,
            xp=np,
        )
    )


def resolve_collisions(state: WorldState, cfg: SimulationConfig, rng: np.random.Generator) -> None:
    """Resolve collisions under the deterministic scientific contract.

    ``rng`` is retained for API compatibility, but scientific ingestion outcomes
    are keyed by seed/tick/predator identity/target identity so they are invariant
    to chunking, graph replay, and distributed event ordering.
    """
    del rng
    collisions = dequeue_events(state, str(EventKind.COLLISION))
    if not collisions:
        return
    ingestion = []
    ordinary = []
    for event in collisions:
        if event.source is None or event.target is None:
            continue
        source = _validate_position(state, event.source, "collision source")
        target = _validate_position(state, event.target, "collision target")
        if not _alive_at(state, source) or not _alive_at(state, target):
            continue
        sa = Action(int(state.readout[source]))
        ta = Action(int(state.readout[target]))
        if sa == Action.INGEST:
            ingestion.append((source, target))
        elif ta == Action.INGEST:
            ingestion.append((target, source))
        else:
            ordinary.append((source, target))
    if ordinary:
        hd = np.zeros_like(state.health, dtype=np.float64)
        bd = np.zeros_like(state.boundary, dtype=np.float64)
        for source, target in ordinary:
            sy, sx = source
            ty, tx = target
            d = 0.02 * (float(state.aggression[sy, sx]) + float(state.aggression[ty, tx]))
            d = d if d > 0 else 0.005
            hd[sy, sx] += d
            hd[ty, tx] += d
            bd[sy, sx] += 0.5 * d
            bd[ty, tx] += 0.5 * d
        state.health[:] = np.clip(state.health - hd, 0.0, 1.0)
        state.boundary[:] = np.clip(state.boundary - bd, 0.0, 1.0)
    # Preserve sequential target-owner mutation while using a
    # deterministic identity-keyed draw. Stable coordinate ordering is part of
    # the scientific contract.
    for predator, target in sorted(ingestion, key=lambda pair: (pair[0], pair[1])):
        if not _alive_at(state, predator) or not _alive_at(state, target):
            continue
        draw = _ingestion_counter_draw(state, predator, target, cfg)
        attempt_ingestion(state, predator, target, cfg, _FixedScientificDraw(draw))
