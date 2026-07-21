"""Birth, reproduction, mutation, and lineage interfaces.

Reproduction is a physical/topological event: an already-actualized
``Action.REPRODUCE`` readout can place a child observer-window in an empty
neighbor cell. The child inherits physical state, possibility defaults, lineage,
and mutable traits, with bounded mutation supplied by ``owl.core.traits``.

This module stays array-first. It loops only over reproduction candidates, which
are sparse relative to the dense grid.
"""

from __future__ import annotations

from dataclasses import fields as _dataclass_fields
from typing import cast

import numpy as np

from owl.core.actions import Action, BoundaryMode, EventKind
from owl.core.config import SimulationConfig
from owl.core.constants import CELL_FIELDS_2D, CELL_FIELDS_3D
from owl.core.state import EventRecord, WorldState, field_shape
from owl.core.traits import copy_traits_with_mutation
from owl.engine.events import enqueue_event
from owl.engine.health import clip_life_fields
from owl.science.reproduction_contract import (
    apply_reproduction_arrays as _apply_reproduction_arrays,
)

_NEIGHBOR_DELTAS_4: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _validate_position(state: WorldState, position: tuple[int, int], label: str) -> tuple[int, int]:
    """Validate one cell-grid coordinate."""
    y, x = map(int, position)
    height, width = field_shape(state)
    if not (0 <= y < height and 0 <= x < width):
        raise ValueError(f"{label} position {(y, x)} is outside field shape {(height, width)}")
    return y, x


def _cell_is_empty(state: WorldState, position: tuple[int, int]) -> bool:
    """Return whether ``position`` can receive a newborn cell."""
    y, x = position
    return bool(
        (not state.obstacle[y, x]) and state.health[y, x] <= 0.0 and state.occupancy[y, x] < 0
    )


def _flat_cell_id(position: tuple[int, int], width: int) -> int:
    """Return a stable spatial identity id for a cell position."""
    y, x = position
    return int(y) * int(width) + int(x)


def _parent_patch_id_for_position(state: WorldState, position: tuple[int, int]) -> int:
    """Infer parent patch id from current patch tiling."""
    y, x = position
    height, width = field_shape(state)
    patch_h, patch_w = state.patches.integration.shape
    if patch_h <= 0 or patch_w <= 0:
        return -1
    patch_size_y = height // patch_h
    patch_size_x = width // patch_w
    if patch_size_y <= 0 or patch_size_x <= 0:
        return -1
    return int((y // patch_size_y) * patch_w + (x // patch_size_x))


def find_empty_neighbor_positions(
    state: WorldState,
    position: tuple[int, int],
    cfg: SimulationConfig,
) -> list[tuple[int, int]]:
    """Return empty cardinal-neighbor positions available for reproduction.

    Parameters
    ----------
    state:
        Runtime dense state. This function does not mutate state.
    position:
        Parent coordinate ``(row, column)``.
    cfg:
        Simulation configuration. ``cfg.world.boundary_mode`` controls whether
        off-grid neighbors wrap toroidally or are rejected.

    Returns
    -------
    list[tuple[int, int]]
        Empty, non-obstacle cell coordinates. A coordinate is empty when
        ``health <= 0`` and ``occupancy < 0``.
    """
    y, x = _validate_position(state, position, "parent")
    height, width = field_shape(state)
    mode = BoundaryMode(cfg.world.boundary_mode)

    positions: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for dy, dx in _NEIGHBOR_DELTAS_4:
        ny = y + int(dy)
        nx = x + int(dx)
        if mode == BoundaryMode.TOROIDAL:
            candidate = (ny % height, nx % width)
        else:
            if not (0 <= ny < height and 0 <= nx < width):
                continue
            candidate = (ny, nx)

        if candidate in seen:
            continue
        seen.add(candidate)
        if _cell_is_empty(state, candidate):
            positions.append(candidate)

    return positions


def _base_viable_reproduction_mask(state: WorldState, cfg: SimulationConfig) -> np.ndarray:
    """Return cells allowed to place child cells this tick."""
    shape = field_shape(state)
    if state.readout.shape != shape:
        raise ValueError(f"state.readout must have shape {shape}, got {state.readout.shape}")

    rcfg = cfg.reproduction
    alive = (state.health > 0.0) & (~state.obstacle)
    return cast(
        np.ndarray,
        (
            alive
            & (state.readout == int(Action.REPRODUCE))
            & (state.resource >= rcfg.min_resource)
            & (state.health >= rcfg.min_health)
            & (state.boundary >= rcfg.min_boundary)
            & (state.integration >= rcfg.min_integration)
            & (state.reproduction_rate > 0.0)
        ),
    )


def _legacy_apply_reproduction(
    state: WorldState, cfg: SimulationConfig, rng: np.random.Generator
) -> None:
    """Create child cells for viable ``REPRODUCE`` readouts.

    Mutates parent resource and child cell-owned fields through
    :func:`copy_child_from_parent`. Successful births append ``REPRODUCTION``
    events. Candidate order and child-site choice are driven by the explicit
    ``rng`` for deterministic replay.
    """
    if not cfg.reproduction.enabled:
        return

    candidates = np.column_stack(np.nonzero(_viable_reproduction_mask(state, cfg))).astype(
        np.int64, copy=False
    )
    if candidates.size == 0:
        return

    for index in rng.permutation(len(candidates)):
        y, x = map(int, candidates[index])
        # Recheck because earlier births can consume neighboring space or parent
        # resource can have changed if future extensions add multi-birth logic.
        if not _viable_reproduction_mask(state, cfg)[y, x]:
            continue

        # Reproduction rate is a trait gate. Utility/authority already use it,
        # but retaining a bounded stochastic gate lets mutation matter even when
        # a REPRODUCE readout appears.
        if rng.random() > float(np.clip(state.reproduction_rate[y, x], 0.0, 1.0)):
            continue

        empty = find_empty_neighbor_positions(state, (y, x), cfg)
        if not empty:
            continue

        child = empty[int(rng.integers(0, len(empty)))]
        copy_child_from_parent(state, (y, x), child, cfg, rng)
        enqueue_event(
            state,
            EventRecord(
                kind=str(EventKind.REPRODUCTION),
                tick=int(state.tick),
                source=(int(y), int(x)),
                target=(int(child[0]), int(child[1])),
                payload={
                    "parent_lineage": int(state.lineage_id[y, x]),
                    "child_lineage": int(state.lineage_id[child]),
                    "child_resource": float(state.resource[child]),
                },
            ),
        )

    clip_life_fields(state, cfg)


def _base_copy_child_from_parent(
    state: WorldState,
    parent: tuple[int, int],
    child: tuple[int, int],
    cfg: SimulationConfig,
    rng: np.random.Generator,
) -> None:
    """Initialize a newborn child cell from a parent cell.

    Mutates parent ``resource`` and all child-owned physical, possibility,
    communication-trait, identity, and lineage fields at ``child``. Environment
    fields are not copied. Child traits are copied with bounded mutation using
    :func:`owl.core.traits.copy_traits_with_mutation`.
    """
    py, px = _validate_position(state, parent, "parent")
    cy, cx = _validate_position(state, child, "child")
    if not ((state.health[py, px] > 0.0) and (not state.obstacle[py, px])):
        raise ValueError(f"parent cell {(py, px)} is not alive")
    if not _cell_is_empty(state, (cy, cx)):
        raise ValueError(f"child target {(cy, cx)} is not empty")

    # Defensive shape checks for fields later copied by name.
    shape = field_shape(state)
    for name in CELL_FIELDS_2D:
        arr = getattr(state, name)
        if arr.shape != shape:
            raise ValueError(f"state.{name} must have shape {shape}, got {arr.shape}")
    for name in CELL_FIELDS_3D:
        arr = getattr(state, name)
        if arr.shape[:2] != shape:
            raise ValueError(f"state.{name} must begin with shape {shape}, got {arr.shape}")

    parent_resource = float(np.clip(state.resource[py, px], 0.0, cfg.resources.max_resource))
    child_resource = cfg.reproduction.child_resource_fraction * parent_resource
    state.resource[py, px] = np.float32(
        np.clip(parent_resource - child_resource, 0.0, cfg.resources.max_resource)
    )

    # Physical/cognitive initial state: partial inheritance plus configured
    # newborn defaults. These are bounded simulation heuristics.
    state.activation[cy, cx] = np.float32(0.50 * float(state.activation[py, px]))
    state.memory[cy, cx] = np.float32(
        np.clip(cfg.reproduction.memory_inheritance * float(state.memory[py, px]), 0.0, 1.0)
    )
    state.phase[cy, cx] = np.float32(
        (float(state.phase[py, px]) + rng.normal(0.0, cfg.phase.phase_noise_sigma)) % (2.0 * np.pi)
    )
    state.threshold[cy, cx] = np.float32(
        np.clip(
            float(state.threshold[py, px]) + rng.normal(0.0, 0.5 * cfg.reproduction.mutation_sigma),
            0.0,
            1.0,
        )
    )
    state.integration[cy, cx] = np.float32(
        np.clip(0.50 * float(state.integration[py, px]), 0.0, 1.0)
    )
    state.resource[cy, cx] = np.float32(np.clip(child_resource, 0.0, cfg.resources.max_resource))
    state.health[cy, cx] = np.float32(cfg.reproduction.initial_child_health)
    state.boundary[cy, cx] = np.float32(cfg.reproduction.initial_child_boundary)
    state.age[cy, cx] = 0
    state.ow_type[cy, cx] = state.ow_type[py, px]
    state.parent_id[cy, cx] = _parent_patch_id_for_position(state, (cy, cx))
    _, width = shape
    state.occupancy[cy, cx] = _flat_cell_id((cy, cx), width)

    state.readout[cy, cx] = int(Action.REST)
    state.possibility[cy, cx, :] = 0.0
    state.possibility[cy, cx, int(Action.REST)] = 1.0

    # No active emission/reception is inherited. Slow signal memory is inherited
    # as a weak trace of parent history.
    state.signal_emission[cy, cx, :] = 0.0
    state.signal_reception[cy, cx, :] = 0.0
    state.signal_memory[cy, cx, :] = np.clip(
        cfg.reproduction.memory_inheritance * state.signal_memory[py, px, :],
        0.0,
        1.0,
    ).astype(np.float32, copy=False)

    copy_traits_with_mutation(state, (py, px), (cy, cx), cfg, rng)
    update_lineage(state, (py, px), (cy, cx))

    # Keep source and target fields admissible.
    for name in (
        "activation",
        "memory",
        "threshold",
        "integration",
        "resource",
        "health",
        "boundary",
        "mobility",
        "metabolism",
        "predation",
        "grazing",
        "cooperation",
        "aggression",
        "curiosity",
        "reproduction_rate",
        "toxin_resistance",
        "memory_capacity",
        "coupling_strength",
        "emit_strength",
        "emit_efficiency",
        "receive_sensitivity",
        "signal_precision",
        "honesty_bias",
        "deception_bias",
    ):
        arr = getattr(state, name)
        np.clip(arr, 0.0, 1.0 if name != "resource" else cfg.resources.max_resource, out=arr)

    np.clip(state.channel_receptivity, 0.0, 1.0, out=state.channel_receptivity)
    np.clip(state.channel_emission_bias, 0.0, 1.0, out=state.channel_emission_bias)
    np.clip(state.channel_trust_local, 0.0, 1.0, out=state.channel_trust_local)
    np.clip(state.signal_memory, 0.0, 1.0, out=state.signal_memory)


def update_lineage(state: WorldState, parent: tuple[int, int], child: tuple[int, int]) -> None:
    """Assign stable lineage metadata to a child cell.

    Mutates ``lineage_id`` and ``age`` at ``child``. The child inherits the
    parent's lineage id when available; otherwise the parent's flat spatial id
    becomes the founder lineage. ``occupancy`` is intentionally not changed
    here, because occupancy identifies the child cell's current identity slot.
    """
    py, px = _validate_position(state, parent, "parent")
    cy, cx = _validate_position(state, child, "child")
    _, width = field_shape(state)
    parent_lineage = int(state.lineage_id[py, px])
    if parent_lineage < 0:
        parent_lineage = _flat_cell_id((py, px), width)

    state.lineage_id[cy, cx] = parent_lineage
    state.age[cy, cx] = 0


# --- Advanced build overrides ------------------------------------------------
_mvp_copy_child_from_parent = _base_copy_child_from_parent


def mutate_genome(
    genome: np.ndarray, cfg: SimulationConfig, rng: np.random.Generator
) -> np.ndarray:
    """Return a bounded mutated genome vector."""
    sigma = np.float32(
        getattr(cfg.reproduction, "genotype_mutation_sigma", cfg.reproduction.mutation_sigma)
    )
    child = np.asarray(genome, dtype=np.float32).copy()
    if sigma > 0:
        child += rng.normal(0.0, float(sigma), size=child.shape).astype(np.float32)
    return np.clip(child, 0.0, 1.0).astype(np.float32, copy=False)


def recombine_genomes(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Uniformly recombine two bounded genome vectors."""
    av = np.asarray(a, dtype=np.float32)
    bv = np.asarray(b, dtype=np.float32)
    if av.shape != bv.shape:
        raise ValueError(f"genome shapes must match, got {av.shape} and {bv.shape}")
    mask = rng.random(av.shape) < 0.5
    return np.where(mask, av, bv).astype(np.float32, copy=False)


def decode_genome_to_traits(state: WorldState, position: tuple[int, int]) -> None:
    """Decode genome channels onto core trait fields at one cell."""
    y, x = _validate_position(state, position, "genome decode")
    if not isinstance(state.genome, np.ndarray):
        return
    g = state.genome[y, x]
    fields = (
        "mobility",
        "metabolism",
        "predation",
        "grazing",
        "cooperation",
        "aggression",
        "curiosity",
        "reproduction_rate",
    )
    for idx, name in enumerate(fields[: g.shape[0]]):
        getattr(state, name)[y, x] = np.float32(np.clip(g[idx], 0.0, 1.0))


def _find_mate(
    state: WorldState, parent: tuple[int, int], cfg: SimulationConfig
) -> tuple[int, int] | None:
    """Return a neighboring mate for recombination, if available."""
    if not getattr(cfg.reproduction, "recombination_enabled", False):
        return None
    py, px = parent
    radius = int(getattr(cfg.reproduction, "mate_radius", 1))
    h, w = field_shape(state)
    best: tuple[int, int] | None = None
    best_score = -1.0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            y = (py + dy) % h
            x = (px + dx) % w
            if state.health[y, x] <= 0.0 or state.obstacle[y, x]:
                continue
            score = float(state.integration[y, x] + state.resource[y, x])
            if score > best_score:
                best = (y, x)
                best_score = score
    return best


def _advanced_copy_child_from_parent(
    state: WorldState,
    parent: tuple[int, int],
    child: tuple[int, int],
    cfg: SimulationConfig,
    rng: np.random.Generator,
) -> None:
    """Copy child from parent; advanced mode mutates/recombines genome."""
    if not getattr(cfg.reproduction, "advanced_enabled", False):
        _mvp_copy_child_from_parent(state, parent, child, cfg, rng)
        return

    from owl.core.advanced import ensure_advanced_fields

    ensure_advanced_fields(state, cfg)
    assert state.genome is not None
    assert state.symbiosis is not None
    assert state.development_stage is not None
    _mvp_copy_child_from_parent(state, parent, child, cfg, rng)
    py, px = _validate_position(state, parent, "parent")
    cy, cx = _validate_position(state, child, "child")
    mate = _find_mate(state, (py, px), cfg)
    parent_genome = state.genome[py, px]
    if mate is not None:
        base = recombine_genomes(parent_genome, state.genome[mate], rng)
        state.symbiosis[cy, cx] = np.float32(
            0.5 * (state.symbiosis[py, px] + state.symbiosis[mate])
        )
    else:
        base = parent_genome
        state.symbiosis[cy, cx] = state.symbiosis[py, px]
    state.genome[cy, cx] = mutate_genome(base, cfg, rng)
    state.development_stage[cy, cx] = 0.10
    decode_genome_to_traits(state, (cy, cx))
    np.clip(state.genome, 0.0, 1.0, out=state.genome)
    np.clip(state.development_stage, 0.0, 1.0, out=state.development_stage)


# --- Decision-homeostasis reproduction/identity overrides --------------------
def _viable_reproduction_mask(state: WorldState, cfg: SimulationConfig) -> np.ndarray:
    """Return reproduction mask with carrying-capacity homeostasis."""
    shape = field_shape(state)
    if state.readout.shape != shape:
        raise ValueError(f"state.readout must have shape {shape}, got {state.readout.shape}")
    rcfg = cfg.reproduction
    alive = (state.health > 0.0) & (~state.obstacle)
    base = (
        alive
        & (state.readout == int(Action.REPRODUCE))
        & (state.resource >= rcfg.min_resource)
        & (state.health >= rcfg.min_health)
        & (state.boundary >= rcfg.min_boundary)
        & (state.integration >= rcfg.min_integration)
        & (state.reproduction_rate > 0.0)
    )
    if getattr(cfg.cross_scale_homeostasis, "enabled", False):
        try:
            from owl.engine.utility import reproduction_viability_field

            viability = reproduction_viability_field(state, cfg)
            base &= viability > 0.20
        except Exception:
            pass
    return cast(np.ndarray, base)


_decision_homeostasis_copy_child_base = _advanced_copy_child_from_parent


def _homeostasis_copy_child_from_parent(
    state: WorldState,
    parent: tuple[int, int],
    child: tuple[int, int],
    cfg: SimulationConfig,
    rng: np.random.Generator,
) -> None:
    """Copy child and assign globally unique OW identity."""
    from owl.core.advanced import allocate_new_ow_id, ensure_advanced_fields

    ensure_advanced_fields(state, cfg)
    assert state.genome is not None
    assert state.symbiosis is not None
    assert state.development_stage is not None
    _decision_homeostasis_copy_child_base(state, parent, child, cfg, rng)
    py, px = _validate_position(state, parent, "parent")
    cy, cx = _validate_position(state, child, "child")

    if (
        getattr(cfg.identity, "enabled", False)
        or getattr(cfg.decision_homeostasis, "enabled", False)
        or getattr(cfg.cross_scale_homeostasis, "enabled", False)
    ):
        state.occupancy[cy, cx] = allocate_new_ow_id(state)
    parent_lineage = int(state.lineage_id[py, px])
    if parent_lineage < 0:
        parent_lineage = int(state.occupancy[py, px])
    state.lineage_id[cy, cx] = parent_lineage

    # Ensure advanced child diagnostics are initialized with no stale grid traces.
    if isinstance(state.starvation_debt, np.ndarray):
        state.starvation_debt[cy, cx] = 0.0
    if isinstance(state.movement_loop_score, np.ndarray):
        state.movement_loop_score[cy, cx] = 0.0
    if isinstance(state.last_chosen_macro, np.ndarray):
        state.last_chosen_macro[cy, cx] = int(Action.REST)
    if isinstance(state.last_macro_probabilities, np.ndarray):
        state.last_macro_probabilities[cy, cx, :] = 0.0
        state.last_macro_probabilities[cy, cx, int(Action.REST)] = 1.0

    # Stochastic gate for class diversification: niche viability adjusts but
    # never forces persistence of a class.
    if getattr(cfg.reproduction, "advanced_enabled", False) and isinstance(
        state.genome, np.ndarray
    ):
        try:
            from owl.engine.utility import compute_niche_payoff

            niche = compute_niche_payoff(state, cfg)
            state.genome[cy, cx] = np.clip(
                0.90 * state.genome[cy, cx] + 0.10 * niche[py, px], 0.0, 1.0
            )
            decode_genome_to_traits(state, (cy, cx))
        except Exception:
            pass


# --- Newborn decision-state initialization ---------------------------------
_v092_copy_child_base = _homeostasis_copy_child_from_parent


def copy_child_from_parent(
    state: WorldState,
    parent: tuple[int, int],
    child: tuple[int, int],
    cfg: SimulationConfig,
    rng: np.random.Generator,
) -> None:
    _v092_copy_child_base(state, parent, child, cfg, rng)
    cy, cx = child
    rest = int(Action.REST)
    for name in (
        "raqic_readout",
        "raqic_record_action",
        "raqic_record_readout",
        "raqic_legacy_shadow_readout",
    ):
        arr = getattr(state, name, None)
        if isinstance(arr, np.ndarray) and arr.shape == state.health.shape:
            arr[cy, cx] = rest
    for name in (
        "raqic_probabilities",
        "raqic_parent_intention",
        "raqic_legacy_shadow_possibility",
        "last_action_probabilities",
        "last_macro_probabilities",
    ):
        arr = getattr(state, name, None)
        if isinstance(arr, np.ndarray) and arr.shape[:2] == state.health.shape:
            arr[cy, cx, ...] = 0.0
            if arr.shape[-1] > rest:
                arr[cy, cx, rest] = 1.0
    for name in ("raqic_score", "raqic_phase"):
        arr = getattr(state, name, None)
        if isinstance(arr, np.ndarray) and arr.shape[:2] == state.health.shape:
            arr[cy, cx, ...] = 0.0


# --- Deterministic reproduction scheduling ---------------------------------


def apply_reproduction(state: WorldState, cfg: SimulationConfig, rng: np.random.Generator) -> None:
    """Apply the shared backend-neutral target-owner birth transition."""
    del rng
    arrays = {
        field.name: getattr(state, field.name)
        for field in _dataclass_fields(state)
        if isinstance(getattr(state, field.name), np.ndarray)
    }
    scalars = {"next_ow_id": int(getattr(state, "next_ow_id", 1))}
    diag = _apply_reproduction_arrays(
        arrays,
        scalars,
        cfg,
        tick=int(state.tick),
        xp=np,
        patch_shape=state.patches.integration.shape,
    )
    if hasattr(state, "next_ow_id"):
        state.next_ow_id = int(scalars["next_ow_id"])
    for parent, target, child_id in zip(diag.parents, diag.targets, diag.child_ids, strict=True):
        enqueue_event(
            state,
            EventRecord(
                kind=str(EventKind.REPRODUCTION),
                tick=int(state.tick),
                source=parent,
                target=target,
                payload={
                    "parent_lineage": int(state.lineage_id[parent]),
                    "child_lineage": int(state.lineage_id[target]),
                    "child_resource": float(state.resource[target]),
                    "child_ow_id": int(child_id),
                },
            ),
        )
    clip_life_fields(state, cfg)
