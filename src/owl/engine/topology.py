"""Advanced topology operations for merge, split, expulsion, and nested OWs.

The baseline dense cell engine does not yet implement full graph/topology dynamics.
This module therefore provides safe, documented hooks. Detection converts
actualized advanced readouts into sparse events; application consumes those
events and calls harmless extension hooks. This keeps later imports stable
without destabilizing the current array-first baseline.
"""

from __future__ import annotations

import numpy as np

from owl.core.actions import Action, BoundaryMode, EventKind
from owl.core.config import SimulationConfig
from owl.core.state import EventRecord, WorldState, field_shape
from owl.engine.events import dequeue_events, enqueue_event

_ADJACENT_DELTAS_4: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _validate_position(state: WorldState, position: tuple[int, int], label: str) -> tuple[int, int]:
    """Validate one cell-grid coordinate."""
    y, x = map(int, position)
    height, width = field_shape(state)
    if not (0 <= y < height and 0 <= x < width):
        raise ValueError(f"{label} position {(y, x)} is outside field shape {(height, width)}")
    return y, x


def _alive_at(state: WorldState, position: tuple[int, int]) -> bool:
    """Return whether ``position`` currently holds a living cell."""
    y, x = position
    return bool(
        (not state.obstacle[y, x]) and state.health[y, x] > 0.0 and state.boundary[y, x] > 0.0
    )


def _neighbor_positions(
    state: WorldState, position: tuple[int, int], cfg: SimulationConfig
) -> list[tuple[int, int]]:
    """Return cardinal neighbor coordinates under the configured boundary mode."""
    y, x = position
    height, width = field_shape(state)
    mode = BoundaryMode(cfg.world.boundary_mode)
    neighbors: list[tuple[int, int]] = []
    for dy, dx in _ADJACENT_DELTAS_4:
        ny = y + int(dy)
        nx = x + int(dx)
        if mode == BoundaryMode.TOROIDAL:
            neighbors.append((ny % height, nx % width))
        elif 0 <= ny < height and 0 <= nx < width:
            neighbors.append((ny, nx))
    return neighbors


def _best_merge_target(
    state: WorldState, source: tuple[int, int], cfg: SimulationConfig
) -> tuple[int, int] | None:
    """Return the highest-integration adjacent living target for baseline merge events."""
    candidates = [pos for pos in _neighbor_positions(state, source, cfg) if _alive_at(state, pos)]
    candidates = [pos for pos in candidates if pos != source]
    if not candidates:
        return None
    return max(candidates, key=lambda pos: float(state.integration[pos]))


def detect_topology_events(state: WorldState, cfg: SimulationConfig) -> None:
    """Detect advanced topology candidates and enqueue sparse events.

    Mutates only ``state.event_queue``. Full merge/split/expulsion graph
    mechanics are deferred; this detector exists so the baseline loop can route
    advanced readouts without failing imports.
    """
    shape = field_shape(state)
    if state.readout.shape != shape:
        raise ValueError(f"state.readout must have shape {shape}, got {state.readout.shape}")

    alive = (state.health > 0.0) & (~state.obstacle)

    merge_sources = np.column_stack(
        np.nonzero(alive & (state.readout == int(Action.MERGE)))
    ).astype(np.int64, copy=False)
    for y, x in merge_sources:
        source = (int(y), int(x))
        target = _best_merge_target(state, source, cfg)
        if target is None:
            continue
        enqueue_event(
            state,
            EventRecord(
                kind=str(EventKind.MERGE),
                tick=int(state.tick),
                source=source,
                target=target,
                payload={"mvp_hook": True},
            ),
        )

    split_sources = np.column_stack(
        np.nonzero(alive & (state.readout == int(Action.SPLIT)))
    ).astype(np.int64, copy=False)
    for y, x in split_sources:
        enqueue_event(
            state,
            EventRecord(
                kind=str(EventKind.SPLIT),
                tick=int(state.tick),
                source=(int(y), int(x)),
                payload={"mvp_hook": True},
            ),
        )

    expel_sources = np.column_stack(
        np.nonzero(alive & (state.readout == int(Action.EXPEL)))
    ).astype(np.int64, copy=False)
    for y, x in expel_sources:
        source = (int(y), int(x))
        parent_id = int(state.occupancy[source])
        enqueue_event(
            state,
            EventRecord(
                kind=str(EventKind.EXPULSION),
                tick=int(state.tick),
                source=source,
                payload={"parent_id": parent_id, "child_id": -1, "mvp_hook": True},
            ),
        )


def _base_apply_topology_events(state: WorldState, cfg: SimulationConfig) -> None:
    """Apply queued baseline topology events.

    Mutates ``state.event_queue`` by consuming ``MERGE``, ``SPLIT``, and
    ``EXPULSION`` events. The current baseline hooks do not modify dense cell arrays.
    Events of other kinds remain queued for their owning subsystems.
    """
    del cfg

    merge_events = dequeue_events(state, str(EventKind.MERGE))
    split_events = dequeue_events(state, str(EventKind.SPLIT))
    expulsion_events = dequeue_events(state, str(EventKind.EXPULSION))

    for event in merge_events:
        if event.source is not None and event.target is not None:
            merge_ows(state, event.source, event.target)

    for event in split_events:
        if event.source is not None:
            split_ow(state, event.source)

    for event in expulsion_events:
        parent_id = int(event.payload.get("parent_id", -1))
        child_id = int(event.payload.get("child_id", -1))
        expel_child_ow(state, parent_id, child_id)


def _base_merge_ows(state: WorldState, source: tuple[int, int], target: tuple[int, int]) -> None:
    """baseline no-op hook for merging two observer-window structures.

    Parameters are validated so bad topology events fail early, but no dense
    fields are changed. Later advanced topology can replace this with a
    resource/identity-preserving merge operation.
    """
    _validate_position(state, source, "merge source")
    _validate_position(state, target, "merge target")
    return None


def _base_split_ow(state: WorldState, source: tuple[int, int]) -> None:
    """baseline no-op hook for splitting an observer-window structure.

    The source coordinate is validated. Later advanced topology can allocate a
    child/mobile OW record or place split material into neighboring cells.
    """
    _validate_position(state, source, "split source")
    return None


def _base_expel_child_ow(state: WorldState, parent_id: int, child_id: int) -> None:
    """Detach a child sparse OW record from a parent sparse OW record when present.

    The dense cell baseline has no nested child placement. If matching ``OWRecord``
    entries exist in ``state.mobile_ows``, this function removes ``child_id``
    from the parent's child list and clears the child's ``parent_id``. Missing
    ids or ``child_id < 0`` are treated as harmless no-ops.
    """
    parent_id = int(parent_id)
    child_id = int(child_id)
    if child_id < 0:
        return None

    parent = state.mobile_ows.get(parent_id)
    child = state.mobile_ows.get(child_id)
    if parent is not None and child_id in parent.children:
        parent.children.remove(child_id)
    if child is not None and child.parent_id == parent_id:
        child.parent_id = None
    return None


# --- Advanced build overrides ------------------------------------------------
_mvp_merge_ows = _base_merge_ows
_mvp_split_ow = _base_split_ow
_mvp_expel_child_ow = _base_expel_child_ow
_mvp_apply_topology_events = _base_apply_topology_events


def _next_mobile_ow_id(state: WorldState) -> int:
    """Return a fresh sparse OW id."""
    return 1 + max(state.mobile_ows.keys(), default=0)


def merge_ows(state: WorldState, source: tuple[int, int], target: tuple[int, int]) -> None:
    """Merge adjacent dense OWs into a sparse mobile OW record in advanced mode."""
    sy, sx = _validate_position(state, source, "merge source")
    ty, tx = _validate_position(state, target, "merge target")
    if not (_alive_at(state, (sy, sx)) and _alive_at(state, (ty, tx))):
        return
    ow_id = _next_mobile_ow_id(state)
    from owl.core.state import OWRecord

    traits = np.array(
        [
            state.mobility[sy, sx],
            state.predation[sy, sx],
            state.grazing[sy, sx],
            state.cooperation[sy, sx],
            state.aggression[sy, sx],
            state.integration[sy, sx],
        ],
        dtype=np.float32,
    )
    genome = None
    if isinstance(state.genome, np.ndarray):
        genome = np.mean(
            np.stack([state.genome[sy, sx], state.genome[ty, tx]], axis=0), axis=0
        ).astype(np.float32)
    state.mobile_ows[ow_id] = OWRecord(
        id=ow_id,
        type_id=int(state.ow_type[sy, sx]),
        pos_y=int(round((sy + ty) / 2)),
        pos_x=int(round((sx + tx) / 2)),
        occupied_cells=[(sy, sx), (ty, tx)],
        parent_id=None,
        children=[],
        traits=traits,
        alive=True,
        genome=genome,
        resource=float(np.clip(state.resource[sy, sx] + state.resource[ty, tx], 0.0, 1.0)),
        health=float(np.clip(0.5 * (state.health[sy, sx] + state.health[ty, tx]), 0.0, 1.0)),
        boundary=float(np.clip(0.5 * (state.boundary[sy, sx] + state.boundary[ty, tx]), 0.0, 1.0)),
    )
    state.parent_id[sy, sx] = ow_id
    state.parent_id[ty, tx] = ow_id
    state.integration[sy, sx] = state.integration[ty, tx] = state.mobile_ows[ow_id].health


def split_ow(state: WorldState, source: tuple[int, int]) -> None:
    """Split a sparse OW touching source into two child records when possible."""
    src = _validate_position(state, source, "split source")
    for record in list(state.mobile_ows.values()):
        if src in record.occupied_cells and len(record.occupied_cells) >= 2 and record.alive:
            a = record.occupied_cells[::2]
            b = record.occupied_cells[1::2]
            if not b:
                return
            record.occupied_cells = a
            child_id = _next_mobile_ow_id(state)
            from owl.core.state import OWRecord

            cy, cx = b[0]
            state.mobile_ows[child_id] = OWRecord(
                id=child_id,
                type_id=record.type_id,
                pos_y=cy,
                pos_x=cx,
                occupied_cells=list(b),
                parent_id=record.id,
                children=[],
                traits=np.asarray(record.traits, dtype=np.float32).copy(),
                alive=True,
                genome=None
                if record.genome is None
                else np.asarray(record.genome, dtype=np.float32).copy(),
                resource=0.5 * float(record.resource),
                health=float(record.health),
                boundary=float(record.boundary),
            )
            record.children.append(child_id)
            record.resource *= 0.5
            for pos in b:
                state.parent_id[pos] = child_id
            return


def expel_child_ow(state: WorldState, parent_id: int, child_id: int) -> None:
    """Detach a child OW and place it near the parent when possible."""
    parent_id = int(parent_id)
    child_id = int(child_id)
    child = state.mobile_ows.get(child_id)
    parent = state.mobile_ows.get(parent_id)
    if child is None:
        return
    if parent is not None and child_id in parent.children:
        parent.children.remove(child_id)
    child.parent_id = None
    # Best-effort dense placement of first occupied cell if it is empty.
    if child.occupied_cells:
        y, x = child.occupied_cells[0]
        if (
            0 <= y < state.health.shape[0]
            and 0 <= x < state.health.shape[1]
            and state.health[y, x] <= 0
            and not state.obstacle[y, x]
        ):
            state.health[y, x] = np.float32(child.health)
            state.boundary[y, x] = np.float32(child.boundary)
            state.resource[y, x] = np.float32(min(child.resource, 1.0))
            state.occupancy[y, x] = int(y * state.health.shape[1] + x)


def apply_topology_events(state: WorldState, cfg: SimulationConfig) -> None:
    """Apply topology events; advanced mode mutates sparse mobile OW records."""
    if not (
        getattr(cfg.reproduction, "advanced_enabled", False)
        or getattr(cfg.reproduction, "symbiosis_enabled", False)
        or getattr(cfg.hierarchy, "dynamic_patches", False)
    ):
        _mvp_apply_topology_events(state, cfg)
        return
    merge_events = dequeue_events(state, str(EventKind.MERGE))
    split_events = dequeue_events(state, str(EventKind.SPLIT))
    expulsion_events = dequeue_events(state, str(EventKind.EXPULSION))
    for event in merge_events:
        if event.source is not None and event.target is not None:
            merge_ows(state, event.source, event.target)
    for event in split_events:
        if event.source is not None:
            split_ow(state, event.source)
    for event in expulsion_events:
        expel_child_ow(
            state, int(event.payload.get("parent_id", -1)), int(event.payload.get("child_id", -1))
        )
