from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

ACTION_GRAPH_VERSION = "semantic_families_v1"
CANONICAL_ACTION_NAMES: tuple[str, ...] = (
    "REST",
    "SENSE",
    "MOVE_N",
    "MOVE_S",
    "MOVE_E",
    "MOVE_W",
    "MOVE_NE",
    "MOVE_NW",
    "MOVE_SE",
    "MOVE_SW",
    "FEED",
    "COMMUNICATE",
    "INHIBIT",
    "INTEGRATE",
    "REPAIR",
    "REPRODUCE",
    "INGEST",
    "EXPEL",
    "SPLIT",
    "MERGE",
    "FLEE",
    "PURSUE",
)
ACTION_FAMILIES: tuple[tuple[str, ...], ...] = (
    ("MOVE_N", "MOVE_NE", "MOVE_E", "MOVE_SE", "MOVE_S", "MOVE_SW", "MOVE_W", "MOVE_NW"),
    ("REST", "SENSE", "INTEGRATE", "REPAIR"),
    ("FEED", "INGEST", "EXPEL"),
    ("COMMUNICATE", "INHIBIT"),
    ("REPRODUCE", "SPLIT", "MERGE"),
    ("FLEE", "PURSUE"),
)


def action_family_edges(action_names: tuple[str, ...]) -> tuple[tuple[int, int], ...]:
    """Return the canonical deterministic semantic action graph."""
    if tuple(action_names) != CANONICAL_ACTION_NAMES:
        raise ValueError("action names/order must match the canonical v0.9.6 action schema")
    index = {name: position for position, name in enumerate(action_names)}
    if len(index) != len(action_names):
        raise ValueError("action names must be unique")
    required = {name for family in ACTION_FAMILIES for name in family}
    missing = sorted(required - set(index))
    if missing:
        raise ValueError(f"action graph is missing required names: {missing}")

    edges: list[tuple[int, int]] = []
    movement = ACTION_FAMILIES[0]
    for offset, left in enumerate(movement):
        right = movement[(offset + 1) % len(movement)]
        a, b = sorted((index[left], index[right]))
        edges.append((a, b))
    for family in ACTION_FAMILIES[1:]:
        for offset, left in enumerate(family):
            for right in family[offset + 1 :]:
                a, b = sorted((index[left], index[right]))
                edges.append((a, b))
    return tuple(dict.fromkeys(edges))


def action_graph_hash(action_names: tuple[str, ...]) -> str:
    payload = {
        "version": ACTION_GRAPH_VERSION,
        "action_names": action_names,
        "edges": action_family_edges(action_names),
    }
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _pair_rotation_matrix(angle: float) -> np.ndarray:
    c = np.cos(float(angle))
    s = np.sin(float(angle))
    return np.asarray([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)


def legal_subspace_unitary(
    action_count: int,
    authority_row: Any,
    edges: tuple[tuple[int, int], ...],
    *,
    strength: float,
    trotter_steps: int,
) -> np.ndarray:
    """Build the exact audit/Qiskit unitary used by the dense pair mixer."""
    if action_count < 1:
        raise ValueError("action_count must be positive")
    if trotter_steps < 1:
        raise ValueError("trotter_steps must be at least one")
    legal = np.asarray(authority_row, dtype=bool).reshape(-1)
    if legal.size != int(action_count):
        raise ValueError("authority row length must equal action_count")
    unitary = np.eye(action_count, dtype=np.complex128)
    if float(strength) == 0.0:
        return unitary
    angle = float(strength) / (2.0 * float(trotter_steps))
    pair = _pair_rotation_matrix(angle)
    sequence = tuple(edges) + tuple(reversed(edges))
    for _ in range(int(trotter_steps)):
        for left, right in sequence:
            if not (0 <= left < action_count and 0 <= right < action_count):
                raise ValueError("action graph edge is out of range")
            if not (legal[left] and legal[right]):
                continue
            embedded = np.eye(action_count, dtype=np.complex128)
            embedded[np.ix_([left, right], [left, right])] = pair
            unitary = embedded @ unitary
    return unitary
