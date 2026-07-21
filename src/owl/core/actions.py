"""Discrete action, signal, intention, and event enumerations."""

from enum import IntEnum, StrEnum


class Action(IntEnum):
    """Actualized cell-level action/readout identifiers."""

    REST = 0
    SENSE = 1
    MOVE_N = 2
    MOVE_S = 3
    MOVE_E = 4
    MOVE_W = 5
    MOVE_NE = 6
    MOVE_NW = 7
    MOVE_SE = 8
    MOVE_SW = 9
    FEED = 10
    COMMUNICATE = 11
    INHIBIT = 12
    INTEGRATE = 13
    REPAIR = 14
    REPRODUCE = 15
    INGEST = 16
    EXPEL = 17
    SPLIT = 18
    MERGE = 19
    FLEE = 20
    PURSUE = 21


class SignalChannel(IntEnum):
    """Universal communication channels available to all observer windows."""

    FOOD = 0
    DANGER = 1
    THREAT = 2
    COORDINATION = 3
    DISTRESS = 4
    REPRODUCTION = 5
    TERRITORY = 6
    INTEGRATION = 7


class PatchIntention(IntEnum):
    """Regional observer-window policy summaries."""

    REST = 0
    SEEK_FOOD = 1
    AVOID_DANGER = 2
    COORDINATE = 3
    DEFEND = 4
    REPRODUCE = 5
    REPAIR = 6
    EXPLORE = 7


class GlobalIntention(IntEnum):
    """Apex/global observer-window policy summaries."""

    REST = 0
    EXPAND = 1
    CONSERVE = 2
    SEEK_FOOD = 3
    AVOID_THREAT = 4
    COORDINATE = 5
    REPRODUCE = 6
    DEFEND = 7
    EXPLORE = 8
    REPAIR = 9


class BoundaryMode(StrEnum):
    """Boundary behavior for spatial fields."""

    TOROIDAL = "toroidal"
    REFLECTIVE = "reflective"
    ABSORBING = "absorbing"
    OBSTACLE = "obstacle"


class EventKind(StrEnum):
    """Sparse event kinds handled outside dense array kernels."""

    COLLISION = "collision"
    DEATH = "death"
    REPRODUCTION = "reproduction"
    INGESTION = "ingestion"
    EXPULSION = "expulsion"
    RELEASE = "release"
    MERGE = "merge"
    SPLIT = "split"
    SIGNAL_OUTCOME = "signal_outcome"


MOVE_DELTAS: dict[Action, tuple[int, int]] = {
    Action.MOVE_N: (-1, 0),
    Action.MOVE_S: (1, 0),
    Action.MOVE_E: (0, 1),
    Action.MOVE_W: (0, -1),
    Action.MOVE_NE: (-1, 1),
    Action.MOVE_NW: (-1, -1),
    Action.MOVE_SE: (1, 1),
    Action.MOVE_SW: (1, -1),
}


DIAGONAL_MOVES: tuple[Action, ...] = (
    Action.MOVE_NE,
    Action.MOVE_NW,
    Action.MOVE_SE,
    Action.MOVE_SW,
)

CARDINAL_MOVES: tuple[Action, ...] = (
    Action.MOVE_N,
    Action.MOVE_S,
    Action.MOVE_E,
    Action.MOVE_W,
)

REVERSE_MOVE_ACTION: dict[Action, Action] = {
    Action.MOVE_N: Action.MOVE_S,
    Action.MOVE_S: Action.MOVE_N,
    Action.MOVE_E: Action.MOVE_W,
    Action.MOVE_W: Action.MOVE_E,
    Action.MOVE_NE: Action.MOVE_SW,
    Action.MOVE_SW: Action.MOVE_NE,
    Action.MOVE_NW: Action.MOVE_SE,
    Action.MOVE_SE: Action.MOVE_NW,
}
