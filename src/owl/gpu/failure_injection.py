from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class FailureKind(StrEnum):
    NAN_HEALTH = "nan_health"
    BAD_PROBABILITY_ROW = "bad_probability_row"
    ILLEGAL_READOUT = "illegal_readout"
    DEAD_CELL_ACTION = "dead_cell_action"
    TOPOLOGY_OVERFLOW = "topology_overflow"
    FORCE_FALLBACK = "force_fallback"


@dataclass(frozen=True)
class FailureInjection:
    kind: FailureKind
    y: int = 0
    x: int = 0


def inject_failure(device_state: Any, injection: FailureInjection) -> dict[str, Any]:
    y, x = int(injection.y), int(injection.x)
    arrays = device_state.arrays
    xp = device_state.xp
    if injection.kind == FailureKind.NAN_HEALTH:
        arrays["health"][y, x] = xp.nan
    elif injection.kind == FailureKind.BAD_PROBABILITY_ROW:
        p = arrays.get("raqic_probabilities", arrays.get("possibility"))
        p[y, x, ...] = 0.0
    elif injection.kind == FailureKind.ILLEGAL_READOUT:
        arrays["readout"][y, x] = 2**15 - 1
    elif injection.kind == FailureKind.DEAD_CELL_ACTION:
        arrays["health"][y, x] = 0.0
        arrays["readout"][y, x] = 1
        if "raqic_readout" in arrays:
            arrays["raqic_readout"][y, x] = 1
    elif injection.kind == FailureKind.TOPOLOGY_OVERFLOW:
        device_state.metadata["topology_overflow"] = 1
    elif injection.kind == FailureKind.FORCE_FALLBACK:
        device_state.metadata["forced_fallback"] = True
    else:
        raise ValueError(injection.kind)
    return {"injected": injection.kind.value, "y": y, "x": x}
