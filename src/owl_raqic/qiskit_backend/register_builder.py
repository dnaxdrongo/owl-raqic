from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RegisterMap:
    n_actions: int
    n_readouts: int = 1
    include_scale: bool = False
    include_place: bool = False
    action_qubits: int = 0
    readout_qubits: int = 0
    total_qubits: int = 0


def bits_needed(n: int) -> int:
    if n <= 1:
        return 1
    return math.ceil(math.log2(n))


def build_register_map(
    n_actions: int, n_readouts: int = 1, include_scale: bool = False, include_place: bool = False
) -> RegisterMap:
    aq = bits_needed(n_actions)
    rq = bits_needed(n_readouts)
    extra = (1 if include_scale else 0) + (1 if include_place else 0)
    return RegisterMap(
        n_actions=n_actions,
        n_readouts=n_readouts,
        include_scale=include_scale,
        include_place=include_place,
        action_qubits=aq,
        readout_qubits=rq,
        total_qubits=aq + rq + extra,
    )
