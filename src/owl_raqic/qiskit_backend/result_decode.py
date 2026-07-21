from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RegisterLayout:
    action_qubits: tuple[int, ...]
    record_qubits: tuple[int, ...] = ()
    position_qubits: tuple[int, ...] = ()
    classical_action_bits: tuple[int, ...] = ()
    classical_record_bits: tuple[int, ...] = ()
    classical_position_bits: tuple[int, ...] = ()


@dataclass(frozen=True)
class ActionBitLayout:
    """Global classical-bit indexes that encode the action integer.

    Qiskit count strings place the highest global classical bit on the left.
    ``action_bits[output_bit]`` gives the global classical bit used for that
    little-endian output bit, so layouts remain correct with multiple classical
    registers and spaces in count keys.
    """

    action_bits: tuple[int, ...]
    little_endian: bool = True
    classical_register_name: str = "c_action"

    @property
    def width(self) -> int:
        return len(self.action_bits)


def _bitstring_to_int(bitstring: str, layout: ActionBitLayout) -> int:
    clean = bitstring.replace(" ", "")
    if not layout.action_bits:
        raise ValueError("action bit layout is empty")
    required = max(layout.action_bits) + 1
    if len(clean) < required:
        clean = clean.zfill(required)
    value = 0
    for output_bit, classical_bit in enumerate(layout.action_bits):
        source = len(clean) - 1 - int(classical_bit)
        if source < 0:
            raise ValueError(
                f"count key {bitstring!r} is too short for classical bit {classical_bit}"
            )
        bit = 1 if clean[source] == "1" else 0
        target_bit = output_bit if layout.little_endian else layout.width - 1 - output_bit
        value |= bit << target_bit
    return value


def counts_to_action_probabilities(
    counts: Mapping[str, int],
    layout: ActionBitLayout,
    action_count: int,
    *,
    authority: np.ndarray | None = None,
    project_illegal_to_rest: bool = False,
    rest_index: int = 0,
) -> np.ndarray:
    out = np.zeros(int(action_count), dtype=np.float64)
    total = 0
    invalid = 0
    for bitstring, count in counts.items():
        count = int(count)
        if count < 0:
            raise ValueError("counts must be nonnegative")
        index = _bitstring_to_int(str(bitstring), layout)
        legal = 0 <= index < action_count
        if legal and authority is not None:
            legal = bool(np.asarray(authority, dtype=bool)[index])
        if legal:
            out[index] += count
        elif project_illegal_to_rest:
            if not (0 <= int(rest_index) < int(action_count)):
                raise ValueError("rest_index is outside the action range")
            out[int(rest_index)] += count
        else:
            invalid += count
        total += count
    if total <= 0:
        raise ValueError("counts contain no shots")
    if invalid:
        # Invalid padded computational states are retained as a hard diagnostic
        # rather than silently renormalized into a valid action distribution.
        raise ValueError(
            f"counts contain {invalid}/{total} shots outside action_count={action_count}"
        )
    out /= float(total)
    return out


def probabilities_dict_to_array(
    probabilities: Mapping[int | str, float],
    action_count: int,
) -> np.ndarray:
    out = np.zeros(int(action_count), dtype=np.float64)
    for key, value in probabilities.items():
        index = int(key.replace(" ", ""), 2) if isinstance(key, str) else int(key)
        if 0 <= index < action_count:
            out[index] += float(value)
    total = float(out.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("invalid probability dictionary")
    return out / total
