"""Repository-owned Qiskit boundary types."""

from __future__ import annotations

from dataclasses import dataclass

from .json import JSONValue


@dataclass(frozen=True)
class RegisterLayout:
    action_qubits: tuple[int, ...]
    record_qubits: tuple[int, ...] = ()
    position_qubits: tuple[int, ...] = ()
    classical_action_bits: tuple[int, ...] = ()
    classical_record_bits: tuple[int, ...] = ()


@dataclass(frozen=True)
class AerCapabilities:
    version: str
    devices: tuple[str, ...]
    methods: tuple[str, ...]
    supported_options: frozenset[str]


@dataclass(frozen=True)
class QiskitChunkLedger:
    family: str
    expected_ids: tuple[int, ...]
    returned_ids: tuple[int, ...]
    circuit_hashes: tuple[str, ...]
    metadata: tuple[dict[str, JSONValue], ...]
    gpu_verified: bool
