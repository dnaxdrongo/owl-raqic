"""Versioned scientific contracts and parity tooling for OWL.

This package is deliberately backend-neutral.  It describes the scientific
transition independently of the CPU, NumPy-array, CuPy, graph, and distributed
execution strategies.
"""

from .contract import ScientificContract, current_scientific_contract
from .counter_rng import RNGStream, uniform01, uniform_u64
from .stage_contract import STAGE_CONTRACTS, StageContract, scientific_stage_order

__all__ = [
    "ScientificContract",
    "current_scientific_contract",
    "RNGStream",
    "uniform01",
    "uniform_u64",
    "StageContract",
    "STAGE_CONTRACTS",
    "scientific_stage_order",
]
