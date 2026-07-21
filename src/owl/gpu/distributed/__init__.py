"""Process-per-GPU distributed OWL execution."""

from .boundary_consistency import BoundaryConsistencyReport, verify_and_commit_boundaries
from .launch import run_distributed
from .partition import SpatialShard, partition_rows

__all__ = [
    "BoundaryConsistencyReport",
    "SpatialShard",
    "partition_rows",
    "run_distributed",
    "verify_and_commit_boundaries",
]
