from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RAQICMemoryEstimate:
    n_cells: int
    n_actions: int
    n_features: int
    dtype_bytes: int
    core_bytes: int
    scratch_bytes: int
    diagnostic_bytes: int
    total_bytes: int
    total_mb: float


def estimate_dense_raqic_memory(
    n_cells: int, n_actions: int, n_features: int, dtype_bytes: int = 8, diagnostics: bool = False
) -> RAQICMemoryEstimate:
    # features NF, masks NA bool, parent/scores/phases/probs NA, readout N,
    # cdf scratch NA, misc confidence N. Conservative but not absurd.
    core = n_cells * n_features * dtype_bytes
    core += n_cells * n_actions * 1
    core += 4 * n_cells * n_actions * dtype_bytes
    core += 3 * n_cells * dtype_bytes + n_cells * 4
    scratch = 2 * n_cells * n_actions * dtype_bytes
    diag = 0
    if diagnostics:
        diag = n_cells * n_actions * dtype_bytes + min(n_cells, 1024) * n_actions * n_actions * 16
    total = int(core + scratch + diag)
    return RAQICMemoryEstimate(
        n_cells,
        n_actions,
        n_features,
        dtype_bytes,
        int(core),
        int(scratch),
        int(diag),
        total,
        total / 1024 / 1024,
    )


def choose_chunk_size(
    n_cells: int,
    n_actions: int,
    n_features: int,
    free_memory_bytes: int | None,
    dtype_bytes: int = 8,
    safety_fraction: float = 0.50,
) -> int:
    if n_cells <= 0:
        return 0
    if free_memory_bytes is None or free_memory_bytes <= 0:
        return n_cells
    per_cell = estimate_dense_raqic_memory(
        1, n_actions, n_features, dtype_bytes=dtype_bytes
    ).total_bytes
    cap = int((free_memory_bytes * safety_fraction) // max(1, per_cell))
    return max(1, min(n_cells, cap))
