"""Graph-safe scientific epoch snapshots for the accelerated OWL pipeline.

Two different snapshot classes are intentionally maintained:

* ``_tick_start_*`` arrays are *coordinate-resident* snapshots captured before
  environment/action mutation.  They are consumed by post-tick outcome/trust
  learning and must never move with cells.
* ``pre_*`` arrays are *cell-resident causal audit* snapshots captured after
  utility/authority construction.  They move with the cell during action
  consequences, matching :func:`owl.engine.loop.capture_pre_decision_state`.

Conflating these epochs caused delayed CPU/accelerated trajectory divergence.
"""

from __future__ import annotations

from typing import Any

from owl.gpu.array_write import write_array


def ensure_scientific_snapshot_buffers(ds: Any) -> None:
    """Allocate fixed-shape snapshot buffers once before graph capture."""
    xp = ds.xp
    shape = ds.health.shape
    dtype = ds.health.dtype
    for name in (
        "_tick_start_resource",
        "_tick_start_health",
        "_tick_start_integration",
    ):
        current = ds.arrays.get(name)
        if current is None or getattr(current, "shape", None) != shape:
            ds.arrays[name] = xp.zeros(shape, dtype=dtype)


def capture_tick_start_gpu(ds: Any, cfg: Any) -> None:
    """Capture the CPU loop's ``prev_*`` coordinate fields in-place."""
    del cfg
    ensure_scientific_snapshot_buffers(ds)
    write_array(ds, "_tick_start_resource", ds.resource)
    write_array(ds, "_tick_start_health", ds.health)
    write_array(ds, "_tick_start_integration", ds.integration)


def capture_pre_decision_state_gpu(ds: Any, cfg: Any) -> None:
    """Mirror ``owl.engine.loop.capture_pre_decision_state`` on ``ds.xp``."""
    xp = ds.xp
    alive = (ds.health > 0.0) & (~ds.obstacle)
    write_array(
        ds,
        "pre_resource",
        xp.where(
            alive,
            xp.clip(ds.resource, 0.0, float(cfg.resources.max_resource)),
            0.0,
        ).astype(ds.health.dtype),
    )
    write_array(
        ds,
        "pre_health",
        xp.where(alive, xp.clip(ds.health, 0.0, 1.0), 0.0).astype(ds.health.dtype),
    )
    write_array(
        ds,
        "pre_food",
        xp.where(alive, xp.clip(ds.food, 0.0, 1.0), 0.0).astype(ds.health.dtype),
    )
    if "starvation_debt" in ds.arrays:
        write_array(
            ds,
            "pre_starvation_debt",
            xp.where(
                alive,
                xp.clip(ds.starvation_debt, 0.0, 1.0),
                0.0,
            ).astype(ds.health.dtype),
        )

    # Utility/authority/top-down stages write these arrays immediately before
    # this snapshot boundary. Rewriting in place makes the epoch explicit and
    # guarantees dead rows are quiescent without changing their allocation.
    if "pre_authority" in ds.arrays:
        write_array(ds, "pre_authority", ds.pre_authority)
    if "pre_utilities" in ds.arrays:
        write_array(ds, "pre_utilities", ds.pre_utilities)
    if "pre_parent_bias" in ds.arrays:
        write_array(ds, "pre_parent_bias", ds.pre_parent_bias)
