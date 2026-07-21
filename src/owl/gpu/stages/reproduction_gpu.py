"""GPU adapter for the shared deterministic reproduction transition."""

from __future__ import annotations

from typing import Any

from owl.science.reproduction_contract import apply_reproduction_arrays

NEIGHBORS = ((-1, 0), (1, 0), (0, 1), (0, -1), (-1, 1), (-1, -1), (1, 1), (1, -1))


def apply_reproduction_gpu(ds: Any, cfg: Any) -> dict[str, Any]:
    diag = apply_reproduction_arrays(
        ds.arrays,
        ds.scalars,
        cfg,
        tick=int(ds.tick),
        xp=ds.xp,
        patch_shape=tuple(ds.patch_arrays["integration"].shape),
    )
    return {
        "children": int(diag.accepted),
        "candidates": int(diag.candidates),
        "child_ids": list(diag.child_ids),
        "_cadc_plan": diag.plan,
        "_cadc_accepted_indices": diag.accepted_indices,
    }


def apply_reproduction_graph_static(ds: Any, cfg: Any) -> Any:
    return apply_reproduction_gpu(ds, cfg)
