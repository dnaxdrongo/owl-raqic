from __future__ import annotations

from typing import Any

import numpy as np


def collect_gpu_summary_from_state(state: Any, cfg: Any) -> dict[str, Any]:
    """Collect compact metrics after a gpu_full tick."""
    alive = (state.health > 0.0) & (~state.obstacle)
    return {
        "tick": int(state.tick),
        "gpu_live_count": int(np.count_nonzero(alive)),
        "gpu_mean_health": float(np.mean(state.health[alive])) if np.any(alive) else 0.0,
        "gpu_mean_resource": float(np.mean(state.resource[alive])) if np.any(alive) else 0.0,
        "gpu_food_total": float(np.sum(state.food, dtype=np.float64)),
        "gpu_toxin_total": float(np.sum(state.toxin, dtype=np.float64)),
        "gpu_raqic_entropy_mean": float(
            -np.mean(
                np.sum(
                    np.where(
                        state.raqic_probabilities > 0,
                        state.raqic_probabilities * np.log(state.raqic_probabilities + 1e-8),
                        0.0,
                    ),
                    axis=-1,
                )
            )
        )
        if getattr(state, "raqic_probabilities", None) is not None
        else 0.0,
    }
