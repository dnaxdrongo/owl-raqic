from __future__ import annotations

from typing import Any

from owl.gpu.array_write import write_array
from owl.gpu.kernels.stencil_kernels import fused_local_scratch
from owl.gpu.stencil import local_mean_3x3, neighbor_sum_8


def prepare_sensing_stencil_scratch(ds: Any, cfg: Any, scratch_manager: Any) -> Any:
    """Compute shared neighborhood fields once for the pre-decision epoch."""
    outputs = (
        scratch_manager.get("local_alive"),
        scratch_manager.get("local_food"),
        scratch_manager.get("local_toxin"),
        scratch_manager.get("phase_sin_sum"),
        scratch_manager.get("phase_cos_sum"),
    )
    live = (ds.health > 0.0) & ~ds.obstacle
    result = fused_local_scratch(
        live, ds.food, ds.toxin, ds.phase, ds.xp, str(cfg.world.boundary_mode), outputs=outputs
    )
    epochs = ds.metadata.get("field_epochs")
    if epochs is not None:
        result.source_epoch = epochs.snapshot(["health", "obstacle", "food", "toxin", "phase"])
    ds.metadata["sensing_stencil_scratch"] = result
    return result


def compute_sensing_bundle_gpu(ds: Any, cfg: Any, stencil_scratch: Any | None = None) -> None:
    xp = ds.xp
    mode = str(cfg.world.boundary_mode)
    live = (ds.health > 0.0) & ~ds.obstacle
    scratch = stencil_scratch or ds.metadata.get("sensing_stencil_scratch")
    epochs = ds.metadata.get("field_epochs")
    use_scratch = scratch is not None
    if use_scratch and epochs is not None and scratch.source_epoch:
        expected = scratch.source_epoch
        if not epochs.matches(expected):
            if bool(getattr(cfg.debug, "assert_invariants", False)):
                raise RuntimeError("stale sensing stencil scratch")
            use_scratch = False
    if use_scratch:
        write_array(ds, "food_mean", xp.where(live, scratch.food_mean, 0.0))
        write_array(ds, "alive_density", scratch.local_alive_density)
        write_array(ds, "toxin_mean", scratch.toxin_mean)
    else:
        # Diagnostic caches retain the existing boundary-aware stencil policy.
        write_array(ds, "food_mean", xp.where(live, local_mean_3x3(ds.food, xp, mode), 0.0))
        write_array(ds, "alive_density", local_mean_3x3(live.astype(ds.health.dtype), xp, mode))
        write_array(ds, "toxin_mean", local_mean_3x3(ds.toxin, xp, mode))

    if not bool(cfg.communication.enabled):
        write_array(ds, "signal_reception", xp.zeros_like(ds.signal_reception))
        return

    # Exact CPU scientific law: local signal is the center/neighbour blend,
    # then receiver traits, trust, boundary openness, and liveness gate it.
    neighbor = neighbor_sum_8(ds.signal, xp, "toroidal") / 8.0
    local_signal = 0.5 * ds.signal + 0.5 * neighbor
    openness = 0.25 + 0.75 * xp.clip(ds.boundary, 0.0, 1.0)
    reception = (
        local_signal
        * xp.clip(ds.receive_sensitivity, 0.0, 1.0)[..., None]
        * xp.clip(ds.channel_receptivity, 0.0, 1.0)
        * xp.clip(ds.channel_trust_local, 0.0, 1.0)
        * openness[..., None]
        * live[..., None]
    )
    if bool(getattr(cfg.communication, "source_tracking_enabled", False)):
        reception *= (
            1.0
            - 0.5
            * xp.clip(ds.arrays.get("deception_memory", xp.zeros_like(ds.health)), 0.0, 1.0)[
                ..., None
            ]
        )
        reception *= (
            0.50
            + 0.50
            * xp.clip(ds.arrays.get("source_confidence", xp.ones_like(ds.health)), 0.0, 1.0)[
                ..., None
            ]
        )
    write_array(ds, "signal_reception", xp.clip(reception, 0.0, 1.0).astype(ds.signal.dtype))
