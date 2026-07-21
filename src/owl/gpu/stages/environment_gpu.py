from __future__ import annotations

from typing import Any

from owl.gpu.array_write import write_array
from owl.gpu.kernels.stencil_kernels import (
    raw_toroidal_scalar_update,
    raw_toroidal_signal_update,
)
from owl.gpu.stencil import laplacian_4


def apply_obstacle_mask_gpu(ds: Any) -> None:
    xp = ds.xp
    if "obstacle" not in ds.arrays:
        return
    free = ~ds.obstacle
    for name in ("food", "toxin", "noise"):
        if name in ds.arrays:
            write_array(ds, name, xp.where(free, ds.arrays[name], 0.0))
    for name in ("signal", "signal_emission", "signal_reception", "signal_memory"):
        if name in ds.arrays:
            write_array(ds, name, xp.where(free[..., None], ds.arrays[name], 0.0))
    if "occupancy" in ds.arrays:
        write_array(ds, "occupancy", xp.where(free, ds.occupancy, -1))


def update_food_field_gpu(ds: Any, cfg: Any) -> None:
    """Recover the CPU food law with backend-neutral array operations."""
    xp = ds.xp
    mode = str(cfg.world.boundary_mode)
    food = ds.food
    diff = float(cfg.resources.food_diffusion)
    decay = float(cfg.resources.food_decay)
    advanced = bool(getattr(cfg.ecology, "advanced_enabled", False))

    if advanced:
        # CPU advanced order: diffuse current food, add logistic regrowth and
        # recycled waste, then decay and clip.
        updated = food + diff * laplacian_4(food, xp, mode) if diff else food.copy()
        carrying = float(cfg.ecology.food_carrying_capacity)
        growth = (
            float(cfg.ecology.food_regrowth_rate)
            * updated
            * (1.0 - updated / max(carrying, float(cfg.actions.epsilon)))
        )
        recycle = float(cfg.ecology.waste_recycle_rate) * ds.arrays.get("waste", 0.0)
        updated = updated + growth + recycle
        if decay:
            updated = updated - decay * updated
        if "waste" in ds.arrays:
            write_array(
                ds,
                "waste",
                xp.clip(
                    ds.arrays["waste"] * (1.0 - float(cfg.ecology.waste_decay)),
                    0.0,
                    1.0,
                ),
            )
        write_array(ds, "food", xp.clip(updated, 0.0, 1.0))
        return

    # The CPU update order is deliberately in place: add constant growth, diffuse
    # that updated field, then decay the updated result.
    updated = food + float(cfg.resources.food_growth)
    if diff:
        updated = updated + diff * laplacian_4(updated, xp, mode)
    if decay:
        updated = updated - decay * updated
    write_array(ds, "food", xp.clip(updated, 0.0, 1.0))


def update_toxin_field_gpu(ds: Any, cfg: Any) -> None:
    xp = ds.xp
    mode = str(cfg.world.boundary_mode)
    toxin = ds.toxin
    if (
        getattr(cfg.raqic, "full_gpu_stencil_backend", "vectorized") in ("raw_toroidal", "auto")
        and mode == "toroidal"
        and (getattr(xp, "__name__", "") == "cupy")
    ):
        out = xp.empty_like(toxin)
        raw_toroidal_scalar_update(
            toxin,
            ds.obstacle,
            out,
            xp,
            diffusion=float(cfg.resources.toxin_diffusion),
            decay=float(cfg.resources.toxin_decay),
            carrying=1.0,
        )
        write_array(ds, "toxin", out)
        return
    lap = laplacian_4(toxin, xp, mode)
    updated = (
        toxin
        + float(cfg.resources.toxin_diffusion) * lap
        - float(cfg.resources.toxin_decay) * toxin
    )
    write_array(ds, "toxin", xp.clip(updated, 0.0, 1.0))


def _channel_coefficients(ds: Any, cfg: Any, dtype: Any) -> Any:
    xp = ds.xp
    channels = int(ds.signal.shape[-1])
    diff = list(cfg.communication.diffusion)
    decay = list(cfg.communication.decay)
    if len(diff) < channels:
        diff += [diff[-1] if diff else 0.0] * (channels - len(diff))
    if len(decay) < channels:
        decay += [decay[-1] if decay else 0.0] * (channels - len(decay))
    key = (
        "signal_coefficients",
        str(dtype),
        channels,
        tuple(diff[:channels]),
        tuple(decay[:channels]),
    )
    cached = ds.metadata.get("environment_signal_coefficients")
    if cached is None or cached[0] != key:
        cached = (
            key,
            xp.asarray(diff[:channels], dtype=dtype),
            xp.asarray(decay[:channels], dtype=dtype),
        )
        ds.metadata["environment_signal_coefficients"] = cached
    return (cached[1], cached[2])


def update_signal_fields_gpu(ds: Any, cfg: Any) -> None:
    xp = ds.xp
    if "signal" not in ds.arrays:
        return
    if not bool(cfg.communication.enabled):
        for name in ("signal", "signal_emission", "signal_reception"):
            if name in ds.arrays:
                ds.arrays[name].fill(0.0)
        return

    sig = ds.signal
    mode = str(cfg.world.boundary_mode)
    diffusion, decay = _channel_coefficients(ds, cfg, sig.dtype)
    emission = ds.arrays.get("signal_emission")
    out = sig.copy()
    for c in range(sig.shape[-1]):
        plane = sig[..., c]
        updated = plane
        if float(diffusion[c]):
            updated = updated + diffusion[c] * laplacian_4(updated, xp, mode)
        if float(decay[c]):
            updated = updated - decay[c] * updated
        if emission is not None:
            updated = updated + emission[..., c]
        out[..., c] = updated
    write_array(ds, "signal", xp.clip(out, 0.0, 1.0))
    if emission is not None:
        emission.fill(0.0)
    if "signal_source_id" in ds.arrays:
        ds.arrays["signal_source_id"][ds.signal <= 0.0] = -1


def prepare_graph_environment_buffers(ds: Any, cfg: Any, scratch: Any) -> dict[str, Any]:
    """Preallocate fixed-pointer environment outputs for CUDA graph capture."""
    food_out = scratch.get("food_next")
    toxin_out = scratch.get("toxin_next")
    signal_out = scratch.get("signal_next")
    if food_out.shape != ds.food.shape or food_out.dtype != ds.food.dtype:
        food_out = ds.xp.empty_like(ds.food)
    if toxin_out.shape != ds.toxin.shape or toxin_out.dtype != ds.toxin.dtype:
        toxin_out = ds.xp.empty_like(ds.toxin)
    if signal_out.shape != ds.signal.shape or signal_out.dtype != ds.signal.dtype:
        signal_out = ds.xp.empty_like(ds.signal)
    diffusion, decay = _channel_coefficients(ds, cfg, ds.signal.dtype)
    return {
        "food_out": food_out,
        "toxin_out": toxin_out,
        "signal_out": signal_out,
        "signal_diffusion": diffusion,
        "signal_decay": decay,
    }


def update_environment_graph_safe(ds: Any, cfg: Any, buffers: dict[str, Any]) -> None:
    """Allocation-free toroidal update suitable for stream capture.

    It writes to fixed scratch arrays and copies back into persistent state
    arrays, so replay always reads and writes the same device pointers.
    """
    xp = ds.xp
    if getattr(xp, "__name__", "") != "cupy":
        update_environment_gpu(ds, cfg)
        return
    if str(cfg.world.boundary_mode) != "toroidal":
        raise RuntimeError("graph-safe environment kernel currently requires toroidal boundaries")
    raw_toroidal_scalar_update(
        ds.food,
        ds.obstacle,
        buffers["food_out"],
        xp,
        diffusion=float(cfg.resources.food_diffusion),
        decay=float(cfg.resources.food_decay),
        growth=float(cfg.resources.food_growth),
        carrying=float(getattr(cfg.ecology, "food_carrying_capacity", 1.0)),
    )
    raw_toroidal_scalar_update(
        ds.toxin,
        ds.obstacle,
        buffers["toxin_out"],
        xp,
        diffusion=float(cfg.resources.toxin_diffusion),
        decay=float(cfg.resources.toxin_decay),
        carrying=1.0,
    )
    raw_toroidal_signal_update(
        ds.signal,
        ds.obstacle,
        buffers["signal_diffusion"],
        buffers["signal_decay"],
        buffers["signal_out"],
        xp,
    )
    xp.copyto(ds.food, buffers["food_out"])
    xp.copyto(ds.toxin, buffers["toxin_out"])
    xp.copyto(ds.signal, buffers["signal_out"])


def update_environment_gpu(ds: Any, cfg: Any) -> None:
    update_food_field_gpu(ds, cfg)
    update_toxin_field_gpu(ds, cfg)
    update_signal_fields_gpu(ds, cfg)
    apply_obstacle_mask_gpu(ds)
