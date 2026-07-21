from __future__ import annotations

from typing import Any

from owl.gpu.array_write import write_array
from owl.gpu.stencil import neighbor_sum_8, shift_2d
from owl.science.counter_rng import RNGStream, normal01

_TWO_PI = 2.0 * 3.141592653589793


def _ow_ids(ds: Any) -> Any:
    xp = ds.xp
    flat = xp.arange(ds.phase.size, dtype=xp.uint64).reshape(ds.phase.shape)
    occupancy = ds.arrays.get("occupancy")
    if occupancy is None:
        return flat
    return xp.where(occupancy >= 0, occupancy.astype(xp.uint64), flat)


def _neighbor_phase_mean(phase: Any, xp: Any, mode: str) -> Any:
    sin_mean = neighbor_sum_8(xp.sin(phase), xp, mode) / 8.0
    cos_mean = neighbor_sum_8(xp.cos(phase), xp, mode) / 8.0
    return xp.arctan2(sin_mean, cos_mean)


def update_phase_gpu(ds: Any, cfg: Any) -> None:
    xp = ds.xp
    mode = str(cfg.world.boundary_mode)
    # The authoritative physical phase state is float32 in WorldState. Audit64
    # promotes RAQIC/reduction work, but it must not create a different oscillator
    # trajectory merely by retaining extra state precision between ticks.
    phase = ds.phase.astype(xp.float32, copy=False)
    live = (ds.health > 0.0) & (ds.boundary > 0.0)
    parent = ds.arrays.get("_parent_phase", xp.zeros_like(phase)).astype(xp.float32, copy=False)
    noise = (
        normal01(
            int(cfg.world.seed),
            ds.arrays.get("_device_tick", int(ds.tick)),
            _ow_ids(ds),
            RNGStream.PHASE_NOISE,
            0,
            xp=xp,
            dtype=xp.float64,
        )
        * float(cfg.phase.phase_noise_sigma)
    ).astype(xp.float32)
    if bool(getattr(cfg.hierarchy, "dynamic_patches", False)):
        directions = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
        weights = ds.arrays.get("same_scale_weight")
        lag = ds.arrays.get("phase_lag", xp.zeros_like(phase)).astype(xp.float32, copy=False)
        drive = xp.zeros_like(phase, dtype=xp.float32)
        for index, (dy, dx) in enumerate(directions):
            neighbor = shift_2d(phase, xp, int(dy), int(dx), "toroidal")
            weight = (
                xp.float32(1.0 / 8.0)
                if weights is None
                else weights[..., index].astype(xp.float32, copy=False)
            )
            drive = (drive + weight * xp.sin(neighbor - phase - lag).astype(xp.float32)).astype(
                xp.float32
            )
        parent_weight = ds.arrays.get(
            "parent_weight", xp.full_like(phase, float(cfg.phase.parent_coupling))
        ).astype(xp.float32, copy=False)
        frequency = ds.arrays.get(
            "phase_frequency", xp.full_like(phase, float(cfg.phase.base_omega))
        ).astype(xp.float32, copy=False)
        delta = (
            frequency + drive + parent_weight * xp.sin(parent - phase).astype(xp.float32) + noise
        ).astype(xp.float32)
        updated = xp.mod((phase + delta).astype(xp.float32), xp.float32(_TWO_PI)).astype(xp.float32)
        updated = xp.where(live, updated, xp.float32(0.0))
    else:
        neighbor_phase = _neighbor_phase_mean(phase, xp, mode).astype(xp.float32)
        same_pull = (
            xp.float32(cfg.phase.same_scale_coupling) * xp.sin(neighbor_phase - phase)
        ).astype(xp.float32)
        parent_pull = (xp.float32(cfg.phase.parent_coupling) * xp.sin(parent - phase)).astype(
            xp.float32
        )
        delta = (xp.float32(cfg.phase.base_omega) + same_pull + parent_pull + noise).astype(
            xp.float32
        )
        advanced = xp.mod((phase + delta).astype(xp.float32), xp.float32(_TWO_PI)).astype(
            xp.float32
        )
        updated = xp.where(live, advanced, phase)
    write_array(ds, "phase", updated)


def compute_local_synchrony_gpu(ds: Any, cfg: Any) -> Any:
    xp = ds.xp
    mode = str(cfg.world.boundary_mode)
    phase = ds.phase.astype(xp.float32, copy=False)
    sin_phase = xp.sin(phase).astype(xp.float32)
    cos_phase = xp.cos(phase).astype(xp.float32)
    sin_local = (sin_phase + neighbor_sum_8(sin_phase, xp, mode)) / 9.0
    cos_local = (cos_phase + neighbor_sum_8(cos_phase, xp, mode)) / 9.0
    sync = xp.clip(sin_local * sin_local + cos_local * cos_local, 0.0, 1.0)
    live = (ds.health > 0.0) & (ds.boundary > 0.0)
    sync = xp.where(live, sync, 0.0)
    write_array(ds, "_synchrony_current", sync)
    return sync


def compute_cell_coherence_gpu(ds: Any, cfg: Any) -> Any:
    xp = ds.xp
    phase = ds.phase.astype(xp.float32, copy=False)
    neighbor_phase = _neighbor_phase_mean(phase, xp, str(cfg.world.boundary_mode)).astype(
        xp.float32
    )
    coherence = xp.clip(
        xp.float32(0.5) + xp.float32(0.5) * xp.cos(neighbor_phase - phase), 0.0, 1.0
    ).astype(xp.float32)
    live = (ds.health > 0.0) & (ds.boundary > 0.0)
    coherence = xp.where(live, coherence, 0.0)
    write_array(ds, "_coherence_current", coherence)
    return coherence


def compute_cross_scale_coupling_gpu(ds: Any, cfg: Any) -> Any:
    xp = ds.xp
    phase = ds.phase.astype(xp.float32, copy=False)
    parent = ds.arrays.get("_parent_phase", xp.zeros_like(phase)).astype(xp.float32, copy=False)
    coupling = xp.clip(xp.float32(0.5) + xp.float32(0.5) * xp.cos(parent - phase), 0.0, 1.0).astype(
        xp.float32
    )
    live = (ds.health > 0.0) & (ds.boundary > 0.0)
    coupling = xp.where(live, coupling, 0.0)
    write_array(ds, "_cross_scale_current", coupling)
    return coupling
