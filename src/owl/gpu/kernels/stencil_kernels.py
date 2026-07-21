from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from owl.gpu.stencil import laplacian_4, neighbor_sum_8


@dataclass(frozen=True)
class KernelKey:
    source_sha256: str
    options: tuple[str, ...]
    device_id: int
    compute_capability: str


@dataclass
class StencilScratch:
    source_epoch: dict[str, int]
    local_alive_density: Any | None = None
    food_mean: Any | None = None
    toxin_mean: Any | None = None
    phase_sin_sum: Any | None = None
    phase_cos_sum: Any | None = None


_RAW_SOURCE = r"""
extern "C" __global__
void toroidal_laplacian4_f32(const float* x, float* out, int h, int w) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int n = h * w;
    if (idx >= n) return;
    int y = idx / w, xx = idx - y*w;
    int yn = (y == 0) ? h-1 : y-1, ys = (y == h-1) ? 0 : y+1;
    int xw = (xx == 0) ? w-1 : xx-1, xe = (xx == w-1) ? 0 : xx+1;
    float c = x[idx];
    out[idx] = x[yn*w+xx] + x[ys*w+xx] + x[y*w+xw] + x[y*w+xe] - 4.0f*c;
}
extern "C" __global__
void toroidal_laplacian4_f64(const double* x, double* out, int h, int w) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int n = h * w;
    if (idx >= n) return;
    int y = idx / w, xx = idx - y*w;
    int yn = (y == 0) ? h-1 : y-1, ys = (y == h-1) ? 0 : y+1;
    int xw = (xx == 0) ? w-1 : xx-1, xe = (xx == w-1) ? 0 : xx+1;
    double c = x[idx];
    out[idx] = x[yn*w+xx] + x[ys*w+xx] + x[y*w+xw] + x[y*w+xe] - 4.0*c;
}
extern "C" __global__
void update_scalar_f32(
    const float* x, const bool* obstacle, float* out, int h, int w,
    float diffusion, float decay, float growth, float carrying
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int n = h*w; if (idx >= n) return;
    if (obstacle[idx]) { out[idx] = 0.0f; return; }
    int y = idx / w, xx = idx - y*w;
    int yn = (y == 0) ? h-1 : y-1, ys = (y == h-1) ? 0 : y+1;
    int xw = (xx == 0) ? w-1 : xx-1, xe = (xx == w-1) ? 0 : xx+1;
    float c = x[idx];
    float lap = x[yn*w+xx] + x[ys*w+xx] + x[y*w+xw] + x[y*w+xe] - 4.0f*c;
    float logistic = (growth == 0.0f) ? 0.0f : growth*c*(1.0f-c/fmaxf(carrying,1e-20f));
    float v = c + diffusion*lap + logistic - decay*c;
    out[idx] = fminf(fmaxf(v,0.0f),carrying);
}
extern "C" __global__
void update_scalar_f64(
    const double* x, const bool* obstacle, double* out, int h, int w,
    double diffusion, double decay, double growth, double carrying
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int n = h*w; if (idx >= n) return;
    if (obstacle[idx]) { out[idx] = 0.0; return; }
    int y = idx / w, xx = idx - y*w;
    int yn = (y == 0) ? h-1 : y-1, ys = (y == h-1) ? 0 : y+1;
    int xw = (xx == 0) ? w-1 : xx-1, xe = (xx == w-1) ? 0 : xx+1;
    double c = x[idx];
    double lap = x[yn*w+xx] + x[ys*w+xx] + x[y*w+xw] + x[y*w+xe] - 4.0*c;
    double logistic = (growth == 0.0) ? 0.0 : growth*c*(1.0-c/fmax(carrying,1e-30));
    double v = c + diffusion*lap + logistic - decay*c;
    out[idx] = fmin(fmax(v,0.0),carrying);
}
extern "C" __global__
void update_signal_f32(
    const float* x, const bool* obstacle, const float* diffusion,
    const float* decay, float* out, int h, int w, int channels
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int n = h*w*channels; if (idx >= n) return;
    int c = idx % channels;
    int cell = idx / channels;
    if (obstacle[cell]) { out[idx] = 0.0f; return; }
    int y = cell / w, xx = cell - y*w;
    int yn = (y == 0) ? h-1 : y-1, ys = (y == h-1) ? 0 : y+1;
    int xw = (xx == 0) ? w-1 : xx-1, xe = (xx == w-1) ? 0 : xx+1;
    int nidx=(yn*w+xx)*channels+c, sidx=(ys*w+xx)*channels+c;
    int widx=(y*w+xw)*channels+c, eidx=(y*w+xe)*channels+c;
    float center=x[idx];
    float lap=x[nidx]+x[sidx]+x[widx]+x[eidx]-4.0f*center;
    float v=center+diffusion[c]*lap-decay[c]*center;
    out[idx]=fminf(fmaxf(v,0.0f),1.0f);
}
extern "C" __global__
void update_signal_f64(
    const double* x, const bool* obstacle, const double* diffusion,
    const double* decay, double* out, int h, int w, int channels
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int n = h*w*channels; if (idx >= n) return;
    int c = idx % channels;
    int cell = idx / channels;
    if (obstacle[cell]) { out[idx] = 0.0; return; }
    int y = cell / w, xx = cell - y*w;
    int yn = (y == 0) ? h-1 : y-1, ys = (y == h-1) ? 0 : y+1;
    int xw = (xx == 0) ? w-1 : xx-1, xe = (xx == w-1) ? 0 : xx+1;
    int nidx=(yn*w+xx)*channels+c, sidx=(ys*w+xx)*channels+c;
    int widx=(y*w+xw)*channels+c, eidx=(y*w+xe)*channels+c;
    double center=x[idx];
    double lap=x[nidx]+x[sidx]+x[widx]+x[eidx]-4.0*center;
    double v=center+diffusion[c]*lap-decay[c]*center;
    out[idx]=fmin(fmax(v,0.0),1.0);
}
extern "C" __global__
void fused_local_f32(
    const bool* alive, const float* food, const float* toxin, const float* phase,
    float* alive_mean, float* food_mean, float* toxin_mean,
    float* phase_sin_sum, float* phase_cos_sum, int h, int w
) {
    int idx=blockDim.x*blockIdx.x+threadIdx.x; int n=h*w; if(idx>=n)return;
    int y=idx/w, x=idx-y*w; float ac=0,fs=0,ts=0,ss=0,cs=0;
    for(int dy=-1;dy<=1;++dy) for(int dx=-1;dx<=1;++dx) {
        if(dx==0&&dy==0) continue;
        int yy=(y+dy+h)%h, xx=(x+dx+w)%w, j=yy*w+xx;
        ac += alive[j] ? 1.0f : 0.0f; fs+=food[j]; ts+=toxin[j];
        ss+=sinf(phase[j]); cs+=cosf(phase[j]);
    }
    alive_mean[idx]=ac*0.125f; food_mean[idx]=fs*0.125f;
    toxin_mean[idx]=ts*0.125f; phase_sin_sum[idx]=ss; phase_cos_sum[idx]=cs;
}
extern "C" __global__
void fused_local_f64(
    const bool* alive, const double* food, const double* toxin, const double* phase,
    double* alive_mean, double* food_mean, double* toxin_mean,
    double* phase_sin_sum, double* phase_cos_sum, int h, int w
) {
    int idx=blockDim.x*blockIdx.x+threadIdx.x; int n=h*w; if(idx>=n)return;
    int y=idx/w, x=idx-y*w; double ac=0,fs=0,ts=0,ss=0,cs=0;
    for(int dy=-1;dy<=1;++dy) for(int dx=-1;dx<=1;++dx) {
        if(dx==0&&dy==0) continue;
        int yy=(y+dy+h)%h, xx=(x+dx+w)%w, j=yy*w+xx;
        ac += alive[j] ? 1.0 : 0.0; fs+=food[j]; ts+=toxin[j];
        ss+=sin(phase[j]); cs+=cos(phase[j]);
    }
    alive_mean[idx]=ac*0.125; food_mean[idx]=fs*0.125;
    toxin_mean[idx]=ts*0.125; phase_sin_sum[idx]=ss; phase_cos_sum[idx]=cs;
}
"""

_MODULE_CACHE: dict[KernelKey, Any] = {}
_COMPILE_COUNT = 0
_KERNEL_OPTIONS = ("-std=c++11",)


def _kernel_key(xp: Any) -> KernelKey:
    device = xp.cuda.Device()
    capability = str(getattr(device, "compute_capability", "unknown"))
    return KernelKey(
        source_sha256=hashlib.sha256(_RAW_SOURCE.encode("utf-8")).hexdigest(),
        options=_KERNEL_OPTIONS,
        device_id=int(device.id),
        compute_capability=capability,
    )


def _module(xp: Any) -> Any:
    global _COMPILE_COUNT
    key = _kernel_key(xp)
    module = _MODULE_CACHE.get(key)
    if module is None:
        module = xp.RawModule(code=_RAW_SOURCE, options=_KERNEL_OPTIONS)
        _MODULE_CACHE[key] = module
        _COMPILE_COUNT += 1
    return module


def warm_stencil_kernels(xp: Any) -> None:
    """Compile the per-device kernel bundle before measured execution."""
    if getattr(xp, "__name__", "") == "cupy":
        _module(xp)


def stencil_kernel_compile_count() -> int:
    return int(_COMPILE_COUNT)


def _kernel(xp: Any, name: str) -> Any:
    return _module(xp).get_function(name)


def _launch_1d(kernel: Any, n: int, args: Any) -> None:
    block = 256
    grid = ((int(n) + block - 1) // block,)
    kernel(grid, (block,), args)


def raw_toroidal_laplacian4(arr: Any, xp: Any, out: Any | None = None) -> Any:
    """Cached raw CUDA Laplacian for float32/float64, vectorized otherwise."""
    if getattr(xp, "__name__", "") != "cupy":
        result = laplacian_4(arr, xp, "toroidal")
        if out is not None:
            out[...] = result
            return out
        return result
    if arr.dtype.kind != "f" or arr.dtype.itemsize not in (4, 8):
        result = laplacian_4(arr, xp, "toroidal")
        if out is not None:
            out[...] = result
            return out
        return result
    out = xp.empty_like(arr) if out is None else out
    suffix = "f32" if arr.dtype.itemsize == 4 else "f64"
    h, w = map(int, arr.shape)
    _launch_1d(_kernel(xp, f"toroidal_laplacian4_{suffix}"), h * w, (arr, out, h, w))
    return out


def raw_toroidal_scalar_update(
    arr: Any,
    obstacle: Any,
    out: Any,
    xp: Any,
    *,
    diffusion: float,
    decay: float,
    growth: float = 0.0,
    carrying: float = 1.0,
) -> Any:
    if getattr(xp, "__name__", "") != "cupy" or arr.dtype.itemsize not in (4, 8):
        lap = laplacian_4(arr, xp, "toroidal")
        logistic = growth * arr * (1.0 - arr / max(float(carrying), 1e-30))
        out[...] = xp.where(
            obstacle, 0.0, xp.clip(arr + diffusion * lap + logistic - decay * arr, 0.0, carrying)
        )
        return out
    suffix = "f32" if arr.dtype.itemsize == 4 else "f64"
    scalar = xp.float32 if suffix == "f32" else xp.float64
    h, w = map(int, arr.shape)
    _launch_1d(
        _kernel(xp, f"update_scalar_{suffix}"),
        h * w,
        (
            arr,
            obstacle,
            out,
            h,
            w,
            scalar(diffusion),
            scalar(decay),
            scalar(growth),
            scalar(carrying),
        ),
    )
    return out


def raw_toroidal_signal_update(
    signal: Any, obstacle: Any, diffusion: Any, decay: Any, out: Any, xp: Any
) -> Any:
    if getattr(xp, "__name__", "") != "cupy" or signal.dtype.itemsize not in (4, 8):
        channels = signal.shape[-1]
        for c in range(channels):
            lap = laplacian_4(signal[..., c], xp, "toroidal")
            out[..., c] = xp.clip(
                signal[..., c] + diffusion[c] * lap - decay[c] * signal[..., c], 0.0, 1.0
            )
        out[...] = xp.where(obstacle[..., None], 0.0, out)
        return out
    suffix = "f32" if signal.dtype.itemsize == 4 else "f64"
    h, w, channels = map(int, signal.shape)
    _launch_1d(
        _kernel(xp, f"update_signal_{suffix}"),
        h * w * channels,
        (signal, obstacle, diffusion, decay, out, h, w, channels),
    )
    return out


def _fused_local_raw_compatible(
    alive: Any,
    food: Any,
    toxin: Any,
    phase: Any,
    outputs: tuple[Any, Any, Any, Any, Any],
    xp: Any,
    boundary_mode: str,
) -> bool:
    """Return whether the raw fused kernel can safely consume these buffers.

    Raw CUDA pointers are untyped at the Python call boundary.  A float64 kernel
    receiving float32 output buffers writes eight-byte values into four-byte
    allocations and corrupts adjacent device memory.  Require exact dtype,
    shape, and contiguous-layout agreement before launching the raw kernel.
    """
    if getattr(xp, "__name__", "") != "cupy" or boundary_mode != "toroidal":
        return False
    if not (alive.shape == food.shape == toxin.shape == phase.shape):
        return False
    if food.dtype.itemsize not in (4, 8) or toxin.dtype != food.dtype or phase.dtype != food.dtype:
        return False
    if getattr(alive.dtype, "kind", "") != "b":
        return False
    for value in (alive, food, toxin, phase):
        flags = getattr(value, "flags", None)
        if flags is not None and not bool(getattr(flags, "c_contiguous", True)):
            return False
    for out in outputs:
        if out.shape != food.shape or out.dtype != food.dtype:
            return False
        flags = getattr(out, "flags", None)
        if flags is not None and not bool(getattr(flags, "c_contiguous", True)):
            return False
    return True


def fused_local_scratch(
    alive: Any,
    food: Any,
    toxin: Any,
    phase: Any,
    xp: Any,
    boundary_mode: str = "toroidal",
    outputs: tuple[Any, Any, Any, Any, Any] | None = None,
) -> StencilScratch:
    """Compute shared local statistics with a fail-closed raw-kernel gate."""
    if outputs is None:
        outputs = tuple(xp.empty_like(food) for _ in range(5))
    if len(outputs) != 5:
        raise ValueError(f"fused_local_scratch requires five outputs, got {len(outputs)}")
    local_alive, food_mean, toxin_mean, phase_sin, phase_cos = outputs
    for name, out in zip(
        ("local_alive", "food_mean", "toxin_mean", "phase_sin", "phase_cos"),
        outputs,
        strict=True,
    ):
        if out.shape != food.shape:
            raise ValueError(f"{name} must have shape {food.shape}, got {out.shape}")

    use_raw = _fused_local_raw_compatible(alive, food, toxin, phase, outputs, xp, boundary_mode)
    if use_raw:
        suffix = "f32" if food.dtype.itemsize == 4 else "f64"
        h, w = map(int, food.shape)
        _launch_1d(
            _kernel(xp, f"fused_local_{suffix}"),
            h * w,
            (
                alive,
                food,
                toxin,
                phase,
                local_alive,
                food_mean,
                toxin_mean,
                phase_sin,
                phase_cos,
                h,
                w,
            ),
        )
    else:
        # The vectorized fallback is safe across mixed source/output dtypes and
        # intentionally computes in each output's storage dtype. This recovers
        # the float32 OWL physical contract even if a caller supplies promoted
        # source arrays.
        local_dtype = local_alive.dtype
        food_dtype = food_mean.dtype
        toxin_dtype = toxin_mean.dtype
        phase_dtype = phase_sin.dtype
        local_alive[...] = neighbor_sum_8(
            alive.astype(local_dtype, copy=False), xp, boundary_mode
        ) / xp.asarray(8.0, dtype=local_dtype)
        food_mean[...] = neighbor_sum_8(
            food.astype(food_dtype, copy=False), xp, boundary_mode
        ) / xp.asarray(8.0, dtype=food_dtype)
        toxin_mean[...] = neighbor_sum_8(
            toxin.astype(toxin_dtype, copy=False), xp, boundary_mode
        ) / xp.asarray(8.0, dtype=toxin_dtype)
        phase_work = phase.astype(phase_dtype, copy=False)
        phase_sin[...] = neighbor_sum_8(xp.sin(phase_work), xp, boundary_mode)
        phase_cos[...] = neighbor_sum_8(xp.cos(phase_work), xp, boundary_mode)
    return StencilScratch({}, local_alive, food_mean, toxin_mean, phase_sin, phase_cos)


def raw_kernel_cache_info() -> dict[str, int]:
    return {"module_count": len(_MODULE_CACHE), "compile_count": stencil_kernel_compile_count()}
