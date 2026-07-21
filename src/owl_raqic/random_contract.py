"""Versioned counter-based scientific RNG shared by scalar and array paths."""

from __future__ import annotations

from enum import IntEnum
from typing import Any

import numpy as np

RNG_CONTRACT_VERSION = "owl.counter-rng.v1"


class RNGStream(IntEnum):
    RAQIC_READOUT = 100
    MOVEMENT_TIE = 200
    INGESTION_OUTCOME = 250
    REPRODUCTION_TIE = 300
    REPRODUCTION_GATE = 310
    REPRODUCTION_SITE = 320
    REPRODUCTION_MUTATION = 330
    TOPOLOGY_TIE = 400
    ENVIRONMENT_NOISE = 500
    PHASE_NOISE = 600
    QISKIT_READOUT = 700


def rng_stream_registry() -> dict[str, int]:
    """Return immutable stream metadata without changing keys or draw counts."""
    return {stream.name: int(stream) for stream in RNGStream}


_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX2 = np.uint64(0x94D049BB133111EB)


def _xp_for(*values: Any) -> Any:
    for value in values:
        if value.__class__.__module__.startswith("cupy"):
            import cupy as cp  # pragma: no cover

            return cp
    return np


def _as_backend_array(value: Any, xp: Any, dtype: Any) -> Any:
    """Convert a counter-RNG input explicitly onto the requested backend.

    NumPy reference paths must not receive live CuPy device scalars because CuPy
    intentionally forbids implicit host conversion.  CuPy paths can accept Python
    or NumPy scalars through cupy.asarray.
    """
    if xp is np:
        try:
            import cupy as cp  # pragma: no cover

            if isinstance(value, cp.ndarray):
                value = cp.asnumpy(value)
        except Exception:
            pass
    return xp.asarray(value, dtype=dtype)


def _mix64(value: Any, xp: Any) -> Any:
    z = xp.asarray(value, dtype=xp.uint64) & xp.uint64(_MASK)
    if xp is np:
        with np.errstate(over="ignore"):
            z = (z ^ (z >> xp.uint64(30))) * xp.uint64(_MIX1)
            z = (z ^ (z >> xp.uint64(27))) * xp.uint64(_MIX2)
    else:
        z = (z ^ (z >> xp.uint64(30))) * xp.uint64(_MIX1)
        z = (z ^ (z >> xp.uint64(27))) * xp.uint64(_MIX2)
    return (z ^ (z >> xp.uint64(31))) & xp.uint64(_MASK)


def uniform_u64(
    seed: int,
    tick: int,
    ow_id: Any,
    stream: int | RNGStream,
    draw_slot: Any = 0,
    xp: Any | None = None,
) -> Any:
    xp = xp or _xp_for(seed, tick, ow_id, draw_slot)
    key = _as_backend_array(seed, xp, xp.uint64) + xp.uint64(_GOLDEN)
    key = key ^ _mix64(_as_backend_array(tick, xp, xp.uint64) + xp.uint64(0xD1B54A32D192ED03), xp)
    key = key ^ _mix64(_as_backend_array(ow_id, xp, xp.uint64) + xp.uint64(0xABC98388FB8FAC03), xp)
    key = key ^ _mix64(
        _as_backend_array(int(stream), xp, xp.uint64) + xp.uint64(0x8CB92BA72F3D8DD7), xp
    )
    key = key ^ _mix64(
        _as_backend_array(draw_slot, xp, xp.uint64) + xp.uint64(0xDB4F0B9175AE2165), xp
    )
    return _mix64(key, xp)


def uniform01(
    seed: int,
    tick: int,
    ow_id: Any,
    stream: int | RNGStream,
    draw_slot: Any = 0,
    xp: Any | None = None,
    dtype: Any | None = None,
) -> Any:
    xp = xp or _xp_for(seed, tick, ow_id, draw_slot)
    words = uniform_u64(seed, tick, ow_id, stream, draw_slot, xp=xp)
    out = (words >> xp.uint64(11)).astype(xp.float64) * (1.0 / float(1 << 53))
    return out.astype(dtype or xp.float64, copy=False)


def normal01(
    seed: int,
    tick: int,
    ow_id: Any,
    stream: int | RNGStream,
    draw_slot: Any = 0,
    xp: Any | None = None,
    dtype: Any | None = None,
) -> Any:
    xp = xp or _xp_for(seed, tick, ow_id, draw_slot)
    u1 = xp.clip(uniform01(seed, tick, ow_id, stream, 2 * xp.asarray(draw_slot), xp=xp), 1e-15, 1.0)
    u2 = uniform01(seed, tick, ow_id, stream, 2 * xp.asarray(draw_slot) + 1, xp=xp)
    z = xp.sqrt(-2.0 * xp.log(u1)) * xp.cos(2.0 * xp.pi * u2)
    return z.astype(dtype or xp.float64, copy=False)


def categorical(
    probabilities: Any,
    seed: int,
    tick: int,
    ow_id: Any,
    stream: int | RNGStream = RNGStream.RAQIC_READOUT,
    xp: Any | None = None,
) -> Any:
    xp = xp or _xp_for(probabilities, ow_id)
    p = xp.asarray(probabilities)
    u = uniform01(seed, tick, ow_id, stream, 0, xp=xp, dtype=p.dtype).reshape((-1, 1))
    cdf = xp.cumsum(p, axis=-1)
    cdf[..., -1] = 1.0
    return xp.argmax(cdf >= u, axis=-1).astype(xp.int32)
