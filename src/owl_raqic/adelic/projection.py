from __future__ import annotations

from typing import cast

import numpy as np

from .kernels import finite_adelic_kernel
from .phases import adelic_character_phase_proxy


def normalize_complex_amplitudes(amplitudes: np.ndarray) -> np.ndarray:
    amps = np.asarray(amplitudes, dtype=complex)
    norm = np.linalg.norm(amps)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("cannot normalize zero or nonfinite amplitude vector")
    return cast(np.ndarray, amps / norm)


def finite_feature_projection(
    features: np.ndarray,
    adelic_codes: list[dict[int, int]] | None,
    primes: tuple[int, ...],
    prime_weights: dict[int, float],
    epsilon_adelic: float = 1.0,
) -> np.ndarray:
    x = np.asarray(features, dtype=float)
    if x.ndim != 1:
        raise ValueError("features must be one-dimensional")
    codes: list[dict[int, int] | None] = (
        list(adelic_codes) if adelic_codes is not None else [None] * len(x)
    )
    out = []
    for idx, value in enumerate(x):
        code = codes[idx] if idx < len(codes) else None
        k = finite_adelic_kernel(abs(float(value)), code, {}, primes, prime_weights)
        out.append(float(value) + epsilon_adelic * k)
    return np.asarray(out, dtype=float)


def action_phase_vector(
    num_den_pairs: list[tuple[int, int]], primes: tuple[int, ...], diagonal_test: bool = False
) -> np.ndarray:
    return np.asarray(
        [
            adelic_character_phase_proxy(n, d, primes, diagonal_test=diagonal_test)
            for n, d in num_den_pairs
        ],
        dtype=float,
    )
