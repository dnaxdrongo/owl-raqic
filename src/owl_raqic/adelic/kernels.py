from __future__ import annotations

import math

from .padic import bounded_padic_distance


def real_kernel(distance: float, eta: float = 1.0) -> float:
    if distance < 0:
        raise ValueError("distance must be nonnegative")
    return float(math.exp(-eta * distance))


def reciprocal_kernel(distance: float) -> float:
    if distance < 0:
        raise ValueError("distance must be nonnegative")
    return float(1.0 / (1.0 + distance))


def padic_kernel(distance: float, eta: float = 1.0) -> float:
    return real_kernel(distance, eta=eta)


def finite_adelic_distance(
    real_distance: float,
    padic_codes_a: dict[int, int] | None,
    padic_codes_b: dict[int, int] | None,
    primes: tuple[int, ...],
    weights: dict[int, float],
    real_weight: float = 1.0,
) -> float:
    if real_distance < 0:
        raise ValueError("real_distance must be nonnegative")
    dist = real_weight * float(real_distance)
    padic_codes_a = padic_codes_a or {}
    padic_codes_b = padic_codes_b or {}
    for p in primes:
        a = int(padic_codes_a.get(p, 0))
        b = int(padic_codes_b.get(p, 0))
        dist += float(weights.get(p, 0.0)) * bounded_padic_distance(a, b, p)
    return float(dist)


def finite_adelic_kernel(
    real_distance: float,
    padic_codes_a: dict[int, int] | None,
    padic_codes_b: dict[int, int] | None,
    primes: tuple[int, ...],
    weights: dict[int, float],
    eta: float = 1.0,
) -> float:
    return real_kernel(
        finite_adelic_distance(real_distance, padic_codes_a, padic_codes_b, primes, weights),
        eta=eta,
    )
