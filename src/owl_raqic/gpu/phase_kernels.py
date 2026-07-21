from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PhaseCoefficientTable:
    coefficients: Any
    moduli: Any
    weights: Any
    feature_names: tuple[str, ...]
    action_count: int
    active_primes: tuple[int, ...]
    modulus_power: int = 2


def build_phase_coefficients(
    feature_names: tuple[str, ...],
    action_count: int,
    active_primes: tuple[int, ...],
    xp: Any = np,
    modulus_power: int = 2,
) -> PhaseCoefficientTable:
    """Canonical deterministic integer coefficients for device-native RAQIC phases."""
    coeff = xp.zeros((int(action_count), len(feature_names), len(active_primes)), dtype=xp.int32)
    for a in range(int(action_count)):
        for f, _name in enumerate(feature_names):
            for k, p in enumerate(active_primes):
                # Canonical, deterministic, non-hash coefficient. Avoid Python hash.
                coeff[a, f, k] = ((a + 1) * (f + 3) * (k + 5) + p) % (p**modulus_power)
    moduli = xp.asarray([int(p) ** int(modulus_power) for p in active_primes], dtype=xp.int32)
    weights = xp.asarray([1.0 / max(1, int(p)) for p in active_primes], dtype=xp.float64)
    weights = weights / xp.sum(weights)
    return PhaseCoefficientTable(
        coeff,
        moduli,
        weights,
        tuple(feature_names),
        int(action_count),
        tuple(active_primes),
        int(modulus_power),
    )


def compute_canonical_phases(
    feature_bins: Any,
    table: PhaseCoefficientTable,
    xp: Any = np,
    epsilon_adelic: float = 1.0,
    base_phase: Any | None = None,
) -> Any:
    """Compute phase[n,a] from feature bins and active finite-prime table."""
    bins = xp.asarray(feature_bins, dtype=xp.int32)
    if bins.ndim != 2:
        raise ValueError("feature_bins must have shape [N,F]")
    coeff = table.coefficients
    # q[n,a,p] = sum_f coeff[a,f,p]*bins[n,f] mod p^k
    q = xp.einsum("nf,afp->nap", bins, coeff).astype(xp.int64)
    mod = table.moduli.astype(xp.int64)
    q = q % mod[None, None, :]
    frac = q / xp.maximum(mod[None, None, :].astype(xp.float64), 1.0)
    phase = (
        2.0 * xp.pi * float(epsilon_adelic) * xp.sum(frac * table.weights[None, None, :], axis=-1)
    )
    if base_phase is not None:
        phase = phase + xp.asarray(base_phase)
    return xp.mod(phase, 2.0 * xp.pi)


def canonical_phase_numpy(
    feature_bins: Any,
    feature_names: tuple[str, ...],
    action_count: int,
    active_primes: tuple[int, ...],
    epsilon_adelic: float = 1.0,
) -> Any:
    table = build_phase_coefficients(feature_names, action_count, active_primes, xp=np)
    return np.asarray(
        compute_canonical_phases(feature_bins, table, xp=np, epsilon_adelic=epsilon_adelic)
    )
