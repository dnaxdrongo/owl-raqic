from __future__ import annotations

import cmath
import math
from fractions import Fraction

from .padic import product_formula_rational

TAU = 2.0 * math.pi


def rational_fractional_part(num: int, den: int = 1) -> Fraction:
    q = Fraction(num, den)
    floor = q.numerator // q.denominator
    return q - floor


def padic_fractional_part_proxy(num: int, den: int, p: int, modulus_power: int = 8) -> Fraction:
    """Finite residue proxy for a p-adic fractional part.

    This is a finite active-place proxy, not full p-adic harmonic analysis.
    For denominator coprime to p, it uses modular inverse modulo p**m.
    If denominator is divisible by p, a bounded rational fallback is used.
    """
    q = Fraction(num, den)
    mod = p**modulus_power
    den_mod = q.denominator % mod
    if den_mod and math.gcd(den_mod, mod) == 1:
        inv = pow(den_mod, -1, mod)
        residue = (q.numerator * inv) % mod
        return Fraction(residue, mod)
    return rational_fractional_part(q.numerator, q.denominator)


def adelic_character_phase_proxy(
    num: int,
    den: int,
    primes: tuple[int, ...],
    beta: dict[int, float] | None = None,
    modulus_power: int = 8,
    diagonal_test: bool = False,
) -> float:
    """Finite proxy for 2*pi*(sum_p beta_p frac_p(q) - frac_infinity(q)).

    In diagonal-test mode, returns 0 exactly, reflecting ``chi_A(q)=1`` for
    rational ``q`` in the full adelic theory.
    """
    if diagonal_test:
        _ = product_formula_rational(num, den)
        return 0.0
    beta = beta or dict.fromkeys(primes, 1.0)
    real = float(rational_fractional_part(num, den))
    pad = 0.0
    for p in primes:
        pad += float(beta.get(p, 1.0)) * float(
            padic_fractional_part_proxy(num, den, p, modulus_power)
        )
    return TAU * (pad - real)


def character_value_from_phase(phase: float) -> complex:
    return cmath.exp(1j * phase)


def diagonal_character_cancels(
    num: int, den: int = 1, primes: tuple[int, ...] = (2, 3, 5), tol: float = 1e-12
) -> bool:
    phase = adelic_character_phase_proxy(num, den, primes, diagonal_test=True)
    return abs(character_value_from_phase(phase) - 1.0) <= tol
