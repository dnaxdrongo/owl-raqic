from __future__ import annotations

import math
from fractions import Fraction


def is_prime(p: int) -> bool:
    if not isinstance(p, int) or p < 2:
        return False
    if p == 2:
        return True
    if p % 2 == 0:
        return False
    r = int(math.sqrt(p))
    return all(p % k != 0 for k in range(3, r + 1, 2))


def v_p_int(n: int, p: int) -> int:
    if not is_prime(p):
        raise ValueError("p must be prime")
    if n == 0:
        raise ValueError("v_p(0) is infinite; zero is not allowed for finite valuation")
    n = abs(int(n))
    v = 0
    while n % p == 0:
        v += 1
        n //= p
    return v


def v_p_rational(num: int, den: int, p: int) -> int:
    if den == 0:
        raise ZeroDivisionError("denominator cannot be zero")
    if num == 0:
        raise ValueError("v_p(0) is infinite; zero is not allowed for finite valuation")
    return v_p_int(num, p) - v_p_int(den, p)


def padic_abs_rational(num: int, den: int, p: int) -> Fraction:
    v = v_p_rational(num, den, p)
    if v >= 0:
        return Fraction(1, p**v)
    return Fraction(p ** (-v), 1)


def padic_distance_int(a: int, b: int, p: int) -> float:
    d = int(a) - int(b)
    if d == 0:
        return 0.0
    return float(padic_abs_rational(d, 1, p))


def bounded_padic_distance(a: int, b: int, p: int) -> float:
    d = padic_distance_int(a, b, p)
    return d / (1.0 + d)


def prime_factors(n: int) -> set[int]:
    n = abs(int(n))
    out: set[int] = set()
    if n < 2:
        return out
    d = 2
    while d * d <= n:
        while n % d == 0:
            out.add(d)
            n //= d
        d += 1 if d == 2 else 2
    if n > 1:
        out.add(n)
    return out


def product_formula_rational(num: int, den: int = 1) -> Fraction:
    if num == 0:
        raise ValueError("product formula is for nonzero rationals")
    q = Fraction(num, den)
    arch = Fraction(abs(q.numerator), abs(q.denominator))
    prod = arch
    for p in prime_factors(q.numerator) | prime_factors(q.denominator):
        prod *= padic_abs_rational(q.numerator, q.denominator, p)
    return prod
