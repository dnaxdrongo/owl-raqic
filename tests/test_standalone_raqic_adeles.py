from fractions import Fraction

import numpy as np

from owl_raqic.adelic.kernels import finite_adelic_distance, finite_adelic_kernel
from owl_raqic.adelic.padic import is_prime, padic_abs_rational, product_formula_rational, v_p_int
from owl_raqic.adelic.phases import diagonal_character_cancels, padic_fractional_part_proxy
from owl_raqic.adelic.projection import finite_feature_projection, normalize_complex_amplitudes


def test_prime_validation():
    assert is_prime(2)
    assert is_prime(97)
    assert not is_prime(1)
    assert not is_prime(91)


def test_v_p_int():
    assert v_p_int(72, 2) == 3
    assert v_p_int(72, 3) == 2


def test_padic_abs_rational():
    assert padic_abs_rational(18, 35, 2) == Fraction(1, 2)
    assert padic_abs_rational(18, 35, 5) == Fraction(5, 1)


def test_product_formula_exact():
    assert product_formula_rational(18, 35) == Fraction(1, 1)


def test_finite_distance_bounded():
    d = finite_adelic_distance(1.0, {2: 4, 3: 2}, {2: 0, 3: 5}, (2, 3), {2: 0.5, 3: 0.25})
    assert d >= 1.0
    assert np.isfinite(d)


def test_finite_kernel_in_0_1():
    k = finite_adelic_kernel(1.0, {2: 4}, {2: 0}, (2,), {2: 1.0})
    assert 0 < k <= 1


def test_diagonal_character_cancellation():
    assert diagonal_character_cancels(18, 35, (2, 5, 7))


def test_padic_fractional_proxy_bounded():
    x = padic_fractional_part_proxy(1, 3, 2)
    assert 0 <= float(x) < 1


def test_normalize_complex_amplitudes():
    amps = normalize_complex_amplitudes(np.array([1 + 1j, 2, 0]))
    assert np.allclose(np.vdot(amps, amps), 1)


def test_feature_projection_shape():
    x = finite_feature_projection(np.array([0.1, 0.2]), [{2: 1}, {2: 2}], (2,), {2: 0.5})
    assert x.shape == (2,)
    assert np.all(np.isfinite(x))
