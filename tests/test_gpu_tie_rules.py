from owl.gpu.tie_rules import DEFAULT_TIE_RULES, deterministic_uniform01


def test_tie_rules_death_and_viability():
    import numpy as np

    xp = np
    assert bool(DEFAULT_TIE_RULES.death_mask(xp.array([0.0]), 0.0, xp)[0])
    assert bool(DEFAULT_TIE_RULES.viable_mask(xp.array([1.0]), 1.0, xp)[0])


def test_deterministic_uniform_repeatable():
    a = deterministic_uniform01(1, 2, 3, 4)
    b = deterministic_uniform01(1, 2, 3, 4)
    c = deterministic_uniform01(1, 2, 4, 4)
    assert a == b
    assert 0.0 <= a < 1.0
    assert a != c
