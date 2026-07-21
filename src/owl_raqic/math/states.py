from __future__ import annotations

from typing import cast

import numpy as np


def ket0(dim: int) -> np.ndarray:
    if dim < 1:
        raise ValueError("dim must be positive")
    v = np.zeros(dim, dtype=complex)
    v[0] = 1.0
    return v


def density_from_state(psi: np.ndarray) -> np.ndarray:
    psi = np.asarray(psi, dtype=complex)
    return np.outer(psi, psi.conjugate())


def maximally_mixed(dim: int) -> np.ndarray:
    return np.eye(dim, dtype=complex) / dim


def probabilities_from_state(psi: np.ndarray) -> np.ndarray:
    p = np.abs(np.asarray(psi, dtype=complex)) ** 2
    return cast(np.ndarray, p / p.sum())


def pad_to_power_of_two(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=complex)
    n = 1 << (len(vec) - 1).bit_length()
    out = np.zeros(n, dtype=complex)
    out[: len(vec)] = vec
    norm = np.linalg.norm(out)
    if norm == 0:
        raise ValueError("cannot pad zero vector")
    return cast(np.ndarray, out / norm)
