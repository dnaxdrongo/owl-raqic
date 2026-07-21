import numpy as np

from owl.gpu.kernels.stencil_kernels import raw_toroidal_laplacian4
from owl.gpu.stencil import laplacian_4


def test_raw_toroidal_laplacian_fallback_matches_vectorized_numpy():
    arr = np.arange(25, dtype=np.float64).reshape(5, 5)
    assert np.allclose(raw_toroidal_laplacian4(arr, np), laplacian_4(arr, np, "toroidal"))
