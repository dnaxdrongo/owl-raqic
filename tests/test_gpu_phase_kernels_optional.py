import numpy as np

from owl_raqic.gpu.phase_kernels import canonical_phase_numpy


def test_canonical_phase_shape_and_bounds():
    bins = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    phases = canonical_phase_numpy(bins, ("a", "b", "c"), 4, (2, 3, 5))
    assert phases.shape == (2, 4)
    assert np.all(phases >= 0)
    assert np.all(phases < 2 * np.pi + 1e-12)
