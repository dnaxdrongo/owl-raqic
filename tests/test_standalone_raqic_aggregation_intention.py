import numpy as np

from owl_raqic.math.aggregation import aggregate_records, bottom_up_weights, tissue_over_cell_demo
from owl_raqic.math.checks import check_bottom_up_weights, check_intention_simplex
from owl_raqic.math.intentions import normalize_intention, stable_softmax, update_parent_intention


def test_bottom_up_weights_normalized():
    W = bottom_up_weights(
        np.array([0.5, 0.7]), np.array([0.6, 0.9]), np.array([0.8, 0.8]), np.array([1.0, 0.2])
    )
    assert check_bottom_up_weights(W)["passed"]


def test_tissue_weight_exceeds_cell_weight():
    demo = tissue_over_cell_demo()
    assert demo["passed"]
    assert demo["tissue"] > demo["cell"]


def test_aggregate_records():
    W = np.array([0.25, 0.75])
    X = np.array([[1, 0], [0, 1]], dtype=float)
    assert np.allclose(aggregate_records(W, X), np.array([0.25, 0.75]))


def test_intention_update_simplex():
    prev = np.ones(3) / 3
    agg = np.array([0.1, 1.2, -0.1])
    intention = update_parent_intention(prev, agg)
    assert check_intention_simplex(intention)["passed"]
    assert intention[1] > intention[0]


def test_normalize_zero_intention():
    intention = normalize_intention(np.zeros(4))
    assert np.allclose(intention, np.ones(4) / 4)


def test_masked_softmax_all_false_returns_rest():
    p = stable_softmax(np.array([1.0, 2.0]), mask=np.array([False, False]))
    assert np.allclose(p, [1, 0])
