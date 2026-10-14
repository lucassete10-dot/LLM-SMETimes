import numpy as np

from evaluate_nasa_boundary_soh_v2 import boundary_adjustment


def test_boundary_adjustment_changes_only_low_end_and_never_raises_prediction():
    tree = np.array([.95, .79, .72, .70])
    linear = np.array([.50, .65, .80, .62])
    result = boundary_adjustment(tree, linear, training_floor=.70)
    assert np.allclose(result[:3], tree[:3])
    assert result[3] < tree[3]
    assert np.all(result <= tree)


def test_boundary_adjustment_is_identity_above_floor():
    tree = np.array([.75, .8, .9])
    assert np.array_equal(boundary_adjustment(tree, tree-.1, .70), tree)
