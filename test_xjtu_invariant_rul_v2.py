import numpy as np

from evaluate_xjtu_invariant_rul_v2 import invariant_view


def test_invariant_view_removes_absolute_raw_features():
    full = np.arange(2*(4*3+1+16+2), dtype='float32').reshape(2, -1)
    with_usage = invariant_view(full, 3, True)
    without_usage = invariant_view(full, 3, False)
    assert np.array_equal(with_usage, full[:, 3:])
    assert np.array_equal(without_usage, np.column_stack(
        (full[:, 3:-18], full[:, -2:])))
