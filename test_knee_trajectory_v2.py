import numpy as np

from evaluate_knee_trajectory_v2 import knee_curve


def test_knee_curve_is_bounded_and_crosses_declared_eol():
    curve = knee_curve(.97, .8, 100, np.arange(0, 151))
    assert np.isclose(curve[0], .97)
    assert np.isclose(curve[100], .8)
    assert np.all(np.diff(curve) <= 1e-12)
    assert np.all((curve >= 0) & (curve <= .97))
