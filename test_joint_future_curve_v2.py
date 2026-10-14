import numpy as np

from evaluate_joint_future_curve_v2 import bounded_curve


def test_bounded_curve_repairs_negative_and_upward_forecasts():
    prediction = bounded_curve(.95, np.array([.1, .01, 1., -.5]),
                               np.array([1, 2, 3, 4]))
    assert prediction[0] <= .95
    assert np.all(prediction >= 0)
    assert np.all(np.diff(prediction) <= 0)


def test_bounded_curve_rejects_nonincreasing_horizons():
    try:
        bounded_curve(.95, np.array([.1, .2]), np.array([2, 1]))
    except ValueError:
        return
    raise AssertionError('nonincreasing horizons were accepted')
