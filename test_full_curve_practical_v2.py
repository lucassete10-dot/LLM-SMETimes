import numpy as np

from evaluate_full_curve_practical_v2 import project


def test_curve_projection_starts_at_current_soh_and_is_nonincreasing():
    horizons = np.array([0, 10, 20, 50])
    curves = project(.95, .8, 100, 80, horizons)
    for curve in curves.values():
        assert np.isclose(curve[0], .95)
        assert np.all(np.diff(curve) <= 1e-9)


def test_fixed_half_is_between_local_and_direct():
    curves = project(.95, .8, 100, 80, np.array([0, 10, 50]))
    average = .5*(curves['local_linear']+curves['direct_linear'])
    assert np.allclose(curves['fixed_half'], average)


def test_bounded_half_local_curve_stays_physical_for_bad_rate():
    curves = project(.91, .8, 10, 2, np.array([0, 10, 100]))
    candidate = curves['bounded_half_local_rate']
    assert np.all(candidate >= 0)
    assert np.all(candidate <= .91)
    assert np.all(np.diff(candidate) <= 0)
