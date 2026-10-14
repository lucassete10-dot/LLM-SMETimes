import unittest

import numpy as np

from audit_mit_partial_qv_v2 import discharge_prefix
from audit_mit_750s_multicycle_curve_v2 import power_exact80_crossing
from evaluate_mit_deltaq_stats_v2 import deltaq_statistics
from evaluate_mit_deltaq_v2 import GRID_LENGTH, deltaq_variance
from evaluate_mit_partial_deltaq_v2 import (VOLTAGE_GRID,
                                            capacity_on_voltage_grid,
                                            partial_statistics)
from evaluate_mit_paper_discharge_v2 import capacity_features
from evaluate_mit_750s_multicycle_v2 import GRID as LONG_GRID, feature_vector


class MitDeltaQTests(unittest.TestCase):
    def test_discharge_prefix_uses_no_samples_after_cutoff(self):
        seconds = np.arange(0, 901, 10, dtype=float)
        time_min = seconds/60
        voltage = 3.55-.0007*seconds
        current = np.full(len(seconds), -4.)
        capacity = seconds*4/3600
        first = discharge_prefix(time_min, voltage, current, capacity, 300)
        longer = discharge_prefix(time_min, voltage, current, capacity, 750)
        changed_v = voltage.copy()
        changed_q = capacity.copy()
        changed_v[seconds > 300] = 99
        changed_q[seconds > 300] = 99
        repeated = discharge_prefix(time_min, changed_v, current, changed_q, 300)
        np.testing.assert_allclose(first[0], repeated[0])
        np.testing.assert_allclose(first[1], repeated[1])
        self.assertGreater(len(longer[0]), len(first[0]))
        self.assertLess(longer[0][-1], first[0][-1])

    def test_statistics_use_only_same_grid_difference(self):
        q10 = np.linspace(0, 1, GRID_LENGTH)
        change = np.linspace(-.1, .05, GRID_LENGTH)
        q100 = q10+change
        stats = deltaq_statistics(q10, q100)
        self.assertAlmostEqual(stats[0], np.log10(np.var(change)))
        self.assertAlmostEqual(stats[1], -.1)
        self.assertAlmostEqual(stats[2], -.025)

    def test_invalid_grid_and_nonfinite_values_are_rejected(self):
        q = np.linspace(0, 1, GRID_LENGTH)
        with self.assertRaises(ValueError):
            deltaq_variance(q[:-1], q[:-1])
        invalid = q.copy()
        invalid[5] = np.nan
        with self.assertRaises(ValueError):
            deltaq_variance(q, invalid)
        with self.assertRaises(ValueError):
            deltaq_variance(q, q)

    def test_partial_voltage_grid_is_fixed_and_order_invariant(self):
        voltage = np.linspace(3.55, 3.1, 100)
        capacity = np.linspace(0, .3, 100)
        expected = np.interp(VOLTAGE_GRID, voltage[::-1], capacity[::-1])
        np.testing.assert_allclose(capacity_on_voltage_grid(voltage, capacity),
                                   expected)
        stats = partial_statistics((voltage, capacity),
                                   (voltage, capacity+np.linspace(0, .01, 100)))
        self.assertTrue(np.isfinite(stats).all())
        with self.assertRaises(ValueError):
            capacity_on_voltage_grid(voltage[voltage > 3.2],
                                     capacity[voltage > 3.2])

    def test_early_capacity_features_ignore_every_cycle_after_100(self):
        cycles = np.arange(1, 201)
        capacity = 1.1-.0001*cycles
        original = capacity_features({'cycle': cycles, 'QDischarge': capacity})
        changed = capacity.copy()
        changed[100:] = .3
        altered = capacity_features({'cycle': cycles, 'QDischarge': changed})
        np.testing.assert_allclose(original, altered)
        self.assertAlmostEqual(original[1], -.0001)

    def test_multicycle_partial_features_have_no_full_capacity_input(self):
        q10 = np.linspace(.8, 0, len(LONG_GRID))
        q50 = q10+.01*np.linspace(1, -1, len(LONG_GRID))
        q100 = q10+.02*np.linspace(1, -1, len(LONG_GRID))
        features = feature_vector(q10, q50, q100)
        self.assertEqual(features.shape, (5,))
        self.assertTrue(np.isfinite(features).all())
        self.assertAlmostEqual(features[2], q10[0])

    def test_power_curve_exact80_crossing_may_follow_source_endpoint(self):
        crossing = power_exact80_crossing(1., .85, 100, 500, 2.)
        self.assertAlmostEqual(crossing, 100+400*(.2/.15))
        earlier = power_exact80_crossing(1., .75, 100, 500, 2.)
        self.assertAlmostEqual(earlier, 100+400*np.sqrt(.2/.25))


if __name__ == '__main__':
    unittest.main()
