import unittest

import numpy as np

from audit_xjtu_operating_throughput_v2 import (discharge_ah,
                                                 fill_missing_causally)


class OperatingThroughputTests(unittest.TestCase):
    def test_discharge_integral_uses_stage_time_minutes(self):
        entry = {'relative_time_min': np.array([0., 1., 0., 1., 0., 60.]),
                 'current_A': np.array([1., 1., 0., 0., -2., -2.])}
        self.assertAlmostEqual(discharge_ah(entry), 2.)

    def test_missing_trace_uses_only_prior_cycles(self):
        values, count = fill_missing_causally([1., 1.5, np.nan, 2.])
        self.assertEqual(count, 1)
        self.assertEqual(values[2], 1.25)

    def test_nonphysical_trace_is_rejected(self):
        with self.assertRaises(ValueError):
            fill_missing_causally([4.])


if __name__ == '__main__':
    unittest.main()
