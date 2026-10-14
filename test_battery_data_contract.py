"""Focused checks for the label rules used by the four-dataset build."""

import unittest

import numpy as np

from battery_data_contract import LabelSpec, build_labels, confirmed_threshold_event


class LabelContractTests(unittest.TestCase):
    def test_first_five_xjtu_measurements_are_calibration_only(self):
        cycle = np.array([7, 13, 19, 25, 31, 37, 43])
        capacity = np.array([1.91, 1.92, 1.93, 1.94, 1.95, 1.90, 1.88])
        arrays, meta = build_labels(cycle, capacity, cycle,
                                    LabelSpec(5, 'median first five', 'source life', 43))
        self.assertTrue(np.isnan(arrays['soh'][:5]).all())
        self.assertTrue(np.isnan(arrays['rul_cycles'][:5]).all())
        self.assertEqual(meta['reference_established_at_cycle'], 31)
        self.assertAlmostEqual(arrays['soh'][5], 1.90 / 1.93)
        self.assertEqual(arrays['rul_cycles'][5], 6)
        self.assertEqual(meta['soh_examples'], 2)

    def test_censoring_and_post_endpoint_rows(self):
        cycle = [1, 2, 3, 4, 5]
        capacity = [2, 1.95, 1.9, 1.85, 1.8]
        censored, meta = build_labels(cycle, capacity, cycle,
                                      LabelSpec(1, 'first', 'not observed'))
        self.assertFalse(meta['endpoint_observed'])
        self.assertEqual(meta['censor_cycle'], 5)
        self.assertTrue(np.isnan(censored['rul_cycles']).all())
        observed, meta = build_labels(cycle, capacity, cycle,
                                      LabelSpec(1, 'first', 'event', 3))
        self.assertTrue(np.array_equal(observed['rul_available'], [False, True, True, False, False]))
        self.assertTrue(np.array_equal(observed['rul_cycles'][1:3], [1, 0]))
        self.assertEqual(meta['post_endpoint_rows_excluded_from_rul'], 2)

    def test_isolated_low_capacity_is_not_life_endpoint(self):
        cycle = [1, 2, 3, 4, 5, 6, 7, 8]
        q = [1., .79, .94, .89, .78, .77, .76, .75]
        self.assertEqual(confirmed_threshold_event(cycle, q, .8), 5)
        self.assertIsNone(confirmed_threshold_event(cycle[:4], q[:4], .8))

    def test_reject_unsorted_or_invalid_measurements(self):
        spec = LabelSpec(1, 'first', 'unknown')
        with self.assertRaises(ValueError):
            build_labels([1, 1], [2., 1.9], [1, 2], spec)
        with self.assertRaises(ValueError):
            build_labels([1, 2], [2., float('nan')], [1, 2], spec)


if __name__ == '__main__':
    unittest.main()
