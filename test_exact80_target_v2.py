"""Regression checks for exact per-cell 80%-SOH target auditing."""

import unittest
from types import SimpleNamespace

import numpy as np

from audit_exact80_target_v2 import audit_cell, summarize
from audit_mit_exact80_rul_v2 import score
from audit_exact80_full_curve_v2 import (inferred_exact80_cycle,
                                         source_anchored_linear_curve)
from evaluate_mit_exact80_threshold_transfer_v2 import transferred_endpoint


def make_cell(soh, endpoint=4):
    soh = np.asarray(soh, dtype=float)
    return SimpleNamespace(
        cell_id='test-cell', group='test-group',
        reference_capacity_ah=2.0,
        cycle=np.arange(1, len(soh) + 1), soh=soh,
        soh_available=np.ones(len(soh), dtype=bool),
        endpoint_cycle=endpoint,
    )


class Exact80TargetAuditTests(unittest.TestCase):
    def test_source_endpoint_is_not_an_exact80_event(self):
        row = audit_cell(make_cell([.99, .91, .86, .85, .84]))
        self.assertEqual(row['exact80_threshold_ah'], 1.6)
        self.assertIsNone(row['first_observed_exact80_cycle'])
        self.assertIsNone(row['first_confirmed_exact80_cycle'])
        self.assertEqual(summarize([row])['right_censored_for_confirmed_event'], 1)

    def test_isolated_crossing_is_not_confirmed(self):
        row = audit_cell(make_cell([.9, .79, .85, .79, .79, .79]))
        self.assertEqual(row['first_observed_exact80_cycle'], 2)
        self.assertEqual(row['first_confirmed_exact80_cycle'], 4)
        self.assertEqual(row['source_minus_confirmed_cycles'], 0)

    def test_final_crossing_needs_confirmation(self):
        row = audit_cell(make_cell([.9, .85, .84, .79]))
        self.assertEqual(row['first_observed_exact80_cycle'], 4)
        self.assertIsNone(row['first_confirmed_exact80_cycle'])

    def test_rul_error_is_measured_from_forecast_origin(self):
        self.assertAlmostEqual(score(500, 460, origin=100), 10.0)
        self.assertIsNone(score(100, 120, origin=100))

    def test_physical_threshold_transfer_uses_reference_capacity(self):
        # 0.88 Ah source endpoint, 0.84 Ah exact80 target, 0.002 Ah/cycle.
        self.assertAlmostEqual(transferred_endpoint(400, 1.05, .002), 420)

    def test_source_curve_exact80_crossing_uses_own_threshold(self):
        # A source SOH threshold of 0.85 at 100 future cycles is not the
        # target: continuing the same curve to 0.80 takes 133.3 cycles.
        predicted = inferred_exact80_cycle(1.0, .85, 20, 100)
        self.assertAlmostEqual(predicted, 20+100*(.2/.15))
        curve = source_anchored_linear_curve(1.0, .85, 20, 100,
                                             np.array([120, predicted]))
        np.testing.assert_allclose(curve, [.85, .8])


if __name__ == '__main__':
    unittest.main()
