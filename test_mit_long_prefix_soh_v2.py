"""No-future-input regression tests for MIT 750-second SOH features."""

import unittest
from types import SimpleNamespace

import numpy as np

from evaluate_mit_750s_current_soh_v2 import long_matrix
from extract_mit_750s_soh_v2 import FEATURE_NAMES, prefix_features


class MitLongPrefixSohTests(unittest.TestCase):
    def test_current_prefix_ignores_samples_after_750_seconds(self):
        seconds = np.arange(0, 901, 10, dtype=float)
        minute = seconds/60
        current = np.full(len(seconds), -4.)
        voltage = 3.55-.0007*seconds
        temperature = 30+.001*seconds
        original = prefix_features(minute, current, voltage, temperature)
        changed_i = current.copy()
        changed_v = voltage.copy()
        changed_t = temperature.copy()
        changed_i[seconds > 750] = 0
        changed_v[seconds > 750] = 99
        changed_t[seconds > 750] = 99
        repeated = prefix_features(minute, changed_i, changed_v, changed_t)
        np.testing.assert_allclose(original, repeated, equal_nan=True)
        self.assertEqual(len(original), len(FEATURE_NAMES))

    def test_early_matrix_rows_ignore_later_cycle_signals(self):
        cycle = np.arange(1, 9)
        raw = np.tile(np.linspace(3.55, 3.1, 51), (len(cycle), 1))
        base = SimpleNamespace(cycle=cycle, raw=raw, delta=raw-raw[0],
                               reference_count=1)
        long_raw = np.tile(np.arange(len(FEATURE_NAMES), dtype=float),
                           (len(cycle), 1))
        first = long_matrix(base, long_raw)[:4]
        long_raw[4:] = 999
        np.testing.assert_allclose(first, long_matrix(base, long_raw)[:4],
                                   equal_nan=True)

    def test_discharge_ending_before_750_returns_no_long_features(self):
        seconds = np.arange(0, 901, 10, dtype=float)
        current = np.where(seconds <= 700, -4., 0.)
        row = prefix_features(seconds/60, current,
                              3.55-.0007*seconds, np.full(len(seconds), 30.))
        self.assertTrue(np.isnan(row).all())


if __name__ == '__main__':
    unittest.main()
