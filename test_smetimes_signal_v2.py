import unittest

import numpy as np

from battery_smetimes_signal_v2 import CHANNELS, build_sequence


class SMETimesSignalTests(unittest.TestCase):
    def test_xjtu_charge_and_discharge_slots_are_label_free_and_ordered(self):
        raw = np.zeros((2, 46), dtype='float32')
        raw[:, :5] = np.arange(5)
        raw[:, 5:10] = 2.
        raw[:, 10:15] = 25.
        raw[:, 23:28] = np.arange(5) + 10
        raw[:, 28:33] = -1.
        raw[:, 33:38] = 26.
        sequence, context = build_sequence('XJTU', raw)
        self.assertEqual(sequence.shape, (2, 10, len(CHANNELS)))
        self.assertEqual(context.shape, (2, 4))
        np.testing.assert_array_equal(sequence[0, :5, 0], np.arange(5))
        np.testing.assert_array_equal(sequence[0, 5:, 0], np.arange(5)+10)
        self.assertTrue((sequence[:, :5, 6] == 1).all())
        self.assertTrue((sequence[:, 5:, 7] == 1).all())
        self.assertTrue((sequence[:, :, 1] >= 0).all())
        self.assertTrue((context[:, 2:] == 1).all())

    def test_missing_charge_and_calce_temperature_are_masked(self):
        raw = np.zeros((1, 29), dtype='float32')
        raw[:, :11] = np.arange(11) + 3
        raw[:, 11:22] = -1.
        sequence, context = build_sequence('CALCE', raw)
        self.assertTrue((sequence[:, :5] == 0).all())
        self.assertTrue((sequence[:, :, 5] == 0).all())
        self.assertEqual(context[0, 2], 0.)
        self.assertEqual(context[0, 3], 0.)
        self.assertEqual(context[0, 1], 1000.)

    def test_wrong_schema_and_missing_required_stage_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'invalid raw signal array'):
            build_sequence('MIT', np.zeros((2, 46)))
        raw = np.full((2, 46), np.nan, dtype='float32')
        with self.assertRaisesRegex(ValueError, 'usable voltage and current'):
            build_sequence('XJTU', raw)

    def test_later_diagnostic_cannot_change_earlier_sequence(self):
        raw = np.ones((3, 44), dtype='float32')
        first, _ = build_sequence('NASA', raw)
        raw[2] += 100.
        second, _ = build_sequence('NASA', raw)
        np.testing.assert_array_equal(first[:2], second[:2])


if __name__ == '__main__':
    unittest.main()
