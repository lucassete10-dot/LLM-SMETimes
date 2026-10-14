import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from adapt_xjtu_mat_v2 import extract_from_cycles
from external_test_v2 import read_signal


def fake_cycle(description, shift=0.):
    time = np.tile(np.arange(0, 122, dtype='float64') / 60, 3)
    voltage = np.r_[np.linspace(3.5, 3.7, 122), np.full(122, 3.7),
                    np.linspace(3.7, 3.4, 122)] + shift
    current = np.r_[np.full(122, 2.), np.zeros(122), np.full(122, -1.)]
    temperature = np.full(len(time), 25.)
    return {'description': description, 'relative_time_min': time,
            'voltage_V': voltage, 'current_A': current, 'temperature_C': temperature}


class XJTUAdapterTests(unittest.TestCase):
    def test_label_free_selection_and_shape(self):
        cycles = [fake_cycle('[Test Capacity] 1C discharge', shift=i * .01)
                  for i in range(7)]
        cycles.insert(2, fake_cycle('random partial discharge'))
        number, raw = extract_from_cycles(cycles, 4)
        np.testing.assert_array_equal(number, [1, 2, 4, 5, 6, 7, 8])
        self.assertEqual(raw.shape, (7, 46))
        self.assertTrue((raw[:, 23] > raw[:, 27]).all())  # discharge voltage falls
        self.assertTrue((raw[:, 28:33] < 0).all())  # discharge current is negative

    def test_protocol_and_minimum_reference(self):
        cycles = [fake_cycle('1C discharge') for _ in range(6)]
        number, raw = extract_from_cycles(cycles, 1)
        self.assertEqual(len(number), 6)
        self.assertEqual(raw.shape[1], 46)
        with self.assertRaises(ValueError):
            extract_from_cycles(cycles[:5], 1)
        with self.assertRaises(ValueError):
            extract_from_cycles(cycles, 7)

    def test_output_contract_matches_external_signal_reader(self):
        cycle, raw = extract_from_cycles([fake_cycle('[Test Capacity] 1C discharge')
                                          for _ in range(6)], 4)
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'signals.npz'
            np.savez_compressed(path, cycle=cycle, raw_features=raw)
            loaded_cycle, loaded_raw = read_signal(path, 'XJTU')
        np.testing.assert_array_equal(loaded_cycle, cycle)
        np.testing.assert_array_equal(loaded_raw, raw)


if __name__ == '__main__':
    unittest.main()
