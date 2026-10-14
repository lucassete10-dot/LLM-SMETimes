import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.io import savemat

from adapt_nasa_mat_v2 import extract_mat, extract_records
from external_test_v2 import read_signal


def fake_discharge(with_capacity=True):
    time = np.linspace(0, 1200, 121)
    data = {'Time': time, 'Voltage_measured': np.linspace(4.1, 3.3, 121),
            'Current_measured': np.full(121, -2.),
            'Temperature_measured': np.linspace(25, 30, 121)}
    if with_capacity:
        data['Capacity'] = 1.5
    return {'type': 'discharge', 'data': data}


class NASAAdapterTests(unittest.TestCase):
    def test_extracts_without_capacity_field(self):
        records = [fake_discharge(False), {'type': 'charge', 'data': {}},
                   fake_discharge(False)]
        cycle, raw = extract_records(records)
        np.testing.assert_array_equal(cycle, [1, 2])
        self.assertEqual(raw.shape, (2, 44))
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'signals.npz'
            np.savez_compressed(path, cycle=cycle, raw_features=raw)
            checked_cycle, checked_raw = read_signal(path, 'NASA')
        np.testing.assert_array_equal(checked_cycle, cycle)
        np.testing.assert_array_equal(checked_raw, raw)

    def test_mat_variable_and_short_discharge(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'cell.mat'
            savemat(path, {'TESTCELL': {'cycle': [fake_discharge(False),
                                               fake_discharge(False)]}})
            cycle, raw = extract_mat(path, 'TESTCELL')
            self.assertEqual(raw.shape, (2, 44))
            with self.assertRaises(ValueError):
                extract_mat(path, 'MISSING')
        short = fake_discharge(False)
        short['data']['Time'] = np.linspace(0, 500, 121)
        with self.assertRaises(ValueError):
            extract_records([short, fake_discharge(False)])


if __name__ == '__main__':
    unittest.main()
