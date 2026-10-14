import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np

from adapt_mit_mat_v2 import extract_mat
from external_test_v2 import read_signal


def synthetic_mit_mat(path):
    with h5py.File(path, 'w') as mat:
        batch = mat.create_group('batch')
        summary = mat.create_group('summary0')
        summary.create_dataset('cycle', data=np.arange(1, 7).reshape(1, -1))
        cycles = mat.create_group('cycles0')
        for name in ('I', 'V', 't', 'T'):
            refs = np.empty((6, 1), dtype=h5py.ref_dtype)
            for index in range(6):
                if name == 'I':
                    values = np.full(121, -3.)
                elif name == 'V':
                    values = np.linspace(3.5, 3.0, 121) - index * .001
                elif name == 't':
                    values = np.linspace(0, 6, 121)
                else:
                    values = np.linspace(25, 27, 121)
                dataset = mat.create_dataset(f'{name}{index}', data=values)
                refs[index, 0] = dataset.ref
            cycles.create_dataset(name, data=refs)
        batch.create_dataset('summary', data=np.array([[summary.ref]], dtype=h5py.ref_dtype))
        batch.create_dataset('cycles', data=np.array([[cycles.ref]], dtype=h5py.ref_dtype))


class MITAdapterTests(unittest.TestCase):
    def test_extracts_without_any_capacity_or_life_field(self):
        with TemporaryDirectory() as folder:
            mat_path = Path(folder) / 'synthetic.mat'
            synthetic_mit_mat(mat_path)
            cycle, raw = extract_mat(mat_path, 1, 0)
            self.assertEqual(raw.shape, (6, 51))
            np.testing.assert_array_equal(cycle, np.arange(1, 7))
            signal_path = Path(folder) / 'signal.npz'
            np.savez_compressed(signal_path, cycle=cycle, raw_features=raw)
            checked_cycle, checked_raw = read_signal(signal_path, 'MIT')
            np.testing.assert_array_equal(checked_cycle, cycle)
            np.testing.assert_array_equal(checked_raw, raw)

    def test_bad_batch_or_cell_rejected(self):
        with TemporaryDirectory() as folder:
            mat_path = Path(folder) / 'synthetic.mat'
            synthetic_mit_mat(mat_path)
            with self.assertRaises(ValueError):
                extract_mat(mat_path, 4, 0)
            with self.assertRaises(ValueError):
                extract_mat(mat_path, 1, 1)


if __name__ == '__main__':
    unittest.main()
