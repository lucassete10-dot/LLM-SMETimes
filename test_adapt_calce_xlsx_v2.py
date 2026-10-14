import unittest
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from audit_calce_label_free_extraction_v2 import extract_folder
from external_test_v2 import read_signal


SAMPLE = Path(r'D:\LLM\CALCE\raw\extracted\CS2_35\CS2_35\CS2_35_10_15_10.xlsx')


class CALCEAdapterTests(unittest.TestCase):
    def test_real_workbook_without_capacity_column(self):
        self.assertTrue(SAMPLE.is_file())
        cycle, raw, detail = extract_folder(SAMPLE)
        self.assertGreater(len(cycle), 1)
        self.assertEqual(raw.shape, (len(cycle), 29))
        self.assertEqual(detail['workbooks'], 1)
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'signals.npz'
            np.savez_compressed(path, cycle=cycle, raw_features=raw)
            checked_cycle, checked_raw = read_signal(path, 'CALCE')
        np.testing.assert_array_equal(checked_cycle, cycle)
        np.testing.assert_array_equal(checked_raw, raw)

    def test_missing_source_rejected(self):
        with self.assertRaises(ValueError):
            extract_folder(Path(r'D:\LLM\CALCE\raw\definitely_missing'))

    def test_cli_writes_signal_only_and_refuses_overwrite(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'signals.npz'
            command = [sys.executable, str(Path(__file__).with_name('adapt_calce_xlsx_v2.py')),
                       'adapt', str(SAMPLE), str(path)]
            first = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(first.returncode, 0, first.stderr)
            with np.load(path) as saved:
                self.assertEqual(set(saved.files), {'cycle', 'raw_features'})
            checked_cycle, checked_raw = read_signal(path, 'CALCE')
            self.assertEqual(checked_raw.shape, (len(checked_cycle), 29))
            second = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn('never overwritten', second.stderr)


if __name__ == '__main__':
    unittest.main()
