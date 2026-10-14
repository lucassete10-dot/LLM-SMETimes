import unittest
import numpy as np
from scripts.audit_tail_numeric_gates import metric, aggregate, gates


class NumericGatesTest(unittest.TestCase):
    def test_flat_prediction_fails_despite_small_full_mape(self):
        rows = [dict(cell_id='a', seed=s, **metric(np.arange(101), np.linspace(1, .94, 101), np.ones(101))) for s in range(5)]
        summary, cells = aggregate(rows)
        self.assertLess(summary['mape_percent']['mean'], 6)
        self.assertFalse(cells[0]['joint_pass'])
        self.assertTrue(cells[0]['severe_flat_failure'])
        self.assertFalse(gates(summary, summary)['drop_mean'])

    def test_nasa_four_cells_requires_four_passes(self):
        rows = []
        for i in range(4):
            y = np.linspace(1, .8, 10); p = y if i < 3 else np.ones(10)
            rows += [dict(cell_id=str(i), seed=s, **metric(np.arange(10), y, p)) for s in range(5)]
        summary, _ = aggregate(rows)
        self.assertEqual(summary['joint_pass_fraction'], .75)
        self.assertFalse(gates(summary, summary)['joint_fraction'])


if __name__ == '__main__': unittest.main()
