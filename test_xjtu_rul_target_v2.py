import unittest

from evaluate_xjtu_rul_target_v2 import score_summary


class XJTURULTargetTests(unittest.TestCase):
    def test_summary_preserves_fold_and_cell_units(self):
        folds = [{'raw_cycle_target': {
            'cell_macro_mae_cycles': 10., 'cell_macro_signed_bias_cycles': -2.,
            'phase_macro_mae_cycles': {'early': 12., 'middle': 8., 'late': 3.},
            'by_cell': [{'mae_cycles': 7.}, {'mae_cycles': 13.}]}},
            {'raw_cycle_target': {
                'cell_macro_mae_cycles': 20., 'cell_macro_signed_bias_cycles': 4.,
                'phase_macro_mae_cycles': {'early': 25., 'middle': 15., 'late': 4.},
                'by_cell': [{'mae_cycles': 20.}]}}]
        report = score_summary(folds, 'raw_cycle_target')
        self.assertEqual(report['equal_fold_macro_mae_cycles'], 15.)
        self.assertEqual(report['worst_cell_mae_cycles'], 20.)
        self.assertEqual(report['phase_equal_fold_macro_mae_cycles']['early'], 18.5)


if __name__ == '__main__':
    unittest.main()
