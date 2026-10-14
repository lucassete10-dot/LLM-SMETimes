import unittest

from evaluate_xjtu_rul_target_nested_v2 import choose_target


class XJTURULTargetNestedTests(unittest.TestCase):
    def test_inner_fold_selection_is_equal_group_and_ties_keep_log(self):
        folds = [{'log1p_target': {'cell_macro_mae_cycles': 10.},
                  'raw_cycle_target': {'cell_macro_mae_cycles': 20.}},
                 {'log1p_target': {'cell_macro_mae_cycles': 30.},
                  'raw_cycle_target': {'cell_macro_mae_cycles': 20.}}]
        selected, means = choose_target(folds)
        self.assertEqual(means, {'log1p_target': 20., 'raw_cycle_target': 20.})
        self.assertEqual(selected, 'log1p_target')
        folds[1]['raw_cycle_target']['cell_macro_mae_cycles'] = 18.
        self.assertEqual(choose_target(folds)[0], 'raw_cycle_target')


if __name__ == '__main__':
    unittest.main()
