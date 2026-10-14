import unittest

import numpy as np

from battery_v2_features import load_cell, load_manifest
from evaluate_xjtu_300_rul_assisted_v2 import assisted_matrix
from train_rul_v2 import assisted_features


class XJTU300AssistedRULTests(unittest.TestCase):
    def test_feature_formula_matches_original_assisted_route(self):
        record = next(row for row in load_manifest()['cells'] if row['dataset'] == 'XJTU')
        cell = load_cell(record)
        predicted_soh = np.linspace(1., .8, len(cell.cycle), dtype='float32')
        old = assisted_features(cell, predicted_soh)
        candidate = assisted_matrix(cell.matrix('signal_history'), cell.cycle,
                                    predicted_soh)
        np.testing.assert_array_equal(candidate, old)

    def test_future_soh_cannot_change_earlier_rul_features(self):
        record = next(row for row in load_manifest()['cells'] if row['dataset'] == 'XJTU')
        cell = load_cell(record)
        signal = cell.matrix('signal_history')
        original = np.linspace(1., .8, len(cell.cycle), dtype='float32')
        changed = original.copy()
        changed[30:] += .1
        first = assisted_matrix(signal, cell.cycle, original)
        second = assisted_matrix(signal, cell.cycle, changed)
        np.testing.assert_array_equal(first[:30], second[:30])


if __name__ == '__main__':
    unittest.main()
