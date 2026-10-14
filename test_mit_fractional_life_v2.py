import unittest
from types import SimpleNamespace

import numpy as np

from audit_mit_challenger_uncertainty_v2 import stratified_difference
from evaluate_mit_fractional_life_v2 import score_cell


class MitFractionalLifeTests(unittest.TestCase):
    def make_cell(self, endpoint=120):
        return SimpleNamespace(
            cell_id='synthetic', group='batch1',
            cycle=np.arange(1, 111), reference_count=1,
            endpoint_cycle=endpoint,
        )

    def test_origin_and_landmarks_use_only_available_cycles(self):
        cell = self.make_cell()
        direct = np.full(len(cell.cycle), 104.0)
        fraction = np.full(len(cell.cycle), 120.0)
        result = score_cell(cell, direct, fraction)
        self.assertEqual(result['origin_cycle'], 16)
        self.assertAlmostEqual(result['direct_origin_ape_percent'], 0)
        self.assertAlmostEqual(result['fraction_origin_ape_percent'], 0)
        self.assertEqual(result['landmarks']['50']['observed_cycle'], 50)
        self.assertEqual(result['landmarks']['100']['observed_cycle'], 100)
        self.assertNotIn('200', result['landmarks'])

    def test_censored_cell_has_no_event_error(self):
        cell = self.make_cell()
        cell.endpoint_cycle = None
        predictions = np.ones(len(cell.cycle))
        self.assertIsNone(score_cell(cell, predictions, predictions))

    def test_landmark_at_or_after_true_endpoint_is_excluded(self):
        cell = self.make_cell(endpoint=50)
        predictions = np.full(len(cell.cycle), 50.0)
        result = score_cell(cell, predictions, predictions)
        self.assertNotIn('50', result['landmarks'])

    def test_stratified_difference_preserves_cell_weighting(self):
        rows = [{'group': 'a', 'old': 3, 'new': 1},
                {'group': 'a', 'old': 5, 'new': 4},
                {'group': 'b', 'old': 2, 'new': 3}]
        result = stratified_difference(rows, lambda row: row['old'],
                                        lambda row: row['new'])
        self.assertEqual(result['scored_cells'], 3)
        self.assertAlmostEqual(result['improvement_percentage_points'], 2/3)
        self.assertAlmostEqual(result['batch_improvement_percentage_points']['a'], 1.5)
        self.assertAlmostEqual(result['batch_improvement_percentage_points']['b'], -1)


if __name__ == '__main__':
    unittest.main()
