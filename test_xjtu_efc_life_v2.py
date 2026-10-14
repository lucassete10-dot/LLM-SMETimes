import unittest

from evaluate_xjtu_efc_life_v2 import score_one


class EFCLifeTests(unittest.TestCase):
    def test_efc_to_source_cycles_uses_only_early_usage(self):
        row = {'cell_id': 'cell', 'group': 'g', 'origin_cycle': 20,
               'endpoint_cycle': 100, 'early_mean_ah_per_cycle': 1.,
               'early_equivalent_full_cycles': 10.,
               'endpoint_equivalent_full_cycles': 50.}
        result = score_one(row, predicted_efc=50., predicted_cycle=90.)
        self.assertEqual(result['predicted_efc_endpoint_cycle'], 100.)
        self.assertEqual(result['oracle_conversion_ape_percent'], 0.)

    def test_prior_damage_cannot_predict_past_origin(self):
        row = {'cell_id': 'cell', 'group': 'g', 'origin_cycle': 20,
               'endpoint_cycle': 100, 'early_mean_ah_per_cycle': 1.,
               'early_equivalent_full_cycles': 10.,
               'endpoint_equivalent_full_cycles': 50.}
        result = score_one(row, predicted_efc=5., predicted_cycle=10.)
        self.assertEqual(result['predicted_efc_endpoint_cycle'], 20.)
        self.assertEqual(result['predicted_direct_endpoint_cycle'], 20.)


if __name__ == '__main__':
    unittest.main()
