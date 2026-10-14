import unittest

from audit_capacity_reference_v2 import audit


class ReferenceSensitivityTests(unittest.TestCase):
    def test_saved_references_soh_and_derived_endpoints(self):
        result = audit()
        self.assertEqual({name: row['cells'] for name, row in result['domains'].items()},
                         {'NASA': 4, 'CALCE': 4, 'MIT': 134, 'XJTU': 55})
        self.assertEqual(len(result['cells']), 197)
        self.assertTrue(all(row['first_reference_endpoint_cycle'] is not None
                            for row in result['cells']
                            if row['dataset'] in ('NASA', 'CALCE')))


if __name__ == '__main__':
    unittest.main()
