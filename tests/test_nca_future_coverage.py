import unittest
from scripts.audit_nca_future_coverage import coverage


class CoverageTests(unittest.TestCase):
    def test_short_forecast(self):
        result = coverage(range(101, 109))
        self.assertEqual(result['target_count'], 8)
        self.assertEqual(result['maximum_horizon_from_origin'], 8)

    def test_gaps_not_renumbered(self):
        result = coverage([103, 105, 200])
        self.assertEqual(result['target_count'], 3)
        self.assertEqual(result['maximum_horizon_from_origin'], 100)
        self.assertEqual(result['observed_source_span'], 98)

    def test_invalid_cycles_rejected(self):
        for x in ([], [100, 101], [102, 101], [101, 101]):
            with self.assertRaises(ValueError):
                coverage(x)
