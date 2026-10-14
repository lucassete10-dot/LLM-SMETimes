import unittest

import numpy as np

from scripts.build_canonical_v3 import cell_row, cycle_rows


class CanonicalV3Test(unittest.TestCase):
    def setUp(self):
        self.record = {
            "dataset": "MIT", "cell_id": "synthetic", "group": "batch",
            "cycle_unit": "source cycle number", "rows": 4,
            "reference_capacity_ah": 2.0, "reference_rule": "test reference",
            "reference_established_at_cycle": 1,
            "reference_measurement_count": 1,
            "endpoint_rule": "source criterion", "endpoint_cycle": 4,
            "endpoint_observed": True, "observed_through_cycle": 4,
            "labels_file": "synthetic.npz",
        }
        self.event = {
            "reference_capacity_ah": 2.0,
            "exact80_threshold_ah": 1.6,
            "first_observed_exact80_cycle": 3,
            "first_confirmed_exact80_cycle": 3,
            "last_observed_soh_cycle": 4,
        }
        self.labels = {
            "cycle": np.array([1, 2, 3, 4]),
            "source_cycle": np.array([2, 4, 6, 8]),
            "capacity_ah": np.array([2.0, 1.9, 1.6, 1.5]),
            "soh": np.array([np.nan, .95, .8, .75]),
            "soh_available": np.array([False, True, True, True]),
            "rul_cycles": np.array([np.nan, 2, 1, 0]),
            "rul_available": np.array([False, True, True, True]),
            "capacity_review_flag": np.array([False] * 4),
        }

    def test_distinct_source_and_exact80_events(self):
        rows = cycle_rows(self.record, self.event, self.labels)
        self.assertEqual([r["exact80_rul_cycles"] for r in rows],
                         [None, 1.0, 0.0, None])
        self.assertEqual([r["source_rul_cycles"] for r in rows],
                         [None, 2.0, 1.0, 0.0])
        self.assertEqual(rows[1]["source_cycle"], 4)
        self.assertIsNone(rows[0]["soh_capacity"])

    def test_censored_cell_has_only_lower_bounds(self):
        event = dict(self.event, first_confirmed_exact80_cycle=None,
                     first_observed_exact80_cycle=None)
        record = dict(self.record, endpoint_observed=False, endpoint_cycle=None)
        labels = dict(self.labels, rul_cycles=np.full(4, np.nan),
                      rul_available=np.full(4, False))
        rows = cycle_rows(record, event, labels)
        self.assertTrue(all(r["exact80_rul_cycles"] is None for r in rows))
        self.assertEqual([r["exact80_rul_lower_bound_cycles"] for r in rows],
                         [None, 2.0, 1.0, 0.0])
        self.assertFalse(cell_row(record, event)["exact80_confirmed_event_observed"])

    def test_reference_mismatch_fails(self):
        with self.assertRaisesRegex(ValueError, "reference mismatch"):
            cell_row(self.record, dict(self.event, reference_capacity_ah=1.8))


if __name__ == "__main__":
    unittest.main()
