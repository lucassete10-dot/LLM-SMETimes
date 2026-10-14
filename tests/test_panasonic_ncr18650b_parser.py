import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from parsers.panasonic_ncr18650b import (aging_exposure_by_test,
                                         canonical_diagnostic_candidates,
                                         inspect_cell, load_test,
                                         rpt_1c_capacity_candidates)
from scripts.audit_panasonic_ncr18650b import threshold_interval


INFO_DTYPE = np.dtype([
    ("idcell", "S5"), ("idtest", "<u2"), ("idexp", "u1"),
    ("expname", "S6"), ("startdatetime", "<f8"),
    ("schedulefilename", "S45"),
])


def make_file(path, *, duplicate=False, bad_time=False, padded=False):
    with h5py.File(path, "w") as file:
        rows = [(b"2_097", 4802, 1, b"RTP25", 1.0, b"RTP")]
        if duplicate:
            rows.append(rows[0])
        file.create_dataset("info", data=np.array(rows, dtype=INFO_DTYPE))
        tests = file.create_group("tests")
        group = tests.create_group("test_04802" if padded else "test_4802")
        values = {
            "test_time": [1.0, 2.0, 1.5 if bad_time else 3.0],
            "voltage": [4.1, 3.8, 3.5],
            "current": [-1.0, -1.0, 0.0],
            "temperature": [25.0, -273.16, 26.0],
            "chargecap": [0.0, 0.0, 0.0],
            "dischargecap": [0.0, 0.2, 0.4],
            "stepindex": [1, 1, 2],
        }
        for name, array in values.items():
            group.create_dataset(name, data=array)


class PanasonicParserTest(unittest.TestCase):
    def test_aging_exposure_excludes_rpt_and_future_load(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "exposure.h5"
            with h5py.File(path, "w") as file:
                file.create_dataset("info", data=np.array([
                    (b"2_101", 10, 1, b"RTP25", 1., b"RTP"),
                    (b"2_101", 11, 2, b"L214", 2., b"L214"),
                    (b"2_101", 12, 3, b"RTP25", 3., b"RTP"),
                ], dtype=INFO_DTYPE))
                tests = file.create_group("tests")
                for test_id, current in ((10, -5.), (11, -1.), (12, -5.)):
                    group = tests.create_group(f"test_{test_id}")
                    group.create_dataset("test_time", data=[0., 3600.])
                    group.create_dataset("current", data=[current, current])
            exposure = aging_exposure_by_test(path, "L214")
            self.assertEqual(exposure[10]["cumulative_aging_discharge_ah"], 0)
            self.assertEqual(exposure[11]["cumulative_aging_discharge_ah"], 0)
            self.assertAlmostEqual(exposure[12]["cumulative_aging_discharge_ah"], 1.)
            self.assertAlmostEqual(exposure[12]["equivalent_full_cycles"], 1 / 3.25)

    def test_eol80_is_an_interval_not_an_exact_cycle(self):
        rows = [{"status": "two_repeat_candidate", "test_id": i,
                 "soh_candidate": value, "elapsed_source_days_candidate": day}
                for i, value, day in [(1, 1.0, 0), (2, .85, 20),
                                       (3, .78, 41), (4, .75, 62)]]
        interval = threshold_interval(rows)
        self.assertEqual(interval["status"], "interval_observed_not_exact_cycle")
        self.assertEqual((interval["last_above_test_id"],
                          interval["first_below_test_id"]), (2, 3))
        self.assertNotIn("eol_cycle", interval)

    def test_inspection_does_not_fabricate_soh_or_rul(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "2_097.h5"
            make_file(path)
            result = inspect_cell(path)
            self.assertEqual(result["cell_id"], "2_097")
            self.assertEqual(result["test_count"], 1)
            self.assertEqual(result["samples"], 3)
            self.assertEqual(result["invalid_temperature_samples"], 1)
            self.assertEqual(result["invalid_samples_by_signal"]["voltage"], 0)
            self.assertEqual(result["soh_label_status"], "not_derived_protocol_audit_pending")
            self.assertEqual(result["rul_label_status"], "not_derived_endpoint_audit_pending")
            self.assertNotIn("soh", result["tests"][0])
            self.assertTrue(np.isnan(load_test(path, 4802)["temperature"][1]))
            self.assertEqual(rpt_1c_capacity_candidates(path)["rpt_rows"][0]["status"],
                             "incomplete_rpt")

    def test_rpt_two_repeats_are_capacity_candidates_only(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "rpt.h5"
            make_file(path)
            step = np.array([28] * 12 + [29] + [28] * 12)
            q = np.r_[np.linspace(0, 3.0, 12), 0, np.linspace(0, 2.9, 12)]
            voltage = np.r_[np.linspace(4.05, 2.5, 12), 2.6,
                            np.linspace(4.05, 2.5, 12)]
            with h5py.File(path, "a") as file:
                group = file["tests/test_4802"]
                values = {"stepindex": step, "dischargecap": q,
                          "chargecap": np.zeros(25), "voltage": voltage,
                          "current": np.r_[[-3.35] * 12, 0.0, [-3.35] * 12],
                          "temperature": np.full(25, 25.0),
                          "test_time": np.arange(25, dtype=float)}
                for name, array in values.items():
                    del group[name]
                    group.create_dataset(name, data=array)
            result = rpt_1c_capacity_candidates(path)
            row = result["rpt_rows"][0]
            self.assertEqual(row["status"], "two_repeat_candidate")
            self.assertAlmostEqual(row["capacity_median_source"], 2.95)
            self.assertAlmostEqual(row["soh_candidate"], 1.0)
            self.assertFalse(row["capacity_review_flag"])
            self.assertFalse(result["training_eligible"])
            rows = canonical_diagnostic_candidates(path, "L207", {
                "aging_mode": "cyclic", "chamber_temperature_c": 20})
            self.assertEqual(rows[0]["label_status"], "provisional_not_training_approved")
            self.assertIsNone(rows[0]["cycle_index"])
            self.assertIsNone(rows[0]["rul_cycles"])

    def test_duplicate_test_id_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "duplicate.h5"
            make_file(path, duplicate=True)
            with self.assertRaisesRegex(ValueError, "duplicate test IDs"):
                inspect_cell(path)

    def test_leading_zero_group_name(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "padded.h5"
            make_file(path, padded=True)
            self.assertEqual(inspect_cell(path)["test_count"], 1)
            self.assertEqual(len(load_test(path, 4802)["voltage"]), 3)

    def test_time_reversal_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad_time.h5"
            make_file(path, bad_time=True)
            with self.assertRaisesRegex(ValueError, "invalid time order"):
                inspect_cell(path)
            with self.assertRaisesRegex(ValueError, "increase strictly"):
                load_test(path, 4802)


if __name__ == "__main__":
    unittest.main()
