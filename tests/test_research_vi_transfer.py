import copy
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from battery_smetimes.research_transfer_vi120 import (
    DIRECTIONS, SCHEMA, VI_FEATURE_NAMES, extract_mit_vi, extract_xjtu_vi, strict_vi_prefix,
)
from battery_smetimes.research_transfer_signals import RESEARCH_MODE, make_transfer_plans
from battery_smetimes.research_transfer_baselines import transfer_prefix, transfer_input_schema
from tests import test_research_transfer


class StrictVITransferTests(unittest.TestCase):
    def test_strict_window_ignores_later_values_and_never_brackets_endpoint(self):
        t = np.array([0., 30, 60, 90, 119, 121, 300])
        v, i = 4 - t / 1000, np.full(len(t), -2.)
        expected = strict_vi_prefix(t, v, i, 1.)
        v[t > 120], i[t > 120] = np.nan, np.inf
        np.testing.assert_array_equal(expected, strict_vi_prefix(t, v, i, 1.))
        self.assertEqual(expected.shape, (11,))
        self.assertAlmostEqual(float(expected[4]), -.119, places=6)
        np.testing.assert_array_equal(expected[5:10], np.full(5, 2.))

    def test_no_temperature_or_measured_capacity_normalization(self):
        value = strict_vi_prefix([0, 60, 120], [4, 3.9, 3.8], [-4, -4, -4], 2.)
        np.testing.assert_array_equal(value[5:10], np.full(5, 2.))
        self.assertFalse(any("temperature" in name for name in VI_FEATURE_NAMES))
        for capacity in (None, 0, -1, np.nan, True):
            with self.assertRaisesRegex(ValueError, "rated capacity"):
                strict_vi_prefix([0, 60, 120], [4, 3.9, 3.8], [-4]*3, capacity)

    def test_short_nonfinite_reversed_and_mismatched_histories_rejected(self):
        for t, v, i in (([0, 60, 113], [4]*3, [-2]*3),
                        ([0, 60, 120], [4, np.nan, 4], [-2]*3),
                        ([0, 80, 60, 120], [4]*4, [-2]*4),
                        ([0, 60, 120], [4]*2, [-2]*3)):
            with self.assertRaises(ValueError):
                strict_vi_prefix(t, v, i, 2.)

    def test_one_normal_5s_interval_is_held_without_reading_next_sample(self):
        output = strict_vi_prefix([0, 60, 115, 120.1], [4, 3.9, 3.8, np.nan], [-2, -2, -2, np.inf], 2.)
        self.assertAlmostEqual(float(output[4]), -.2, places=6)

    def test_duplicates_and_tiny_clock_roundoff_are_explicitly_tolerated(self):
        out = strict_vi_prefix([0, 30, 30 - 1e-8, 60, 120], [4, 3.9, 3.9, 3.8, 3.7], [-2]*5, 2.)
        self.assertTrue(np.isfinite(out).all())
        with self.assertRaisesRegex(ValueError, "distinct"):
            strict_vi_prefix([0, 0, 120], [4]*3, [-2]*3, 2.)

    def xjtu(self):
        t = np.r_[[0, 1], [0, 1], np.arange(121)/60., [0, 1]]
        return {"relative_time_min": t, "voltage_V": np.r_[[99]*4, 4 - np.arange(121)/1000., [88]*2],
                "current_A": np.r_[[99.]*4, [-4.]*121, [88.]*2]}

    def test_xjtu_discharge_not_charge_and_no_temperature_field_required(self):
        cycle = self.xjtu()
        rows, rejected = extract_xjtu_vi([1], [cycle], 2.)
        self.assertEqual(rejected, [])
        self.assertEqual(rows[0][0], 1)
        self.assertAlmostEqual(float(rows[0][1][-1]), 4.)
        np.testing.assert_array_equal(rows[0][1][5:10], [2]*5)
        cycle["current_A"][:4] = np.nan
        np.testing.assert_array_equal(rows[0][1], extract_xjtu_vi([1], [cycle], 2.)[0][0][1])

    def test_xjtu_rejection_record_keeps_original_cycle_and_never_reads_future_cycle(self):
        broken = self.xjtu()
        broken["voltage_V"][5] = np.nan
        rows, rejected = extract_xjtu_vi([1, 2], [broken, self.xjtu(), object()], 2.)
        self.assertEqual([n for n, _ in rows], [2])
        self.assertEqual(rejected[0]["cycle"], 1)
        with self.assertRaisesRegex(ValueError, "observed raw source"):
            extract_xjtu_vi([101], [self.xjtu()] * 101, 2.)

    def test_mit_raw_reader_needs_only_cycle_time_voltage_current(self):
        with tempfile.TemporaryDirectory() as tmp:
            with h5py.File(Path(tmp)/"minimal.mat", "w") as mat:
                batch, summary, cycles = [mat.create_group(name) for name in ("batch", "summary", "cycles")]
                summary.create_dataset("cycle", data=[1])
                for name, group in (("summary", summary), ("cycles", cycles)):
                    d = batch.create_dataset(name, (1, 1), dtype=h5py.ref_dtype)
                    d[0, 0] = group.ref
                t = np.arange(0., 301., 1.)
                for name, values in (("t", t/60.), ("V", 4-t/1000.), ("I", np.full(len(t), -4.4))):
                    raw = mat.create_dataset("raw_"+name, data=values)
                    d = cycles.create_dataset(name, (1, 1), dtype=h5py.ref_dtype)
                    d[0, 0] = raw.ref
                rows, rejected = extract_mit_vi("b1c0", [1], mat, 1.1)
                self.assertEqual(rejected, [])
                np.testing.assert_array_equal(rows[0][1][5:10], [4]*5)
                with self.assertRaisesRegex(ValueError, "missing/duplicated"):
                    extract_mit_vi("b1c0", [2], mat, 1.1)

    def test_generic_transfer_plans_preserve_families_and_source_only_roles(self):
        helper = test_research_transfer.SharedTransferTests()
        helper.setUp()
        records, main = helper.plan_fixture()
        records = {key.replace("NASA", "XJTU"): {**row, "metadata": {"dataset": row["metadata"]["dataset"].replace("NASA", "XJTU")},
                    "physical_family": row["physical_family"].replace("NASA", "XJTU")} for key, row in records.items()}
        for plan in main:
            plan["dataset"] = plan["dataset"].replace("NASA", "XJTU")
            for role in ("train", "validation", "test"):
                plan[role] = [key.replace("NASA", "XJTU") for key in plan[role]]
        plans = make_transfer_plans(records, main, mode=RESEARCH_MODE, directions=DIRECTIONS)
        self.assertEqual([p["name"] for p in plans], ["MIT_to_XJTU", "XJTU_to_MIT"])
        before = copy.deepcopy(plans)
        for row in records.values():
            row["final_life"], row["soh"] = -1, object()
        self.assertEqual(before, make_transfer_plans(records, main, mode=RESEARCH_MODE, directions=DIRECTIONS))
        for p in plans:
            self.assertNotEqual(p["source_physical_family"], p["target_physical_family"])
            self.assertEqual(p["target_calibration"], [])
        with self.assertRaisesRegex(ValueError, "unique"):
            make_transfer_plans(records, main, mode=RESEARCH_MODE, directions=(DIRECTIONS[0], DIRECTIONS[0]))

    def test_vi_and_vit_not_interchangeable_despite_same_width(self):
        cell = {"input_schema": SCHEMA, "feature_names": VI_FEATURE_NAMES,
            "reference_established_source_cycle": 1, "cycle": np.arange(1, 7),
            "common": np.ones((6, 11)), "minimum_signal_observations": 6}
        self.assertEqual(transfer_input_schema(cell), SCHEMA)
        with self.assertRaisesRegex(ValueError, "insufficient"):
            transfer_prefix(cell, 5)
        self.assertTrue(np.isfinite(transfer_prefix(cell, 6)).all())
        cell["input_schema"] = "shared_discharge_cached_v1_not_native_smetimes"
        with self.assertRaisesRegex(ValueError, "contradictory"):
            transfer_prefix(cell, 6)


if __name__ == "__main__":
    unittest.main()
