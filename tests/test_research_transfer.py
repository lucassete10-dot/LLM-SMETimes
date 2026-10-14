import copy
import unittest

import numpy as np

from battery_smetimes.physical_types import load_registry
from battery_smetimes.research_transfer_signals import (
    FEATURE_NAMES, NATIVE_GRID, NATIVE_WIDTH, RESEARCH_MODE,
    make_transfer_plans, shared_discharge_history,
)


class SharedTransferTests(unittest.TestCase):
    def setUp(self):
        self.registry = load_registry()
        self.families = {dataset: next(key for key, row in self.registry["families"].items()
                                     if dataset in row["datasets"]) for dataset in ("MIT", "NASA")}

    def cell(self, domain):
        grid = NATIVE_GRID[domain]
        capacity = self.registry["families"][self.families[domain]]["specifications"]["nominal_capacity_ah"]
        row = np.r_[4 - grid / 1000, np.full(len(grid), -2 * capacity), 25 + grid / 100,
                    np.zeros(NATIVE_WIDTH[domain] - 3 * len(grid))]
        return {"metadata": {"dataset": domain}, "physical_family": self.families[domain],
                "cycle": np.arange(1, 6), "raw": np.tile(row, (5, 1)),
                "soh": object(), "label_cycle": object(), "reference_capacity_ah": -123}

    def shared(self, cell, registry=None):
        return shared_discharge_history(cell, 3, registry or self.registry, mode=RESEARCH_MODE)

    def plan_fixture(self):
        records, plans = {}, []
        for domain in ("MIT", "NASA"):
            ids = [f"{domain}-{j}" for j in range(4)]
            for key in ids:
                records[key] = {"metadata": {"dataset": domain}, "physical_family": self.families[domain],
                                "soh": object(), "final_life": object()}
            for j, key in enumerate(ids):
                others = sorted(set(ids) - {key})
                plans.append({"name": f"{domain}_fold{j + 1}", "dataset": domain,
                              "train": others[1:], "validation": others[:1], "test": [key]})
        return records, plans

    def test_common_grid_and_rated_current_units_match_for_known_linear_signals(self):
        c1, x1 = self.shared(self.cell("MIT"))
        c2, x2 = self.shared(self.cell("NASA"))
        np.testing.assert_array_equal(c1, [1, 2, 3])
        np.testing.assert_array_equal(c1, c2)
        np.testing.assert_allclose(x1, x2, atol=1e-6)
        self.assertEqual(x1.shape, (3, len(FEATURE_NAMES)))
        np.testing.assert_allclose(x1[0], [0, -.1, -.2, 2, 2, 2, 0, 1, 2, 3.9, 26], atol=1e-6)

    def test_labels_future_raw_and_unused_summary_columns_do_not_affect_input(self):
        cell = self.cell("NASA")
        before = self.shared(cell)[1]
        cell["raw"][3:] = np.inf
        cell["raw"][:3, 33:] = np.inf
        cell["raw"][:3, [0, 4, 5, 6, 7, 8, 9, 10]] = np.inf
        cell.update({"soh": [-999], "label_cycle": [-999], "reference_capacity_ah": np.inf})
        np.testing.assert_array_equal(before, self.shared(cell)[1])

    def test_missing_temperature_remains_missing_but_bad_voltage_is_rejected(self):
        cell = self.cell("MIT")
        cell["raw"][:, 22:33] = np.nan
        values = self.shared(cell)[1]
        self.assertTrue(np.isnan(values[:, 6:9]).all())
        self.assertTrue(np.isnan(values[:, 10]).all())
        cell["raw"][0, 4] = np.nan
        with self.assertRaisesRegex(ValueError, "invalid shared visible"):
            self.shared(cell)

    def test_xjtu_window_cannot_be_extended_or_temperature_invented(self):
        cell = {"metadata": {"dataset": "XJTU"}, "raw": np.zeros((100, 46))}
        with self.assertRaisesRegex(ValueError, "no shared"):
            self.shared(cell)

    def test_requires_explicit_research_mode_and_preserves_family(self):
        cell = self.cell("MIT")
        before = copy.deepcopy(cell)
        with self.assertRaisesRegex(ValueError, "research-only"):
            shared_discharge_history(cell, 3, self.registry, mode="deployment")
        self.shared(cell)
        self.assertEqual(cell["metadata"], before["metadata"])
        self.assertEqual(cell["physical_family"], before["physical_family"])
        np.testing.assert_array_equal(cell["raw"], before["raw"])

    def test_missing_rated_capacity_cannot_be_replaced_by_measured_capacity(self):
        registry = copy.deepcopy(self.registry)
        registry["families"][self.families["NASA"]]["specifications"]["nominal_capacity_ah"] = None
        with self.assertRaisesRegex(ValueError, "RATED capacity"):
            self.shared(self.cell("NASA"), registry)

    def test_cannot_assign_mit_to_nasa_family_for_input_compatibility(self):
        cell = self.cell("MIT")
        cell["physical_family"] = self.families["NASA"]
        with self.assertRaisesRegex(ValueError, "contradicts"):
            self.shared(cell)

    def test_source_only_selection_and_target_test_roles_in_both_directions(self):
        records, main = self.plan_fixture()
        directions = make_transfer_plans(records, main, mode=RESEARCH_MODE)
        self.assertEqual([p["name"] for p in directions], ["MIT_to_NASA", "NASA_to_MIT"])
        for plan in directions:
            self.assertEqual(plan["target_calibration"], [])
            self.assertNotEqual(plan["source_physical_family"], plan["target_physical_family"])
            source, target = set(plan["source_refit"]), set(plan["target_test"])
            self.assertFalse(source & target)
            for fold in plan["source_selection_folds"]:
                self.assertFalse(set(fold["source_train"]) & set(fold["source_validation"]))
                self.assertEqual(set(fold["source_train"] + fold["source_validation"]), source)
                self.assertFalse(set(fold["source_train"] + fold["source_validation"]) & target)

    def test_target_labels_and_lifetime_do_not_change_transfer_splits(self):
        records, main = self.plan_fixture()
        before = make_transfer_plans(records, main, mode=RESEARCH_MODE)
        for row in records.values():
            row.update({"soh": [np.inf], "final_life": -1, "mape": 999})
        self.assertEqual(before, make_transfer_plans(records, main, mode=RESEARCH_MODE))

    def test_target_cell_cannot_enter_source_selection(self):
        records, main = self.plan_fixture()
        main[0]["train"].append("NASA-0")
        with self.assertRaisesRegex(ValueError, "incomplete or overlaps"):
            make_transfer_plans(records, main, mode=RESEARCH_MODE)

    def test_missing_source_validation_fold_and_same_family_are_rejected(self):
        records, main = self.plan_fixture()
        with self.assertRaisesRegex(ValueError, "each source cell exactly once"):
            make_transfer_plans(records, main[1:], mode=RESEARCH_MODE)
        for row in records.values():
            row["physical_family"] = self.families["MIT"]
        with self.assertRaisesRegex(ValueError, "distinct isolated"):
            make_transfer_plans(records, main, mode=RESEARCH_MODE)


if __name__ == "__main__":
    unittest.main()
