"""Condition holdouts use metadata, whole cells and the unchanged physical gate."""

import copy
import unittest

from battery_smetimes.research_generalization_splits import (
    heldout_condition, make_condition_splits, metadata_only, validate_condition_splits,
)


class ConditionSplitTests(unittest.TestCase):
    def setUp(self):
        self.config = {"split_seed": 20260930, "validation_fraction_of_development": .2}
        self.registry = {"families": {"MIT-family": {"status": "verified"},
                                     "XJTU-family": {"status": "cohort"},
                                     "NASA-family": {"status": "unverified"}}}
        self.records = {}
        for j, protocol in enumerate(("3C(50%)-3C", "4C(50%)-2C", "2C(50%)-4C")):
            for k in range(6):
                self.records[f"mit-{j}-{k}"] = {"metadata": {"dataset": "MIT",
                    "group": "batch1" if k < 3 else "batch2", "charge_protocol": protocol},
                    "physical_family": "MIT-family", "soh": object(), "final_life": object()}
        for j, protocol in enumerate(("2C charge/1C full discharge", "3C charge/1C full discharge",
                                     "random partial discharge; 1C diagnostic")):
            for k in range(4):
                self.records[f"xjtu-{j}-{k}"] = {"metadata": {"dataset": "XJTU",
                    "group": f"batch{j + 1}", "charge_protocol": protocol},
                    "physical_family": "XJTU-family"}
        self.records["nasa"] = {"metadata": {"dataset": "NASA", "group": "NASA",
                                             "discharge_cutoff_v": 2.5},
                                "physical_family": "NASA-family"}

    def plans(self):
        return make_condition_splits(self.records, self.config, self.registry)

    def test_whole_condition_is_held_out_and_each_cell_tested_once(self):
        plans = self.plans()
        self.assertEqual(len(plans), 6)
        ids = [key for plan in plans for key in plan["test"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), set(self.records) - {"nasa"})
        for plan in plans:
            self.assertFalse(set(plan["train"]) & set(plan["validation"]))
            for key in plan["test"]:
                self.assertEqual(heldout_condition(self.records[key]["metadata"]), plan["heldout_condition"])
            for key in plan["train"] + plan["validation"]:
                self.assertNotEqual(heldout_condition(self.records[key]["metadata"]), plan["heldout_condition"])

    def test_lifetime_labels_and_predictions_do_not_affect_plans(self):
        before = self.plans()
        for row in self.records.values():
            row.update({"soh": [float("inf")], "final_life": -999, "mape": 999,
                        "reference_capacity_ah": -1, "raw": object()})
            row["metadata"]["test_error"] = object()
        self.assertEqual(before, self.plans())
        for row in metadata_only(self.records).values():
            self.assertEqual(set(row), {"metadata", "physical_family"})
            self.assertNotIn("test_error", row["metadata"])

    def test_order_independent_and_reproducible(self):
        reverse = dict(reversed(list(self.records.items())))
        self.assertEqual(self.plans(), make_condition_splits(reverse, self.config, self.registry))

    def test_different_physical_family_cannot_be_merged(self):
        self.records["mit-0-0"]["physical_family"] = "NASA-family"
        with self.assertRaisesRegex(ValueError, "incompatible physical-family"):
            self.plans()

    def test_partial_condition_holdout_rejected(self):
        plans = self.plans()
        plans[0]["train"].append(plans[0]["test"].pop())
        with self.assertRaisesRegex(ValueError, "entire held-out condition"):
            validate_condition_splits(plans, self.records, self.registry)

    def test_overlapping_partitions_rejected(self):
        plans = self.plans()
        plans[0]["validation"].append(plans[0]["train"][0])
        with self.assertRaisesRegex(ValueError, "nonempty and disjoint"):
            validate_condition_splits(plans, self.records, self.registry)

    def test_missing_or_duplicate_condition_plan_rejected(self):
        plans = self.plans()
        with self.assertRaisesRegex(ValueError, "test each source cell exactly once"):
            validate_condition_splits(plans[1:], self.records, self.registry)
        with self.assertRaisesRegex(ValueError, "duplicate held-out condition"):
            validate_condition_splits(plans + [copy.deepcopy(plans[0])], self.records, self.registry)

    def test_omitting_entire_dataset_cannot_look_complete(self):
        plans = [plan for plan in self.plans() if plan["dataset"] != "MIT"]
        with self.assertRaisesRegex(ValueError, "entire supported dataset"):
            validate_condition_splits(plans, self.records, self.registry)

    def test_missing_known_condition_is_not_a_new_type(self):
        for metadata in ({"dataset": "MIT", "charge_protocol": None},
                         {"dataset": "XJTU", "charge_protocol": ""}):
            with self.assertRaisesRegex(ValueError, "known"):
                heldout_condition(metadata)

    def test_invalid_fraction_or_one_condition_rejected(self):
        with self.assertRaisesRegex(ValueError, "validation fraction"):
            make_condition_splits(self.records, {**self.config, "validation_fraction_of_development": 0}, self.registry)
        records = {k: v for k, v in self.records.items() if k.startswith("mit-0-")}
        with self.assertRaisesRegex(ValueError, "at least two"):
            make_condition_splits(records, self.config, self.registry)

    def test_nominal_rate_order_is_not_lifetime_or_batch(self):
        self.assertEqual(heldout_condition({"dataset": "MIT", "group": "batch99",
            "charge_protocol": "1C(4%)-6C-newstructure"}), "increasing")
        self.assertEqual(heldout_condition({"dataset": "MIT", "group": "batch99",
            "charge_protocol": "6C(80%)-1"}), "decreasing")


if __name__ == "__main__":
    unittest.main()
