import copy
import unittest

from battery_smetimes.physical_types import (compare_specifications,
    family_for_registered_record, load_registry, match_external_specifications,
    require_same_registered_family)
from battery_smetimes.research_splits import make_research_splits


class PhysicalTypeTests(unittest.TestCase):
    def setUp(self):
        self.registry = load_registry()
        self.families = self.registry["families"]
        self.mit, self.xjtu, self.nasa = list(self.families)
        self.spec = copy.deepcopy(self.families[self.mit]["specifications"])

    def test_mit_cannot_be_nasa_even_with_unknown_nasa_chemistry(self):
        result = compare_specifications(self.spec, self.families[self.nasa]["specifications"])
        self.assertEqual(result["status"], "incompatible")
        self.assertIn("nominal_capacity_ah", result["conflicts"])
        with self.assertRaisesRegex(ValueError, "incompatible"):
            require_same_registered_family(self.mit, [self.nasa], self.registry)

    def test_mit_and_xjtu_chemistry_is_incompatible(self):
        result = compare_specifications(self.spec, self.families[self.xjtu]["specifications"])
        self.assertEqual(result["status"], "incompatible")
        self.assertIn("chemistry", result["conflicts"])

    def test_model_not_dataset_label_defines_external_match(self):
        self.spec["dataset"] = "NASA"
        self.spec["group"] = "batch_does_not_matter"
        self.assertEqual(match_external_specifications(self.spec)["physical_family"], self.mit)
        self.spec["model"] = "a_different_model"
        self.assertIsNone(match_external_specifications(self.spec)["physical_family"])

    def test_missing_specs_are_not_wildcards_or_a_forced_class(self):
        result = match_external_specifications({"dataset": "MIT", "nominal_capacity_ah": 2.0})
        self.assertEqual(result["status"], "needs_specs")
        self.assertIsNone(result["physical_family"])
        for key in (self.xjtu, self.nasa):
            result = match_external_specifications(self.families[key]["specifications"])
            self.assertEqual(result["status"], "needs_specs")
            self.assertIsNone(result["physical_family"])

    def test_ncm_alias_exact_composition_not_broad_family(self):
        a = {"chemistry": "NMC532"}
        b = {"chemistry": "NCM532"}
        self.assertNotEqual(compare_specifications(a, b)["status"], "incompatible")
        self.assertEqual(compare_specifications({"chemistry": "NMC811"}, b)["status"], "incompatible")

    def test_invalid_numeric_values_rejected(self):
        for bad in (float("nan"), float("inf"), -1, True):
            self.spec["nominal_capacity_ah"] = bad
            self.assertEqual(match_external_specifications(self.spec)["status"], "invalid_specs")

    def test_registered_metadata_contradiction_is_rejected(self):
        record = {"cell_id": "b1c0", "metadata": {"dataset": "MIT", "chemistry": "NCM532"}}
        with self.assertRaisesRegex(ValueError, "contradicts"):
            family_for_registered_record(record)

    def test_batch_does_not_change_physical_type(self):
        families = [family_for_registered_record({"cell_id": "b1c0", "metadata": {
            "dataset": "MIT", "chemistry": "LFP/graphite", "group": group}})
            for group in ("batch1", "batch2", "batch3")]
        self.assertEqual(set(families), {self.mit})

    def test_five_fold_whole_cell_coverage_and_reproducibility(self):
        records = {f"c{i}": {"physical_family": self.mit, "metadata": {"dataset": "MIT",
            "group": "batch1", "charge_protocol": "4C(80%)-4C"}} for i in range(17)}
        cfg = {"outer_folds": 5, "split_seed": 123, "validation_fraction_of_development": .2}
        plans = make_research_splits(records, cfg, self.registry)
        self.assertEqual(plans, make_research_splits(dict(reversed(list(records.items()))), cfg, self.registry))
        self.assertEqual(sorted(k for plan in plans for k in plan["test"]), sorted(records))
        for plan in plans:
            self.assertFalse(set(plan["train"]) & set(plan["test"]))
            self.assertFalse(set(plan["validation"]) & set(plan["test"]))


if __name__ == "__main__":
    unittest.main()
