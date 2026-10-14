import unittest

import numpy as np

from battery_smetimes.physical_types import load_registry
from battery_smetimes.research_baselines import fit_baseline, predict_baseline


class ExtendedBaselineTests(unittest.TestCase):
    def setUp(self):
        self.registry = load_registry()
        self.family = next(iter(self.registry["families"]))
        rng = np.random.default_rng(12)
        self.cells = {key: {"cycle": np.arange(1, 151), "raw": rng.normal(size=(150, 51)),
            "label_cycle": np.arange(1, 151), "soh": np.linspace(1, .8, 150) + shift,
            "reference_established_source_cycle": 1, "physical_family": self.family,
            "metadata": {"dataset": "MIT", "group": "batch1", "charge_protocol": "4C(80%)-4C",
                         "diagnostic_discharge_rate_c": 4}}
            for key, shift in (("a", 0), ("b", .01), ("heldout", -.01))}
        self.config = {"origins_source_cycles": [20, 50, 100]}

    def fit(self, name, parameters):
        return fit_baseline(self.cells, ["a", "b"], self.family, self.registry,
                            self.config, name, parameters, 123, smoke=True)

    def test_cycle_control_does_not_read_voltage_current_temperature(self):
        bundle = self.fit("cycle_conditions_only", {"alpha": 10})
        before = predict_baseline(bundle, self.cells["heldout"], 50, [60, 100])
        self.cells["heldout"]["raw"][:] = 1e8
        np.testing.assert_array_equal(before, predict_baseline(bundle, self.cells["heldout"], 50, [60, 100]))

    def test_family_mean_uses_only_listed_training_labels(self):
        before = self.fit("physical_family_mean", {})
        self.cells["heldout"]["soh"][:] = 10000
        after = self.fit("physical_family_mean", {})
        np.testing.assert_array_equal(predict_baseline(before, self.cells["heldout"], 50, [60, 100]),
                                      predict_baseline(after, self.cells["heldout"], 50, [60, 100]))
        self.assertEqual({r["cell_id"] for r in before["templates"]}, {"a", "b"})

    def test_gpr_coreset_retains_every_training_cell_only(self):
        bundle = self.fit("gpr", {"initial_length_factor": 1., "initial_noise": .03, "max_training_pairs": 16})
        self.assertEqual({r["cell_id"] for r in bundle["coreset_by_cell"]}, {"a", "b"})
        self.assertLessEqual(bundle["estimator_training_pairs"], 16)
        self.assertTrue(np.isfinite(predict_baseline(bundle, self.cells["heldout"], 50, [60, 100])).all())

    def test_elastic_net_has_finite_deterministic_predictions(self):
        bundle = self.fit("elastic_net", {"alpha": .003, "l1_ratio": .5})
        prediction = predict_baseline(bundle, self.cells["heldout"], 50, [60, 100])
        self.assertTrue(np.isfinite(prediction).all())
        self.assertIn("dual_gap", bundle)

    def test_lightgbm_gate_and_future_signal_invariance(self):
        bundle = self.fit("lightgbm", {"num_leaves": 7, "learning_rate": .03, "reg_lambda": 1.})
        prediction = predict_baseline(bundle, self.cells["heldout"], 50, [60, 100])
        self.cells["heldout"]["raw"][50:] = np.inf
        np.testing.assert_array_equal(prediction, predict_baseline(bundle, self.cells["heldout"], 50, [60, 100]))
        self.cells["heldout"]["physical_family"] = list(self.registry["families"])[1]
        with self.assertRaisesRegex(ValueError, "incompatible"):
            predict_baseline(bundle, self.cells["heldout"], 50, [60, 100])


if __name__ == "__main__":
    unittest.main()
