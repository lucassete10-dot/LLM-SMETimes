import copy
import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np

from battery_smetimes.research_inference import load_model
from scripts.prepare_same_type_forecast import sha
from tests import test_research_baselines as baseline_fixture


class ResearchInferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name) / "run"
        self.folder = self.run / "MIT_fold1/123/ridge"
        self.folder.mkdir(parents=True)
        fixture = baseline_fixture.ResearchBaselineTests()
        fixture.setUp()
        self.fixture = fixture
        self.bundle = fixture.fit()
        self.plan = {"name": "MIT_fold1", "train": ["a"], "validation": ["b"], "test": ["held_out"]}
        self.save(self.folder / "config.json", {"plan": self.plan, "seed": 123, "model": "ridge",
            "physical_family": fixture.family})
        self.save(self.run / "frozen_audit/splits.json", {"plans": [self.plan]})
        registry_path = self.run / "frozen_audit/source_snapshot/battery_physical_registry.json"
        self.save(registry_path, fixture.registry)
        self.save(self.run / "frozen_audit/config.json", {"registry_sha256": sha(registry_path)})
        self.save(self.run / "frozen_audit/cell_registry.json", {"cells": {
            key: {"physical_family": fixture.family} for key in fixture.cells}})
        source = self.run / "source_snapshot/fixture.txt"
        source.parent.mkdir(parents=True)
        source.write_text("immutable test source", encoding="utf-8")
        self.save(self.run / "config.json", {"config": fixture.config, "source_sha256": {"fixture.txt": sha(source)}})
        joblib.dump(self.bundle, self.folder / "model.joblib")

    @staticmethod
    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def test_reload_exact_prediction_with_no_labels_or_future_signals(self):
        bundle, predict, evidence = load_model(self.folder)
        cell = copy.deepcopy(self.fixture.cells["held_out"])
        expected = predict(bundle, cell, 50, [60, 100, 150])
        del cell["soh"], cell["label_cycle"]
        cell["raw"][50:] = np.inf
        np.testing.assert_array_equal(expected, predict(bundle, cell, 50, [60, 100, 150]))
        self.assertTrue(evidence["source_snapshot_verified"])
        self.assertTrue(evidence["no_fit"])
        self.assertEqual(len(evidence["artifact_sha256"]["model.joblib"]), 64)

    def test_changed_source_snapshot_is_rejected(self):
        (self.run / "source_snapshot/fixture.txt").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source snapshot"):
            load_model(self.folder)

    def test_test_battery_cannot_enter_saved_fit_ids(self):
        self.bundle["train_ids"].append("held_out")
        joblib.dump(self.bundle, self.folder / "model.joblib")
        with self.assertRaisesRegex(ValueError, "train IDs"):
            load_model(self.folder)

    def test_mutated_registry_is_rejected(self):
        self.bundle["registry"] = copy.deepcopy(self.bundle["registry"])
        self.bundle["registry"]["families"][self.fixture.family]["specifications"]["nominal_capacity_ah"] = 2.0
        joblib.dump(self.bundle, self.folder / "model.joblib")
        with self.assertRaisesRegex(ValueError, "registry differs"):
            load_model(self.folder)

    def test_changed_frozen_registry_file_fails_fingerprint(self):
        self.save(self.run / "frozen_audit/source_snapshot/battery_physical_registry.json", {})
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            load_model(self.folder)


if __name__ == "__main__":
    unittest.main()
