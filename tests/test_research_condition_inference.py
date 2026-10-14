"""Synthetic fitted checkpoint tests, not real battery/condition accuracy evidence."""

import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np

from battery_smetimes.research_baselines import fit_baseline, evaluate_baseline
from battery_smetimes.research_conditions_execution import load_condition_contract, condition_job_configuration
from battery_smetimes.research_inference import load_model
from scripts.prepare_same_type_forecast import save_json, sha
from tests import test_research_conditions_execution as condition_fixture


class ConditionInferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        original_plan, _, main = condition_fixture.ConditionContractVerificationTests().fixture(self.root)
        condition = json.loads((original_plan / "config.json").read_text())
        audit = Path(condition["frozen_audit"])
        frozen = json.loads((audit / "config.json").read_text())
        main_config = json.loads((main / "config.json").read_text())
        main_config.update(seeds=condition["training_seeds"], config=frozen["config"],
                           model_candidates={"ridge": [{"alpha": n} for n in (1., 10., 100.)]})
        save_json(main / "config.json", main_config)
        condition["main_runs_with_frozen_model_search_and_source"][0]["config_sha256"] = sha(main / "config.json")
        save_json(original_plan / "config.json", condition)
        # Main CV provenance stays untouched and is NOT the condition split.
        save_json(audit / "splits.json", {"plans": [{"name": "unrelated_main_CV"}]})
        condition, _, records, registry, plans = load_condition_contract(original_plan)
        config, selected = condition_job_configuration(main_config, condition, plans, smoke=False)
        self.run = self.root / "condition_run"
        self.run.mkdir()
        for tree, source in (("source_snapshot", main / "source_snapshot"),
                             ("frozen_condition_plan", original_plan), ("frozen_audit", audit)):
            shutil.copytree(source, self.run / tree)
        workspace = Path(__file__).resolve().parents[1]
        execution = {}
        for name in ("battery_smetimes/research_inference.py", "battery_smetimes/research_conditions_execution.py"):
            destination = self.run / "execution_snapshot" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(workspace / name, destination)
            execution[name] = sha(destination)
        config.update(condition_plan_config_sha256=sha(original_plan / "config.json"),
            condition_plan_splits_sha256=sha(original_plan / "splits.json"),
            execution_source_sha256=execution, main_bindings=condition["main_runs_with_frozen_model_search_and_source"],
            bound_main_run=condition["main_runs_with_frozen_model_search_and_source"][0],
            frozen_audit_config_sha256=sha(audit / "config.json"),
            frozen_audit_files_sha256={str(p.relative_to(self.run / "frozen_audit")): sha(p)
                for p in (self.run / "frozen_audit").rglob("*") if p.is_file()})
        save_json(self.run / "config.json", config)
        save_json(self.run / "splits.json", {"plans": selected})
        self.plan = plans[0]
        self.folder = self.run / self.plan["name"] / "1/ridge"
        self.folder.mkdir(parents=True)
        self.recorded = {"plan": self.plan, "model": "ridge", "seed": 1, "smoke": False,
            "physical_family": self.plan["physical_family"], "candidates": config["model_candidates"]["ridge"]}
        save_json(self.folder / "config.json", self.recorded)
        rng = np.random.default_rng(4)
        self.cells = {key: {"metadata": row["metadata"], "physical_family": row["physical_family"],
            "reference_established_source_cycle": 1, "cycle": np.arange(1, 151),
            "raw": rng.normal(size=(150, 51)), "label_cycle": np.arange(1, 151), "soh": np.linspace(1., .8, 150)}
            for key, row in records.items() if row["physical_family"] == self.plan["physical_family"]}
        bundle = fit_baseline(self.cells, self.plan["train"] + self.plan["validation"], self.plan["physical_family"],
                              registry, frozen["config"], "ridge", {"alpha": 10.}, 1, smoke=False)
        joblib.dump(bundle, self.folder / "model.joblib")
        report, arrays = evaluate_baseline(bundle, self.cells, self.plan["test"])
        (self.folder / "test").mkdir()
        save_json(self.folder / "test/metrics.json", report)
        np.savez_compressed(self.folder / "test/predictions.npz", **arrays)
        self.saved = arrays
        for name in ("selected.json", "resources.json", "validation_scores.json"):
            save_json(self.folder / name, {"synthetic_fixture_only": True})
        save_json(self.folder / "artifact_audit.json", {"status": "passed static artifact/metric audit",
            "artifact_sha256": {str(p.relative_to(self.folder)): sha(p) for p in self.folder.rglob("*") if p.is_file()}})
        save_json(self.run / "progress.json", {"complete": False, "model_training_started": True,
            "expected_units": len(selected) * 5, "completed": [{"fold": self.plan["name"], "seed": 1, "model": "ridge"}]})

    def test_explicit_condition_reload_uses_true_partition_and_no_future_or_labels(self):
        with patch("sklearn.linear_model.Ridge.fit", side_effect=AssertionError("reload must never fit")):
            bundle, predict, evidence = load_model(self.folder, evaluation_protocol="condition")
            for key in self.plan["test"][:2]:
                for origin in (20, 50, 100):
                    prefix = f"{key}__{origin}__"
                    cell = copy.deepcopy(self.cells[key])
                    cell.pop("soh")
                    cell.pop("label_cycle")
                    cell["raw"][cell["cycle"] > origin] = np.inf
                    np.testing.assert_array_equal(predict(bundle, cell, origin, self.saved[prefix + "cycle"]),
                                                  self.saved[prefix + "prediction"])
        self.assertEqual(evidence["plan"], self.plan)
        self.assertTrue(evidence["condition_contract"]["not_main_grouped_cv"])
        self.assertTrue(evidence["no_fit"])

    def test_default_main_protocol_cannot_silently_load_condition_model(self):
        with self.assertRaisesRegex(ValueError, "explicit condition"):
            load_model(self.folder)

    def test_tampering_refused_before_deserialization(self):
        for mode in ("weights", "split", "execution", "progress", "search", "family", "copied_audit"):
            with self.subTest(mode=mode):
                # Restore pristine files between independent mutations.
                target = {"weights": self.folder / "model.joblib", "split": self.run / "splits.json",
                    "execution": self.run / "execution_snapshot/battery_smetimes/research_inference.py",
                    "progress": self.run / "progress.json", "search": self.folder / "config.json",
                    "family": self.folder / "config.json", "copied_audit": self.run / "frozen_audit/splits.json"}[mode]
                original = target.read_bytes()
                if mode in ("weights", "execution"):
                    target.write_bytes(b"changed source/checkpoint")
                else:
                    value = json.loads(target.read_text())
                    if mode == "progress":
                        value["completed"] = []
                    elif mode == "search":
                        value["candidates"] = [{"alpha": 999.}]
                    elif mode == "family":
                        value["physical_family"] = "XJTU-family"
                    else:
                        value["plans"] = []
                    save_json(target, value)
                try:
                    with patch("joblib.load", side_effect=AssertionError("tampered data reached deserialization")):
                        with self.assertRaises(ValueError):
                            load_model(self.folder, evaluation_protocol="condition")
                finally:
                    target.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
