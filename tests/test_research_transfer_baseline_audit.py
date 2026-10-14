"""Toy 4/20-unit provenance ledgers; unit numerics are separately actual-tested."""

import copy
import importlib.metadata
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from battery_smetimes import research_transfer_baseline_audit as audit
from scripts.train_soh_research_transfer_baselines_v5 import ORIGINAL_MODEL_FILES, derive_baseline_source
from scripts.prepare_same_type_forecast import save_json
from battery_smetimes.research_transfer_vi120 import stream_sha


class BaselineWholeGridTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        native = self.root / "native"
        original = {}
        for name in ORIGINAL_MODEL_FILES:
            path = native / "source_snapshot" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# toy original source, no learned weights\n")
            original[name] = stream_sha(path)
        seeds, origins = [1, 2, 3, 4, 5], [20, 50, 100]
        native_config = {"source_sha256": original, "smoke": False, "tree_estimators": 160, "seeds": seeds,
            "config": {"origins_source_cycles": origins, "initial_candidates": {
                "ridge": [{"alpha": v} for v in (.1, 10., 100.)],
                "extra_trees": [{"min_samples_leaf": v} for v in (2, 4, 16)]}}}
        save_json(native / "config.json", native_config)
        self.binding = {"run": str(native), "config_sha256": stream_sha(native / "config.json")}
        prepared = self.root / "prepared"
        source = prepared / "source_snapshot/source.py"
        source.parent.mkdir(parents=True)
        source.write_text("# toy prepared contract only\n")
        self.inputs = {"training_seeds": seeds, "origins_source_cycles": origins, "input_schema": "toy_shared_schema",
            "feature_names": [], "source_sha256": {"source.py": stream_sha(source)}}
        save_json(prepared / "config.json", self.inputs)
        save_json(prepared / "manifest.json", {"toy": True})
        self.plans = [{"name": "MIT_to_XJTU"}, {"name": "XJTU_to_MIT"}]
        save_json(prepared / "splits.json", {"directions": self.plans})
        self.full = self.root / "runs/full"
        sources = {}
        for name in ORIGINAL_MODEL_FILES:
            path = self.full / "source_snapshot" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((native / "source_snapshot" / name).read_bytes())
            sources[name] = stream_sha(path)
        self.config = {"prepared": str(prepared), "prepared_config_sha256": stream_sha(prepared / "config.json"),
            "prepared_manifest_sha256": stream_sha(prepared / "manifest.json"), "prepared_splits_sha256": stream_sha(prepared / "splits.json"),
            "main_run_bindings": [self.binding], "baseline_source": derive_baseline_source([self.binding], self.inputs, smoke=False),
            "source_sha256": sources, "original_model_sha256": original, "seeds": seeds, "origins": origins, "models": ["ridge", "extra_trees"],
            "smoke": False, "prepare_only": False, "device": "cpu", "target_calibration": [], "not_same_type_deployment": True,
            "global_summary_writer": False, "input_schema": self.inputs["input_schema"], "feature_names": [],
            "packages": {"numpy": importlib.metadata.version("numpy")}}
        self.fake_unit = {"target_summary": {"100": {"cell_macro_mape_percent": 900.}}, "toy_unit": True}
        self.install(self.full, self.config)

    def install(self, run, config):
        run.mkdir(parents=True, exist_ok=True)
        save_json(run / "config.json", config)
        save_json(run / "splits.json", {"directions": self.plans})
        completed = []
        for plan in self.plans:
            for seed in config["seeds"]:
                for model in config["models"]:
                    unit = run / plan["name"] / str(seed) / model
                    unit.mkdir(parents=True, exist_ok=True)
                    save_json(unit / "static_unit_audit.json", self.fake_unit)
                    completed.append({"direction": plan["name"], "seed": seed, "model": model})
        save_json(run / "progress.json", {"complete": True, "model_training_started": True, "completed": completed,
            "expected_final_model_units": len(completed)})

    def test_full_twenty_unit_grid_required_and_finalization_flag_separate(self):
        from battery_smetimes.research_transfer_neural_audit import read
        with patch.object(audit, "audit_baseline_unit", return_value=self.fake_unit):
            report = audit.audit_baseline_run(self.full, {})
            self.assertEqual(report["finalized_units"], 20)
            self.assertEqual(len(report["units"]), 20)
            original = read(self.full / "progress.json")
            for kind in ("missing", "duplicate", "unknown", "not_complete"):
                state = copy.deepcopy(original)
                if kind == "missing":
                    state["completed"].pop()
                elif kind == "duplicate":
                    state["completed"].append(copy.deepcopy(state["completed"][0]))
                elif kind == "unknown":
                    state["completed"][0]["model"] = "llm_e1"
                else:
                    state["complete"] = False
                save_json(self.full / "progress.json", state)
                with self.assertRaises(ValueError):
                    audit.audit_baseline_run(self.full, {})
                if kind == "not_complete":
                    self.assertEqual(audit.audit_baseline_run(self.full, {}, require_complete=False), report)
            save_json(self.full / "progress.json", original)

    def test_original_source_budget_seed_and_input_mutation_rejected_without_unit_loading(self):
        for kind in ("source", "budget", "seed", "input", "calibration", "device"):
            config = copy.deepcopy(self.config)
            if kind == "source":
                config["original_model_sha256"][ORIGINAL_MODEL_FILES[0]] = "wrong"
            elif kind == "budget":
                config["baseline_source"]["candidates"]["ridge"] = [{"alpha": 1}]
            elif kind == "seed":
                config["seeds"] = [1]
            elif kind == "input":
                config["prepared_manifest_sha256"] = "wrong"
            elif kind == "calibration":
                config["target_calibration"] = ["target_cell"]
            else:
                config["device"] = "cuda"
            save_json(self.full / "config.json", config)
            with patch.object(audit, "audit_baseline_unit", side_effect=AssertionError("unchecked contract reached unit")):
                with self.assertRaises(ValueError):
                    audit.audit_baseline_run(self.full, {})
        save_json(self.full / "config.json", self.config)

    def test_complete_same_source_four_unit_smoke_consumed_without_accuracy_threshold(self):
        from battery_smetimes.research_transfer_neural_audit import read
        smoke = self.root / "runs/smoke"
        import shutil
        shutil.copytree(self.full / "source_snapshot", smoke / "source_snapshot")
        config = {**copy.deepcopy(self.config), "smoke": True, "seeds": [1],
            "baseline_source": derive_baseline_source([self.binding], self.inputs, smoke=True)}
        self.install(smoke, config)
        with patch.object(audit, "audit_baseline_unit", return_value=self.fake_unit):
            save_json(smoke / "static_run_audit.json", audit.audit_baseline_run(smoke, {}))
            accepted = audit.assert_baseline_smoke(self.root / "runs", self.config, {})
            self.assertEqual(accepted["run"], str(smoke.resolve()))
            self.assertIn("no target-score threshold", accepted["scope"])
            original = read(smoke / "progress.json")
            for kind in ("partial", "failed", "stale", "prepare_only"):
                if kind == "partial":
                    save_json(smoke / "progress.json", {**original, "complete": False})
                elif kind == "failed":
                    save_json(smoke / "failure.json", {"injected": True})
                elif kind == "stale":
                    save_json(smoke / "static_run_audit.json", {"toy": "changed"})
                else:
                    save_json(smoke / "config.json", {**config, "prepare_only": True})
                with self.assertRaisesRegex(ValueError, "four-unit baseline smoke"):
                    audit.assert_baseline_smoke(self.root / "runs", self.config, {})
                save_json(smoke / "progress.json", original)
                if (smoke / "failure.json").exists():
                    (smoke / "failure.json").unlink()  # Own explicitly named temporary injected artifact only.
                save_json(smoke / "config.json", config)
                save_json(smoke / "static_run_audit.json", audit.audit_baseline_run(smoke, {}))


if __name__ == "__main__":
    unittest.main()
