"""Toy main-source contracts; never evidence of completed public training."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
import shutil
import subprocess
import sys

from scripts.prepare_same_type_forecast import save_json, sha
from scripts.train_soh_research_transfer_neural_v5 import derive_model_sources


class SharedNeuralContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inputs = {"training_seeds": [1, 2, 3, 4, 5], "origins_source_cycles": [20, 50, 100]}
        self.bindings = []
        for name, architectures in (("small", ["gru", "statistics_mlp"]), ("llm", ["llm_e1", "llm_e2"])):
            run = self.root / name
            snapshot = run / "source_snapshot/source.py"
            snapshot.parent.mkdir(parents=True)
            snapshot.write_text("# source identity fixture only\n")
            config = {"smoke": False, "seeds": self.inputs["training_seeds"],
                "config": {"origins_source_cycles": self.inputs["origins_source_cycles"]},
                "architectures": architectures, "sequence_config": {"candidates": [{"parameter": n} for n in (1, 2, 3)],
                    "max_epochs": 60, "smoke_epochs": 2}, "source_sha256": {"source.py": sha(snapshot)},
                "checkpoint_files_sha256": {"model.safetensors": "toy-digest"} if name == "llm" else {}}
            save_json(run / "config.json", config)
            self.bindings.append({"run": str(run), "config_sha256": sha(run / "config.json")})

    def test_full_keeps_three_candidates_smoke_explicitly_one_and_two_epochs(self):
        full = derive_model_sources(self.bindings, self.inputs, smoke=False)
        smoke = derive_model_sources(self.bindings, self.inputs, smoke=True)
        self.assertEqual(set(full), {"gru", "statistics_mlp", "llm_e1", "llm_e2"})
        for model in full:
            self.assertEqual(len(full[model]["configuration"]["candidates"]), 3)
            self.assertEqual(full[model]["configuration"]["max_epochs"], 60)
            self.assertEqual(len(smoke[model]["configuration"]["candidates"]), 1)
            self.assertEqual(smoke[model]["configuration"]["max_epochs"], 2)
        self.assertEqual(full["llm_e1"]["checkpoint_files_sha256"], {"model.safetensors": "toy-digest"})
        self.assertEqual(json.loads((self.root / "small/config.json").read_text())["sequence_config"]["max_epochs"], 60)

    def test_missing_ambiguous_or_mutated_sources_rejected(self):
        for bindings in (self.bindings[:1], self.bindings + self.bindings[:1]):
            with self.assertRaises(ValueError):
                derive_model_sources(bindings, self.inputs, smoke=False)
        (self.root / "small/source_snapshot/source.py").write_text("changed")
        with self.assertRaisesRegex(ValueError, "source snapshot"):
            derive_model_sources(self.bindings, self.inputs, smoke=False)

    def test_main_budget_seed_origin_or_smoke_change_not_silently_accepted(self):
        path = self.root / "small/config.json"
        original = json.loads(path.read_text())
        for kind in ("smoke", "seed", "origin", "candidates", "binding"):
            value, bindings = copy.deepcopy(original), copy.deepcopy(self.bindings)
            if kind == "smoke":
                value["smoke"] = True
            elif kind == "seed":
                value["seeds"] = [1]
            elif kind == "origin":
                value["config"]["origins_source_cycles"] = [100]
            elif kind == "candidates":
                value["sequence_config"]["candidates"] = [{"parameter": 1}]
            else:
                value["sequence_config"]["max_epochs"] = 999
            save_json(path, value)
            if kind != "binding":
                bindings[0]["config_sha256"] = sha(path)
            with self.assertRaises(ValueError):
                derive_model_sources(bindings, self.inputs, smoke=False)
        save_json(path, original)

    def test_saved_direct_worker_bootstrap_and_prepare_only_guard(self):
        # Separate interpreter catches Windows script sys.path and namespace
        # package __path__ behavior that in-process imports cannot exercise.
        workspace = Path(__file__).resolve().parents[1]
        parent = workspace / "experiments/soh_research_v5/transfer_neural_runs"
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="bootstrap_test_", dir=parent) as name:
            run = Path(name)
            entry = "scripts/train_soh_research_transfer_neural_v5.py"
            destination = run / "source_snapshot" / entry
            destination.parent.mkdir(parents=True)
            shutil.copy2(workspace / entry, destination)
            save_json(run / "config.json", {"source_sha256": {entry: sha(destination)}, "prepare_only": True})
            save_json(run / "progress.json", {"completed": [], "model_training_started": False})
            process = subprocess.run([sys.executable, str(destination), "--run", str(run)], cwd=workspace,
                capture_output=True, text=True, timeout=30)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("prepared-only contract cannot be repurposed", process.stderr)
            failure = json.loads((run / "failure.json").read_text())
            self.assertFalse(failure["state"]["model_training_started"])
            self.assertFalse(list(run.rglob("model.pt")))


if __name__ == "__main__":
    unittest.main()
