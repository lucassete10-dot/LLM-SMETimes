import copy
import unittest
import json
from pathlib import Path
import tempfile
import shutil

from battery_smetimes.research_conditions_execution import condition_job_configuration, load_condition_contract
from battery_smetimes.research_generalization_splits import metadata_only, make_condition_splits
from scripts.prepare_same_type_forecast import save_json, sha
from scripts.train_soh_research_conditions_v5 import refresh_training_state, assert_condition_smoke
from tests import test_research_generalization


class ConditionJobConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.condition = {"origins_source_cycles": [20, 50, 100], "training_seeds": [1, 2, 3, 4, 5]}
        self.main = {"smoke": False, "seeds": [1, 2, 3, 4, 5], "folds": ["original_CV"],
            "config": {"origins_source_cycles": [20, 50, 100], "initial_candidates": {"ridge": [{"alpha": n} for n in (1, 10, 100)]}},
            "source_sha256": {"frozen_model.py": "digest"}, "evidence_level": "main grouped CV"}
        self.plans = [{"name": f"{domain}_condition{j}", "dataset": domain,
            "physical_family": domain+"-different-family", "train": ["train"], "validation": ["val"], "test": ["test"]}
            for domain, count in (("MIT", 3), ("XJTU", 6)) for j in range(1, count+1)]

    def job(self, smoke=False):
        return condition_job_configuration(self.main, self.condition, self.plans, smoke=smoke)

    def test_full_keeps_all_conditions_seeds_and_candidate_budget(self):
        original = copy.deepcopy(self.main)
        config, plans = self.job()
        self.assertEqual(plans, self.plans)
        self.assertEqual(len(config["folds"]), 9)
        self.assertEqual(config["seeds"], self.main["seeds"])
        self.assertEqual(config["model_candidates"], self.main["config"]["initial_candidates"])
        self.assertEqual(config["source_sha256"], self.main["source_sha256"])
        self.assertFalse(config["global_summary_writer"])
        self.assertTrue(config["not_main_grouped_cv"])
        self.assertEqual(self.main, original)

    def test_smoke_one_seed_and_one_whole_condition_per_supported_dataset(self):
        config, plans = self.job(True)
        self.assertEqual([p["name"] for p in plans], ["MIT_condition1", "XJTU_condition1"])
        self.assertEqual(config["seeds"], [1])
        # Baseline smoke still checks all three frozen parameters.
        self.assertEqual(len(config["model_candidates"]["ridge"]), 3)

    def test_sequence_budget_and_frozen_physical_families_are_not_changed(self):
        self.main.update(sequence_config={"kind": "small_models", "candidates": [{"hidden": n} for n in (32, 64, 96)], "max_epochs": 60},
            architectures=["gru", "lstm", "tcn", "transformer", "statistics_mlp"])
        config, plans = self.job()
        self.assertEqual(config["sequence_config"], self.main["sequence_config"])
        self.assertEqual(config["architectures"], self.main["architectures"])
        self.assertEqual({p["physical_family"] for p in plans}, {"MIT-different-family", "XJTU-different-family"})

    def test_llm_condition_scope_explicit_e1_not_a_new_full_ablation_claim(self):
        self.main.update(sequence_config={"kind": "frozen_llm", "candidates": [{"learning_rate": .001}], "variants": {"llm_e1": {"experts": 4}}},
            architectures=["llm_e1", "llm_e2", "llm_e4", "llm_experts1", "llm_experts8", "llm_matched_plain"])
        config, _ = self.job()
        self.assertEqual(config["architectures"], ["llm_e1"])
        self.assertEqual(config["sequence_config"], self.main["sequence_config"])
        self.assertIn("separate same-type", config["cross_condition_ablation_scope"])

    def test_smoke_main_or_changed_origin_or_seed_cannot_define_condition_budget(self):
        for mutation in ("smoke", "origin", "seed"):
            with self.subTest(mutation=mutation):
                main = copy.deepcopy(self.main)
                if mutation == "smoke":
                    main["smoke"] = True
                elif mutation == "origin":
                    main["config"]["origins_source_cycles"] = [50, 100]
                else:
                    main["seeds"].pop()
                with self.assertRaises(ValueError):
                    condition_job_configuration(main, self.condition, self.plans, smoke=False)


class ConditionContractVerificationTests(unittest.TestCase):
    def fixture(self, root):
        helper = test_research_generalization.ConditionSplitTests()
        helper.setUp()
        metadata = metadata_only(helper.records)
        audit, prepared, main, plan = [root / name for name in ("audit", "prepared", "main", "plan")]
        for folder in (audit, prepared, main, plan):
            (folder / "source_snapshot").mkdir(parents=True)
        registry = audit / "source_snapshot/battery_physical_registry.json"
        save_json(registry, helper.registry)
        save_json(audit / "cell_registry.json", {"cells": metadata})
        save_json(prepared / "manifest.json", {"cells": []})
        save_json(root / "integrity.json", {"issues": []})
        core = {**helper.config, "origins_source_cycles": [20, 50, 100], "training_seeds": [1, 2, 3, 4, 5],
                "integrity_audit": str(root / "integrity.json")}
        frozen = {"config": core, "registry_sha256": sha(registry),
            "prepared_manifest_sha256": sha(prepared / "manifest.json"), "integrity_sha256": sha(root / "integrity.json")}
        save_json(audit / "config.json", frozen)
        save_json(audit / "audit.json", {"prepared_folder": str(prepared)})
        (main / "frozen_audit").mkdir()
        shutil.copy2(audit / "config.json", main / "frozen_audit/config.json")
        code = main / "source_snapshot/fixture.py"
        code.write_text("# toy main source, no fitted model\n")
        main_config = {"smoke": False, "source_sha256": {"fixture.py": sha(code)}}
        save_json(main / "config.json", main_config)
        plan_code = plan / "source_snapshot/fixture.py"
        plan_code.write_text("# toy condition provenance fixture\n")
        condition = {key: core[key] for key in ("origins_source_cycles", "training_seeds", "split_seed", "validation_fraction_of_development")}
        condition.update(frozen_audit=str(audit), audit_config_sha256=sha(audit / "config.json"),
            source_cell_registry_sha256=sha(audit / "cell_registry.json"), registry_sha256=sha(registry),
            prepared_manifest_sha256=frozen["prepared_manifest_sha256"], integrity_sha256=frozen["integrity_sha256"],
            source_sha256={"fixture.py": sha(plan_code)},
            main_runs_with_frozen_model_search_and_source=[{"run": str(main), "config_sha256": sha(main / "config.json"),
                "source_sha256": main_config["source_sha256"]}])
        save_json(plan / "config.json", condition)
        save_json(plan / "metadata_only.json", {"cells": metadata})
        save_json(plan / "splits.json", {"plans": make_condition_splits(metadata, core, helper.registry)})
        return plan, prepared, main

    def test_unchanged_contract_preserves_whole_condition_membership(self):
        with tempfile.TemporaryDirectory() as name:
            plan, _, _ = self.fixture(Path(name))
            config, _, records, _, plans = load_condition_contract(plan)
            self.assertEqual(len(plans), 6)
            self.assertEqual(len(records), 31)
            self.assertEqual(config["training_seeds"], [1, 2, 3, 4, 5])

    def test_saved_partition_or_metadata_tamper_refused(self):
        for kind in ("partition", "metadata"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as name:
                plan, _, _ = self.fixture(Path(name))
                if kind == "partition":
                    p = plan / "splits.json"
                    value = json.loads(p.read_text())
                    value["plans"][0]["test"].pop()
                else:
                    p = plan / "metadata_only.json"
                    value = json.loads(p.read_text())
                    value["cells"]["mit-0-0"]["soh"] = [1., .9]
                save_json(p, value)
                with self.assertRaises(ValueError):
                    load_condition_contract(plan)

    def test_bound_main_source_or_data_fingerprint_change_refused(self):
        for kind in ("source", "main", "data"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as name:
                plan, prepared, main = self.fixture(Path(name))
                if kind == "source":
                    (main / "source_snapshot/fixture.py").write_text("changed\n")
                elif kind == "main":
                    save_json(main / "config.json", {"smoke": True, "source_sha256": {}})
                else:
                    save_json(prepared / "manifest.json", {"cells": ["changed"]})
                with self.assertRaises(ValueError):
                    load_condition_contract(plan)

    def test_training_observation_is_monotone_and_transient_parse_never_restarts(self):
        with tempfile.TemporaryDirectory() as name:
            run = Path(name)
            state = {"model_training_started": False}
            refresh_training_state(state, run)
            self.assertFalse(state["model_training_started"])
            save_json(run / "progress.json", {"model_training_started": True})
            refresh_training_state(state, run)
            self.assertTrue(state["model_training_started"])
            (run / "progress.json").write_text("{partial")
            refresh_training_state(state, run)
            self.assertTrue(state["model_training_started"])
            save_json(run / "progress.json", {"model_training_started": False})
            refresh_training_state(state, run)
            self.assertTrue(state["model_training_started"])


class ConditionSmokeGateTests(unittest.TestCase):
    """Toy artifacts only test prerequisite logic; not numeric/model evidence."""

    def fixture(self, root):
        full, smoke, batch = [root / name for name in ("full", "smoke", "batches/batch1")]
        for path in (full, smoke, batch):
            path.mkdir(parents=True)
        model_source = smoke / "source_snapshot/model.py"
        execution_source = smoke / "execution_snapshot/worker.py"
        for path in (model_source, execution_source):
            path.parent.mkdir(parents=True)
            path.write_text("# toy snapshot, no executable training evidence\n")
        config = {"bound_main_run": {"run": str(root / "bound_main"), "config_sha256": "main-digest"},
            "source_sha256": {"model.py": sha(model_source)}, "execution_source_sha256": {"worker.py": sha(execution_source)},
            "condition_plan_config_sha256": "plan-digest", "condition_plan_splits_sha256": "split-digest",
            "smoke": False, "seeds": [1, 2, 3, 4, 5], "model_candidates": {"ridge": [{"alpha": 1}]}}
        save_json(full / "config.json", config)
        save_json(smoke / "config.json", {**config, "smoke": True, "seeds": [1]})
        plans = [{"name": domain+"_condition1", "dataset": domain} for domain in ("MIT", "XJTU")]
        for run in (full, smoke):
            save_json(run / "splits.json", {"plans": plans})
        completed = []
        for plan in plans:
            unit = smoke / plan["name"] / "1/ridge"
            unit.mkdir(parents=True)
            hashes = {}
            for name in ("config.json", "selected.json", "resources.json", "validation_scores.json", "model.joblib",
                         "test/metrics.json", "test/predictions.npz"):
                p = unit / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"toy prerequisite fingerprint, not a fitted model")
                hashes[str(Path(name))] = sha(p)
            save_json(unit / "artifact_audit.json", {"status": "passed static artifact/metric audit", "artifact_sha256": hashes})
            completed.append({"fold": plan["name"], "seed": 1, "model": "ridge"})
        save_json(smoke / "progress.json", {"complete": True, "completed": completed, "model_training_started": True})
        save_json(batch / "config.json", {"smoke": True, "prepare_only": False})
        save_json(batch / "progress.json", {"complete": True, "model_training_started": True,
            "jobs_prepared": [{"run": str(smoke)}]})
        return full, smoke, batch

    def test_completed_bound_prerequisite_accepted_without_accuracy_threshold(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            full, _, batch = self.fixture(root)
            evidence = assert_condition_smoke(root / "batches", [{"run": str(full)}])
            self.assertEqual(evidence["batch"], str(batch.resolve()))
            self.assertIn("no accuracy threshold", evidence["scope"])

    def test_prepare_only_or_inspection_cannot_qualify_as_trained_smoke(self):
        for mode in ("prepare", "inspect"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as name:
                root = Path(name)
                full, _, batch = self.fixture(root)
                if mode == "prepare":
                    save_json(batch / "config.json", {"smoke": True, "prepare_only": True})
                else:
                    value = json.loads((batch / "progress.json").read_text())
                    value["model_training_started"] = False
                    save_json(batch / "progress.json", value)
                with self.assertRaisesRegex(ValueError, "completed real condition smoke"):
                    assert_condition_smoke(root / "batches", [{"run": str(full)}])

    def test_changed_model_or_missing_condition_or_source_mismatch_refused(self):
        for mode in ("model", "condition", "source", "model_source", "execution_source"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as name:
                root = Path(name)
                full, smoke, _ = self.fixture(root)
                if mode == "model":
                    next(smoke.rglob("model.joblib")).write_bytes(b"changed")
                elif mode == "condition":
                    value = json.loads((smoke / "progress.json").read_text())
                    value["completed"].pop()
                    save_json(smoke / "progress.json", value)
                elif mode == "source":
                    value = json.loads((smoke / "config.json").read_text())
                    value["execution_source_sha256"] = {"worker.py": "changed"}
                    save_json(smoke / "config.json", value)
                else:
                    path = smoke / ("source_snapshot/model.py" if mode == "model_source" else "execution_snapshot/worker.py")
                    path.write_text("changed after smoke\n")
                with self.assertRaisesRegex(ValueError, "completed real condition smoke"):
                    assert_condition_smoke(root / "batches", [{"run": str(full)}])

    def test_transient_or_malformed_newer_batch_cannot_hide_valid_smoke(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            full, _, batch = self.fixture(root)
            newer = root / "batches/z_partial"
            newer.mkdir()
            (newer / "progress.json").write_text("{partial")
            evidence = assert_condition_smoke(root / "batches", [{"run": str(full)}])
            self.assertEqual(evidence["batch"], str(batch.resolve()))
            (batch / "progress.json").write_text("{partial")
            with self.assertRaisesRegex(ValueError, "completed real condition smoke"):
                assert_condition_smoke(root / "batches", [{"run": str(full)}])


if __name__ == "__main__":
    unittest.main()
