"""Synthetic shared-neural transfer tests; tiny mocked LLM is NOT real LLM evidence."""

import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from battery_smetimes.research_sequences import fit_epoch
from battery_smetimes.research_llm import fit_llm_epoch, adapter_state
from battery_smetimes.research_transfer_baselines import evaluate_transfer
from battery_smetimes.research_transfer_sequences import (
    TransferHistoryPreprocessor, make_transfer_sequence_bundle, predict_transfer_sequence,
    shared_statistics_cards, transfer_training_contexts,
)
from battery_smetimes.research_transfer_signals import RESEARCH_MODE
from battery_smetimes.research_transfer_vi120 import SCHEMA as VI_SCHEMA, VI_FEATURE_NAMES
from tests import test_research_transfer_execution as transfer_fixture


class TinyTokenBatch(dict):
    def to(self, device):
        return TinyTokenBatch({k: v.to(device) for k, v in self.items()})


class TinyTokenizer:
    pad_token_id = 0
    def __call__(self, cards, **kwargs):
        # Text really traverses the same mocked frozen backbone as numbers.
        ids = torch.tensor([[sum(map(ord, card)) % 31, 1, 2] for card in cards])
        return TinyTokenBatch(input_ids=ids, attention_mask=torch.ones_like(ids))


class TinyTransferBackbone(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=16)
        self.embeddings = nn.Embedding(32, 16)
        self.layer = nn.Linear(16, 16)
        self.to(dtype=dtype)

    def forward(self, inputs_embeds=None, input_ids=None, **kwargs):
        value = inputs_embeds if inputs_embeds is not None else self.embeddings(input_ids)
        return SimpleNamespace(last_hidden_state=self.layer(value))


class SharedSequenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        fixture = transfer_fixture.TransferFitTests()
        fixture.setUp()
        self.cells, self.plan = fixture.cells, fixture.plan
        self.plan["target_dataset"] = "XJTU"
        for key, cell in self.cells.items():
            cell.update(input_schema=VI_SCHEMA, feature_names=VI_FEATURE_NAMES)
            if key.startswith("t"):
                cell["metadata"]["dataset"] = "XJTU"
                cell["minimum_signal_observations"] = 6
        self.origins = [20, 50, 100]
        self.parameters = {"hidden": 16, "dropout": 0., "learning_rate": .001, "weight_decay": .01}

    def processor(self, ids=None):
        return TransferHistoryPreprocessor().fit(self.cells, ids or ["s1", "s2"], self.plan, self.origins)

    def bundle(self, architecture="gru", llm=None):
        return make_transfer_sequence_bundle(self.processor(), architecture, self.parameters, 9, "cpu", llm_config=llm)

    def test_target_values_and_labels_cannot_affect_any_source_scale(self):
        before = self.processor()
        for key in self.plan["target_test"]:
            self.cells[key]["common"][:] = np.inf
            self.cells[key]["soh"] = object()
            self.cells[key]["label_cycle"] = object()
        after = self.processor()
        for name in ("raw_scale", "stats_scale", "query_scale"):
            np.testing.assert_array_equal(getattr(before, name).imputer.statistics_, getattr(after, name).imputer.statistics_)
            np.testing.assert_array_equal(getattr(before, name).scaler.mean_, getattr(after, name).scaler.mean_)
            np.testing.assert_array_equal(getattr(before, name).scaler.var_, getattr(after, name).scaler.var_)
        self.assertEqual(after.train_ids, ["s1", "s2"])

    def test_target_duplicate_and_empty_fit_ids_rejected(self):
        for ids in (["s1", "t1"], ["s1", "s1"], []):
            with self.assertRaisesRegex(ValueError, "source-only neural"):
                TransferHistoryPreprocessor().fit(self.cells, ids, self.plan, self.origins)

    def test_native_and_equal_width_different_schema_cannot_be_silently_used(self):
        self.cells["t1"].pop("input_schema")
        self.cells["t1"].pop("feature_names")
        with self.assertRaisesRegex(ValueError, "cannot mix"):
            self.processor()
        self.cells["t1"]["input_schema"] = "native_51_columns"
        with self.assertRaisesRegex(ValueError, "schema"):
            self.processor()

    def test_input_shape_mask_and_zero_slots_are_explicit_without_labels(self):
        processor = self.processor()
        cell = copy.deepcopy(self.cells["t1"])
        cell["common"][:20, 2] = np.nan
        cell.pop("soh")
        cell.pop("label_cycle")
        values = processor.model_input(cell, 20)
        self.assertEqual(values["history"].shape, (100, 23))
        self.assertEqual(values["length"], 20)
        self.assertEqual(values["statistics"].shape, (48,))
        self.assertTrue(np.isfinite(values["history"]).all())
        self.assertTrue(np.all(values["history"][:20, 11+2] == 0))
        self.assertTrue(np.all(values["history"][20:] == 0))
        self.assertTrue(np.all(values["conditions"][-5:] == 0))

    def test_future_signals_and_labels_do_not_change_prediction_in_small_models(self):
        for architecture in ("gru", "statistics_mlp"):
            with self.subTest(architecture=architecture):
                bundle = self.bundle(architecture)
                cell = copy.deepcopy(self.cells["t1"])
                before = predict_transfer_sequence(bundle, cell, 20, [30, 80, 120], mode=RESEARCH_MODE)
                cell["common"][20:] = np.inf
                cell.pop("soh")
                cell.pop("label_cycle")
                after = predict_transfer_sequence(bundle, cell, 20, [30, 80, 120], mode=RESEARCH_MODE)
                np.testing.assert_array_equal(before, after)

    def test_production_identity_family_and_origin_rejections(self):
        bundle = self.bundle()
        cell = copy.deepcopy(self.cells["t1"])
        with self.assertRaisesRegex(ValueError, "research-only"):
            predict_transfer_sequence(bundle, cell, 20, [30], mode="deployment")
        for kind in ("id", "family", "domain", "schema"):
            cell = copy.deepcopy(self.cells["t1"])
            if kind == "schema":
                cell.pop("input_schema")
                cell.pop("feature_names")
            elif kind == "id":
                cell["cell_id"] = "unknown"
            elif kind == "family":
                cell["physical_family"] = self.plan["source_physical_family"]
            else:
                cell["metadata"]["dataset"] = "MIT"
            with self.assertRaises(ValueError):
                predict_transfer_sequence(bundle, cell, 20, [30], mode=RESEARCH_MODE)
        with self.assertRaisesRegex(ValueError, "origin outside"):
            predict_transfer_sequence(bundle, self.cells["t1"], 60, [80], mode=RESEARCH_MODE)

    def test_source_contexts_weight_each_cell_equally_and_train_one_epoch(self):
        bundle = self.bundle()
        contexts = transfer_training_contexts(bundle, self.cells, cap=96)
        self.assertEqual({r["cell_id"] for r in contexts}, {"s1", "s2"})
        for key in bundle["source_train_ids"]:
            self.assertAlmostEqual(sum(r["weight"] for r in contexts if r["cell_id"] == key), 1.)
        optimizer = torch.optim.AdamW(bundle["network"].parameters(), lr=.001)
        loss = fit_epoch(bundle["network"], contexts, optimizer, 9, 1, 4, "cpu")
        self.assertTrue(np.isfinite(loss))
        self.cells["t2"]["soh"][:] = .01
        report, arrays = evaluate_transfer(bundle, self.cells, ["t1", "t2"], predictor=predict_transfer_sequence)
        self.assertEqual(report["summary"]["100"]["cells"], 2)
        self.assertGreater(next(r["metrics"]["mape_percent"] for r in report["by_cell"] if r["cell_id"] == "t2"), 6)
        self.assertIn("t2__100__prediction", arrays)

    def test_inadequate_history_or_reference_is_coverage_not_error_filtering(self):
        bundle = self.bundle()
        self.cells["t2"]["reference_established_source_cycle"] = 65
        report, _ = evaluate_transfer(bundle, self.cells, ["t1", "t2"], predictor=predict_transfer_sequence)
        self.assertEqual(report["summary"]["20"]["cells"], 1)
        self.assertEqual(report["summary"]["50"]["cells"], 1)
        self.assertEqual(report["summary"]["100"]["cells"], 2)
        self.assertEqual(len(report["skipped"]), 2)
        with self.assertRaisesRegex(ValueError, "insufficient"):
            shared_statistics_cards({**self.cells["t1"], "minimum_signal_observations": 21}, 20)

    def test_cards_use_physical_schema_units_without_capacity_labels_names(self):
        cell = copy.deepcopy(self.cells["t1"])
        cell["common"] = np.tile(np.r_[-np.arange(5)*.03, np.full(5, 2.), 4.], (100, 1))
        cards = shared_statistics_cards(cell, 20)
        text = " ".join(cards)
        self.assertIn("Voltage mean 3.94 V", text)
        self.assertIn("mean 2 C-rate", text)
        self.assertIn("Temperature unavailable in this input protocol", text)
        self.assertNotIn("XJTU", text)
        self.assertNotIn("target-family", text)
        self.assertNotIn("t1", text)
        cell["common"][20:] = np.inf
        cell.pop("soh")
        cell.pop("label_cycle")
        cell["reference_capacity_ah"] = object()
        self.assertEqual(cards, shared_statistics_cards(cell, 20))
        with self.assertRaisesRegex(ValueError, "patch size"):
            shared_statistics_cards(cell, 20, patch_size=3)

    def test_cached_vit_cards_reconstruct_absolute_temperature_and_missingness(self):
        cell = copy.deepcopy(self.cells["t1"])
        cell.pop("input_schema")
        cell.pop("feature_names")
        cell["common"] = np.tile([0, -.1, -.2, 2, 2, 2, 0, 1, 2, 3.9, 26], (100, 1))
        text = " ".join(shared_statistics_cards(cell, 20))
        self.assertIn("Voltage mean 3.8 V", text)
        self.assertIn("Temperature mean 27 degC", text)
        cell["common"][:, 6:9] = np.nan
        cell["common"][:, 10] = np.nan
        self.assertIn("Temperature unavailable.", " ".join(shared_statistics_cards(cell, 20)))

    def test_source_past_100_and_target_shortage_never_modify_preprocessing(self):
        before = self.processor()
        for key in ("s1", "s2"):
            cell = self.cells[key]
            cell["cycle"] = np.arange(1, 151)
            cell["common"] = np.r_[cell["common"], np.full((50, 11), np.inf)]
        after = self.processor()
        np.testing.assert_array_equal(before.raw_scale.scaler.mean_, after.raw_scale.scaler.mean_)
        self.cells["t1"]["common"] = self.cells["t1"]["common"][:2]
        self.cells["t1"]["cycle"] = self.cells["t1"]["cycle"][:2]
        with self.assertRaisesRegex(ValueError, "insufficient"):
            after.model_input(self.cells["t1"], 100)


class SharedLLMOnlyTests(unittest.TestCase):
    setUp = SharedSequenceTests.setUp
    processor = SharedSequenceTests.processor
    bundle = SharedSequenceTests.bundle

    def mock_backbone(self):
        patches = [patch("battery_smetimes.research_llm.AutoConfig.from_pretrained", return_value=SimpleNamespace(hidden_size=16, use_cache=False)),
            patch("battery_smetimes.research_llm.AutoTokenizer.from_pretrained", return_value=TinyTokenizer()),
            patch("battery_smetimes.research_llm.AutoModel.from_pretrained", side_effect=lambda *a, **k: TinyTransferBackbone(k["dtype"])),
            patch("battery_smetimes.research_llm.AutoModel.from_config", side_effect=lambda *a, **k: TinyTransferBackbone(k["dtype"]))]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        return {"checkpoint": "unused-mocked-only", "patch_size": 5, "max_text_tokens": 128,
            "variants": {name: {"pretrained": name == "llm_e1", "use_statistics": True, "experts": 4, "rank": 64}
                         for name in ("llm_e1", "llm_e2")}}

    def test_mocked_frozen_llm_train_has_adapter_gradients_and_target_causal_invariance(self):
        llm = self.mock_backbone()
        for name in ("llm_e1", "llm_e2"):
            with self.subTest(name=name):
                bundle = self.bundle(name, llm)
                contexts = transfer_training_contexts(bundle, self.cells)
                self.assertEqual({r["cell_id"] for r in contexts}, {"s1", "s2"})
                optimizer = torch.optim.AdamW((p for p in bundle["network"].parameters() if p.requires_grad), lr=.001)
                loss = fit_llm_epoch(bundle["network"], contexts, optimizer, 9, 1, 4, "cpu", .01)
                self.assertTrue(np.isfinite(loss))
                self.assertGreater(float(bundle["network"].numeric_projection[0].weight.grad.norm()), 0.)
                self.assertTrue(all(p.grad is None for p in bundle["network"].backbone.parameters()))
                cell = copy.deepcopy(self.cells["t1"])
                before = predict_transfer_sequence(bundle, cell, 20, [30, 80], mode=RESEARCH_MODE)
                cell["common"][20:] = np.inf
                cell.pop("soh")
                cell.pop("label_cycle")
                bundle["card_cache"].clear()  # recompute, not a stale-cache pass
                after = predict_transfer_sequence(bundle, cell, 20, [30, 80], mode=RESEARCH_MODE)
                np.testing.assert_array_equal(before, after)
                self.assertEqual(bundle["allowed_cells"]["t1"]["physical_family"], "target-family")

    def test_e1_e2_keep_matched_adapter_initialization_and_separate_caches(self):
        llm = self.mock_backbone()
        first, second = self.bundle("llm_e1", llm), self.bundle("llm_e2", llm)
        for key, tensor in adapter_state(first["network"]).items():
            torch.testing.assert_close(tensor, adapter_state(second["network"])[key], rtol=0, atol=0)
        transfer_training_contexts(first, self.cells)
        self.assertNotIn("card_cache", second)
        with self.assertRaisesRegex(ValueError, "explicitly frozen"):
            self.bundle("llm_e1")
        llm["variants"]["llm_e2"]["pretrained"] = True
        with self.assertRaisesRegex(ValueError, "preserve"):
            self.bundle("llm_e2", llm)


if __name__ == "__main__":
    unittest.main()
