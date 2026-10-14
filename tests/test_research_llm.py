from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from battery_smetimes.research_llm import (FrozenSMETimesForecaster, adapter_state,
    load_adapter, statistics_cards, integrated_soh, historical_rates, basis_soh)
from battery_smetimes.research_sequences import HistoryPreprocessor


class TinyBackbone(nn.Module):
    def __init__(self, consume=0):
        super().__init__()
        torch.rand(consume)
        self.config = SimpleNamespace(hidden_size=16)
        self.layer = nn.Linear(16, 16)

    def forward(self, inputs_embeds=None, attention_mask=None, **kwargs):
        return SimpleNamespace(last_hidden_state=self.layer(inputs_embeds))


class ResearchLLMTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.mocks = [patch("battery_smetimes.research_llm.AutoConfig.from_pretrained",
                            return_value=SimpleNamespace(hidden_size=16, use_cache=False)),
                      patch("battery_smetimes.research_llm.AutoTokenizer.from_pretrained",
                            return_value=SimpleNamespace(pad_token_id=0)),
                      patch("battery_smetimes.research_llm.AutoModel.from_pretrained",
                            side_effect=lambda *a, **k: TinyBackbone(5)),
                      patch("battery_smetimes.research_llm.AutoModel.from_config",
                            side_effect=lambda *a, **k: TinyBackbone(15))]
        for mock in self.mocks:
            mock.start()
            self.addCleanup(mock.stop)

    def model(self, **kwargs):
        return FrozenSMETimesForecaster("not-used", 3, device="cpu", dtype=torch.float32,
                                       rank=4, dropout=0, **kwargs)

    def test_cards_ignore_future_labels_and_cell_dataset_names(self):
        rng = np.random.default_rng(2)
        cell = {"cycle": np.arange(1, 151), "raw": rng.normal(size=(150, 51)),
                "soh": np.ones(150), "label_cycle": np.arange(150),
                "metadata": {"dataset": "MIT", "cell_id": "SECRET_CELL"}}
        before = statistics_cards(cell, 20)
        cell["raw"][20:] = np.inf
        del cell["soh"], cell["label_cycle"]
        self.assertEqual(before, statistics_cards(cell, 20))
        self.assertEqual(len(before), 4)
        self.assertNotIn("MIT", " ".join(before))
        self.assertNotIn("SECRET_CELL", " ".join(before))

    def test_basis_head_query_consistency_restore_and_gradient(self):
        model=self.model(output_head='basis_rate').eval()
        history=torch.randn(2,100,3);length=torch.tensor([7,20]);cards=torch.randn(2,20,16)
        encoded=model.encode(history,length,cards)
        conditions=torch.randn(2,9)
        queries=torch.zeros(2,5,3);queries[:,:,1]=torch.tensor([0.,.5,1.,4.,10.])
        prediction=model.decode(encoded,conditions,queries)
        torch.testing.assert_close(prediction[:,2:3],model.decode(encoded,conditions,queries[:,2:3]))
        prediction.square().mean().backward()
        self.assertGreater(float(model.numeric_projection[0].weight.grad.norm()),0)
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
        restored=self.model(output_head='basis_rate').eval()
        load_adapter(restored,adapter_state(model))
        torch.testing.assert_close(prediction,restored.decode(restored.encode(history,length,cards),conditions,queries))

    def test_cards_ignore_future_array_dtype_promotion(self):
        rng = np.random.default_rng(9)
        visible = rng.normal(size=(20, 51)).astype(np.float32)
        cell = {'cycle': np.arange(1, 21), 'raw': visible,
                'metadata': {'dataset': 'MIT'}}
        promoted = {**cell, 'cycle': np.arange(1, 23),
                    'raw': np.r_[visible, np.full((2, 51), np.inf)]}
        self.assertEqual(statistics_cards(cell, 20), statistics_cards(promoted, 20))

    def test_basis_physical_shape_and_no_tail_floor(self):
        p=torch.tensor([[0.,-2.]+[-4.]*11+[0.,0.]],requires_grad=True)
        t=torch.linspace(0.,30.,301)[None]
        y=basis_soh(p,t)
        self.assertAlmostEqual(float(y[0,0].detach()),1.)
        self.assertTrue(bool(torch.all(y[:,1:]<y[:,:-1])))
        self.assertLess(float(y[0,-1].detach()),float(y[0,-2].detach()))
        # Early transition coefficients receive signal on a short forecast.
        loss=(basis_soh(p,t[:,:16])-(1.-.08*t[:,:16])).square().mean()
        gradient=torch.autograd.grad(loss,p)[0]
        self.assertGreater(float(gradient[0,2:5].abs().sum()),1e-6)

    def test_random_control_includes_backbone_and_matched_adapter_init(self):
        pretrained = self.model(pretrained=True)
        random = self.model(pretrained=False)
        for key, value in adapter_state(pretrained).items():
            torch.testing.assert_close(value, adapter_state(random)[key], rtol=0, atol=0)
        self.assertFalse(torch.equal(pretrained.backbone.layer.weight, random.backbone.layer.weight))
        self.assertTrue(all(not p.requires_grad for p in random.backbone.parameters()))

    def test_partial_last_patch_ignores_padding(self):
        model = self.model().eval()
        history = torch.randn(1, 100, 3)
        changed = history.clone()
        changed[:, 7:] = 100
        cards = torch.randn(1, 20, 16)
        with torch.no_grad():
            before = model.encode(history, torch.tensor([7]), cards)
            after = model.encode(changed, torch.tensor([7]), cards)
        torch.testing.assert_close(before, after)

    def test_knee_initialization_changes_only_bias_and_restores_exactly(self):
        control=self.model(output_head='integrated_rate')
        candidate=self.model(output_head='integrated_rate',initial_knee_bias=-1.)
        for key,value in adapter_state(control).items():
            other=adapter_state(candidate)[key]
            if key=='head.3.bias':
                self.assertEqual(float(other[3]),-1.)
                other=other.clone();other[3]=value[3]
            torch.testing.assert_close(value,other,rtol=0,atol=0)
        load_adapter(control,adapter_state(candidate))
        encoding=torch.randn(2,16);conditions=torch.randn(2,9);queries=torch.randn(2,5,3)
        torch.testing.assert_close(control.decode(encoding,conditions,queries),candidate.decode(encoding,conditions,queries),rtol=0,atol=0)
        with self.assertRaises(ValueError):self.model(output_head='integrated_rate',initial_knee_bias=float('nan'))

    def test_earlier_initial_knee_improves_short_horizon_gradient(self):
        time=torch.linspace(.01,1.5,80)[None]
        target=1.-.08*time
        norms=[]
        for bias in (1.,-1.):
            parameters=torch.tensor([[0.,-2.,-2.,bias,-1.,0.,0.]],requires_grad=True)
            loss=(integrated_soh(parameters,time)-target).square().mean()
            gradient=torch.autograd.grad(loss,parameters)[0]
            norms.append(float(gradient[0,3].abs()))
        self.assertGreater(norms[1],norms[0]*100)

    def test_router_normalization_is_single_factor_with_gradient(self):
        original = self.model()
        normalized = self.model(normalize_router=True)
        for key, value in adapter_state(original).items():
            torch.testing.assert_close(value, adapter_state(normalized)[key], rtol=0, atol=0)
        captured = []
        handle = normalized.gate.register_forward_pre_hook(lambda m, args: captured.append(args[0].detach()))
        encoded = normalized.encode(torch.randn(2,100,3),torch.tensor([7,20]),torch.randn(2,20,16))
        handle.remove()
        torch.testing.assert_close(captured[0].mean(-1),torch.zeros(2,20),atol=1e-6,rtol=0)
        prediction = normalized.decode(encoded,torch.randn(2,9),torch.randn(2,4,3))
        (prediction.square().mean()+.01*normalized.balance_loss).backward()
        self.assertGreater(float(normalized.numeric_projection[0].weight.grad.norm()),0)
        self.assertGreater(float(normalized.gate.weight.grad.norm()),0)
        self.assertTrue(all(p.grad is None for p in normalized.backbone.parameters()))

    def test_attention_pooling_matches_mean_at_init_and_masks_padding(self):
        mean=self.model(output_head="integrated_rate").eval()
        attention=self.model(output_head="integrated_rate",pooling="attention").eval()
        for key,value in adapter_state(mean).items():
            torch.testing.assert_close(value,adapter_state(attention)[key],rtol=0,atol=0)
        history=torch.randn(2,100,3);length=torch.tensor([7,20]);cards=torch.randn(2,20,16)
        encoded=attention.encode(history,length,cards)
        torch.testing.assert_close(encoded,mean.encode(history,length,cards))
        changed=history.clone();changed[0,7:]=100;changed[1,20:]=100
        torch.testing.assert_close(encoded,attention.encode(changed,length,cards))
        prediction=attention.decode(encoded,torch.randn(2,9),torch.ones(2,5,3))
        prediction.square().mean().backward()
        self.assertGreater(float(attention.pool_score.weight.grad.norm()),0)
        self.assertGreater(float(attention.numeric_projection[0].weight.grad.norm()),0)
        restored=self.model(output_head="integrated_rate",pooling="attention").eval()
        load_adapter(restored,adapter_state(attention))
        torch.testing.assert_close(restored.encode(history,length,cards),encoded)

    def test_adaptive_fusion_matches_control_and_learns_through_llm(self):
        control=self.model(output_head='integrated_rate').eval()
        model=self.model(output_head='integrated_rate',adaptive_fusion=True).eval()
        for key,value in adapter_state(control).items():
            torch.testing.assert_close(value,adapter_state(model)[key],rtol=0,atol=0)
        history=torch.randn(2,100,3);length=torch.tensor([7,20]);cards=torch.randn(2,20,16)
        encoded=model.encode(history,length,cards)
        torch.testing.assert_close(encoded,control.encode(history,length,cards))
        model.decode(encoded,torch.randn(2,9),torch.ones(2,3,3)).square().mean().backward()
        self.assertGreater(float(model.fusion_gate.weight.grad.norm()),0)
        self.assertGreater(float(model.numeric_projection[0].weight.grad.norm()),0)
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
        with torch.no_grad(): model.fusion_gate.weight.copy_(torch.randn_like(model.fusion_gate.weight)*.1)
        expected=model.encode(history,length,cards)
        self.assertGreater(float((expected-encoded).detach().abs().max()),1e-6)
        changed=history.clone();changed[0,7:]=100;changed[1,20:]=100
        changed_cards=cards.clone();changed_cards[0,2:]=100;changed_cards[1,4:]=100
        torch.testing.assert_close(expected,model.encode(changed,length,changed_cards))
        restored=self.model(output_head='integrated_rate',adaptive_fusion=True).eval()
        load_adapter(restored,adapter_state(model))
        torch.testing.assert_close(expected,restored.encode(history,length,cards))
        with self.assertRaises(ValueError): self.model(use_statistics=False,adaptive_fusion=True)

    def test_dual_scale_preserves_control_init_and_padding_restore(self):
        control=self.model(output_head='integrated_rate').eval()
        model=self.model(output_head='integrated_rate',pooling='dual_scale').eval()
        for key,value in adapter_state(control).items():
            torch.testing.assert_close(value,adapter_state(model)[key],rtol=0,atol=0)
        history=torch.randn(3,100,3);length=torch.tensor([3,7,20]);cards=torch.randn(3,20,16)
        encoded=model.encode(history,length,cards)
        torch.testing.assert_close(encoded,control.encode(history,length,cards))
        model.decode(encoded,torch.randn(3,9),torch.ones(3,3,3)).square().mean().backward()
        self.assertGreater(float(model.temporal_contrast[2].weight.grad.norm()),0)
        self.assertGreater(float(model.numeric_projection[0].weight.grad.norm()),0)
        with torch.no_grad():model.temporal_contrast[2].weight.copy_(torch.randn_like(model.temporal_contrast[2].weight)*.1)
        expected=model.encode(history,length,cards)
        # With only one observed patch, no artificial early/late contrast.
        torch.testing.assert_close(expected[0],encoded[0])
        self.assertGreater(float((expected[1:]-encoded[1:]).detach().abs().max()),1e-6)
        changed=history.clone();changed_cards=cards.clone()
        for i,n in enumerate(length):
            changed[i,n:]=100;changed_cards[i,(int(n)+4)//5:]=100
        torch.testing.assert_close(expected,model.encode(changed,length,changed_cards))
        restored=self.model(output_head='integrated_rate',pooling='dual_scale').eval()
        load_adapter(restored,adapter_state(model))
        torch.testing.assert_close(expected,restored.encode(history,length,cards))

    def test_summary_projection_enters_llm_masks_and_restores(self):
        control=self.model(output_head='integrated_rate').eval()
        model=self.model(output_head='integrated_rate',summary_width=7).eval()
        for key,value in adapter_state(control).items():
            torch.testing.assert_close(value,adapter_state(model)[key],rtol=0,atol=0)
        history=torch.randn(2,100,3);length=torch.tensor([7,20]);cards=torch.randn(2,20,16);stats=torch.randn(2,7)
        encoded=model.encode(history,length,cards,stats)
        torch.testing.assert_close(encoded,control.encode(history,length,cards))
        model.decode(encoded,torch.randn(2,9),torch.ones(2,3,3)).square().mean().backward()
        self.assertGreater(float(model.summary_projection.weight.grad.norm()),0)
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
        with torch.no_grad():model.summary_projection.weight.copy_(torch.randn_like(model.summary_projection.weight)*.1)
        expected=model.encode(history,length,cards,stats)
        self.assertGreater(float((expected-model.encode(history,length,cards,torch.zeros_like(stats))).detach().abs().max()),1e-6)
        changed=history.clone();changed[0,7:]=100;changed[1,20:]=100
        torch.testing.assert_close(expected,model.encode(changed,length,cards,stats))
        restored=self.model(output_head='integrated_rate',summary_width=7).eval()
        load_adapter(restored,adapter_state(model))
        torch.testing.assert_close(expected,restored.encode(history,length,cards,stats))
        with self.assertRaises(ValueError):model.encode(history,length,cards)

    def test_historical_rates_use_actual_gaps_and_missing_masks(self):
        history=torch.tensor([[[1.,1.,.01],[3.,1.,.02],[7.,1.,.06],[99.,0.,.07],[9.,1.,.08],[100.,1.,1.]]])
        result=historical_rates(history,torch.tensor([5]))
        torch.testing.assert_close(result[0,:,0],torch.tensor([0.,2.,1.,0.,0.,0.]))
        torch.testing.assert_close(result[0,:,1],torch.tensor([0.,1.,1.,0.,0.,0.]))
        self.assertAlmostEqual(float(result[0,2,2]),.04,places=6)
        self.assertEqual(float(result[0,5].abs().sum()),0)
        bad=history.clone();bad[0,2,2]=.01
        with self.assertRaises(ValueError):historical_rates(bad,torch.tensor([5]))

    def test_head_capacity_diagnostic_matches_actual_decoder(self):
        from scripts.diagnose_llm_head_capacity import numpy_trajectory
        rng=np.random.default_rng(777)
        for _ in range(5):
            parameters=rng.normal(size=7);time=np.array([0.,.01,.5,2.,8.,20.])
            expected=integrated_soh(torch.tensor(parameters[None]),torch.tensor(time[None])).numpy()[0]
            np.testing.assert_allclose(numpy_trajectory(parameters,time),expected,atol=1e-12,rtol=1e-12)

    def test_dynamics_adapter_matches_initial_control_has_grad_and_masks_padding(self):
        control=self.model(output_head='integrated_rate').eval()
        model=self.model(output_head='integrated_rate',numeric_dynamics=True).eval()
        for key,value in adapter_state(control).items():
            torch.testing.assert_close(value,adapter_state(model)[key],rtol=0,atol=0)
        history=torch.zeros(1,100,3)
        history[0,:7,0]=torch.arange(7.)**2
        history[0,:7,1]=1;history[0,:7,2]=torch.arange(1.,8.)/100
        cards=torch.randn(1,20,16);length=torch.tensor([7])
        encoding=model.encode(history,length,cards)
        torch.testing.assert_close(encoding,control.encode(history,length,cards))
        model.decode(encoding,torch.randn(1,9),torch.ones(1,3,3)).square().mean().backward()
        self.assertGreater(float(model.dynamics_projection.weight.grad.norm()),0)
        self.assertGreater(float(model.numeric_projection[0].weight.grad.norm()),0)
        with torch.no_grad():model.dynamics_projection.weight.fill_(.01)
        changed=history.clone();changed[:,7:]=100
        torch.testing.assert_close(model.encode(history,length,cards),model.encode(changed,length,cards))
        restored=self.model(output_head='integrated_rate',numeric_dynamics=True).eval()
        load_adapter(restored,adapter_state(model))
        torch.testing.assert_close(model.encode(history,length,cards),restored.encode(history,length,cards))

    def test_queries_are_independent_of_requested_endpoint(self):
        model = self.model().eval()
        encoding = model.encode(torch.randn(1,100,3),torch.tensor([20]),torch.randn(1,20,16))
        conditions = torch.randn(1,9)
        query = torch.randn(1,5,3)
        with torch.no_grad():
            whole = model.decode(encoding,conditions,query)
            single = model.decode(encoding,conditions,query[:,2:3])
        torch.testing.assert_close(whole[:,2:3],single,atol=1e-6,rtol=1e-6)

    def test_integrated_head_anchor_monotonic_rate_and_recovery_bound(self):
        parameters=torch.tensor([[0.,-2.,0.,0.,-1.,0.,0.]],requires_grad=True)
        time=torch.linspace(0,20,201)[None]
        prediction=integrated_soh(parameters,time)
        self.assertEqual(float(prediction[0,0].detach()),1.)
        self.assertTrue(torch.all(torch.diff(prediction)<0))
        self.assertLess(float(torch.diff(prediction)[0,-1].detach()),float(torch.diff(prediction)[0,0].detach()))
        changed=parameters.detach().clone();changed[:,5]=100
        self.assertLessEqual(float((integrated_soh(changed,time)-prediction.detach()).abs().max()),.005001)
        prediction.square().mean().backward()
        self.assertTrue(torch.isfinite(parameters.grad).all())

    def test_integrated_head_query_invariance_and_scaler_checkpoint(self):
        model=self.model(output_head="integrated_rate").eval()
        model.bind_query_scaler(SimpleNamespace(mean_=np.array([2.,3.,4.]),scale_=np.array([1.,2.,3.])))
        encoding=model.encode(torch.randn(1,100,3),torch.tensor([20]),torch.randn(1,20,16))
        conditions=torch.randn(1,9);query=torch.randn(1,6,3)
        whole=model.decode(encoding,conditions,query)
        auxiliary_prediction,parameters=model.decode(encoding,conditions,query,return_parameters=True)
        torch.testing.assert_close(auxiliary_prediction,whole,atol=0,rtol=0)
        physical=query*model.query_scale+model.query_mean
        torch.testing.assert_close(auxiliary_prediction,integrated_soh(parameters,physical[...,1].clamp_min(0)),atol=0,rtol=0)
        single=model.decode(encoding,conditions,query[:,2:3])
        torch.testing.assert_close(whole[:,2:3],single,atol=1e-6,rtol=1e-6)
        restored=self.model(output_head="integrated_rate").eval()
        load_adapter(restored,adapter_state(model))
        torch.testing.assert_close(restored.decode(encoding,conditions,query),whole)
        whole.square().mean().backward()
        self.assertGreater(float(model.numeric_projection[0].weight.grad.norm()),0)
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))

    def test_preprocessor_fits_training_only_and_ignores_future(self):
        rng=np.random.default_rng(4)
        cells={key:{"cycle":np.arange(1,151),"raw":rng.normal(size=(150,51)),
            "label_cycle":np.arange(1,151),"soh":np.linspace(1,.8,150),
            "reference_established_source_cycle":1,"metadata":{"dataset":"MIT","group":"batch1",
            "charge_protocol":"4C(80%)-4C","diagnostic_discharge_rate_c":4}} for key in ("train","heldout")}
        processor=HistoryPreprocessor().fit(cells,["train"],[100])
        before=processor.model_input(cells["heldout"],100)
        cells["heldout"]["raw"][100:]=np.inf
        del cells["heldout"]["soh"],cells["heldout"]["label_cycle"]
        after=processor.model_input(cells["heldout"],100)
        for field in ("history","conditions","statistics"):
            np.testing.assert_array_equal(before[field],after[field])
        cells["heldout"]["raw"][:100]=1e9
        other=HistoryPreprocessor().fit(cells,["train"],[100])
        np.testing.assert_array_equal(processor.raw_scale.scaler.mean_,other.raw_scale.scaler.mean_)

    def test_input_gradient_and_real_balance_loss_with_frozen_backbone(self):
        model = self.model()
        history = torch.randn(2, 100, 3)
        encoded = model.encode(history, torch.tensor([7, 20]), torch.randn(2, 20, 16))
        prediction = model.decode(encoded, torch.randn(2, 9), torch.randn(2, 4, 3))
        (prediction.square().mean() + .01 * model.balance_loss).backward()
        self.assertGreater(float(model.numeric_projection[0].weight.grad.norm()), 0)
        self.assertGreater(float(model.gate.weight.grad.norm()), 0)
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
        self.assertGreater(float(model.balance_loss.detach()), 0)

    def test_residual_experts_use_llm_keep_gradients_and_padding(self):
        model = self.model(residual_experts=True).eval()
        history = torch.randn(2, 100, 3)
        lengths = torch.tensor([7, 50])
        cards = torch.randn(2, 20, 16)
        encoded = model.encode(history, lengths, cards)
        changed = history.clone()
        changed[0, 7:] = 100
        changed[1, 50:] = 100
        torch.testing.assert_close(encoded, model.encode(changed, lengths, cards))
        control = self.model().eval()
        load_adapter(control, adapter_state(model))
        self.assertFalse(torch.allclose(encoded, control.encode(history, lengths, cards)))
        query = torch.randn(2, 4, 3)
        conditions = torch.randn(2, 9)
        model.decode(encoded, conditions, query).square().mean().backward()
        self.assertGreater(float(model.numeric_projection[0].weight.grad.norm()), 0)
        self.assertGreater(float(model.experts[0][0].weight.grad.norm()), 0)
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
        restored = self.model(residual_experts=True).eval()
        load_adapter(restored, adapter_state(model))
        torch.testing.assert_close(encoded, restored.encode(history, lengths, cards))

    def test_extended_history_and_cards_capacity_masks_real_length(self):
        model = self.model(history_slots=150).eval()
        history = torch.randn(2, 150, 3)
        length = torch.tensor([7, 143])
        cards = torch.randn(2, 30, 16)
        expected = model.encode(history, length, cards)
        changed = history.clone()
        changed[0, 7:] = 100
        changed[1, 143:] = 100
        changed_cards = cards.clone()
        changed_cards[0, 2:] = 100
        changed_cards[1, 29:] = 100
        torch.testing.assert_close(expected, model.encode(changed, length, changed_cards))
        short = self.model().eval()
        load_adapter(short, adapter_state(model))
        torch.testing.assert_close(expected[:1], short.encode(history[:1, :100], length[:1], cards[:1, :20]))
        with self.assertRaises(ValueError):
            model.encode(history, torch.tensor([0, 151]), cards)
        with self.assertRaises(ValueError):
            self.model(history_slots=103)
        cell = {'cycle': np.arange(1, 201), 'raw': np.ones((200, 51)), 'metadata': {'dataset': 'MIT'}}
        text = statistics_cards(cell, 150, history_slots=150)
        self.assertEqual(len(text), 30)
        cell['raw'][150:] = np.inf
        self.assertEqual(text, statistics_cards(cell, 150, history_slots=150))
        with self.assertRaises(ValueError):
            statistics_cards(cell, 150)

    def test_history_preprocessor_capacity_and_legacy_restore(self):
        class IdentityScale:
            def transform(self, x):
                return np.nan_to_num(x).astype(np.float32)
        processor = HistoryPreprocessor()
        processor.raw_width = 2
        processor.history_slots = 150
        processor.raw_scale = IdentityScale()
        processor.stats_scale = IdentityScale()
        cell = {'cycle': np.arange(1, 201), 'raw': np.ones((200, 2))}
        with patch('battery_smetimes.research_sequences.signal_prefix', return_value=np.ones(12)):
            values = processor.model_input(cell, 143)
            self.assertEqual(values['history'].shape, (150, 5))
            self.assertEqual(values['length'], 143)
            np.testing.assert_array_equal(values['history'][143:], 0)
            cell['raw'][143:] = np.inf
            np.testing.assert_array_equal(values['history'], processor.model_input(cell, 143)['history'])
            del processor.history_slots
            self.assertEqual(processor.model_input(cell, 50)['history'].shape, (100, 5))
            with self.assertRaises(ValueError):
                processor.model_input(cell, 143)

    def test_extra_storage_does_not_add_backbone_padding_tokens(self):
        model = self.model(history_slots=200).eval()
        history = torch.randn(1, 200, 3)
        cards = torch.randn(1, 40, 16)
        seen = []
        hook = model.numeric_projection.register_forward_pre_hook(
            lambda module, inputs: seen.append(inputs[0].shape[1]))
        try:
            model.encode(history, torch.tensor([99]), cards)
            model.encode(history, torch.tensor([149]), cards)
            model.encode(history, torch.tensor([197]), cards)
        finally:
            hook.remove()
        self.assertEqual(seen, [20, 30, 40])

    def test_adapter_checkpoint_leaves_backbone_unchanged(self):
        model = self.model()
        before = model.backbone.layer.weight.detach().clone()
        state = adapter_state(model)
        state["head.3.bias"].add_(1)
        load_adapter(model, state)
        torch.testing.assert_close(model.backbone.layer.weight, before)
        self.assertFalse(any(k.startswith("backbone.") for k in state))

    def test_one_expert_balance_is_zero_but_main_four_experts_is_not_constant(self):
        model = self.model(experts=1)
        model.encode(torch.randn(1, 100, 3), torch.tensor([20]), torch.randn(1, 20, 16))
        self.assertEqual(float(model.balance_loss.detach()), 0)


if __name__ == "__main__":
    unittest.main()
