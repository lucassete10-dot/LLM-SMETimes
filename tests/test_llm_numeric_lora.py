"""Numeric-only adaptation: base weights, text cache and checkpoint contracts."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
import numpy as np
from torch import nn
from battery_smetimes.research_llm import FrozenSMETimesForecaster,adapter_state,load_adapter,adapter_optimizer
from scripts.verify_llm_soh_mvp import audit_numeric_lora


class TinyAttentionBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config=SimpleNamespace(hidden_size=16)
        attention=nn.Module()
        attention.q_proj=nn.Linear(16,16)
        attention.v_proj=nn.Linear(16,16)
        layer=nn.Module();layer.self_attn=attention
        self.layers=nn.ModuleList([layer])

    def forward(self,inputs_embeds=None,input_ids=None,**kwargs):
        x=inputs_embeds if inputs_embeds is not None else torch.nn.functional.one_hot(input_ids,16).float()
        a=self.layers[-1].self_attn
        return SimpleNamespace(last_hidden_state=torch.tanh(a.q_proj(x))+a.v_proj(x))


class Tokens(dict):
    def to(self,device):return Tokens({k:v.to(device) for k,v in self.items()})


class Tokenizer:
    pad_token_id=0
    def __call__(self,texts,**kwargs):
        return Tokens(input_ids=torch.ones(len(texts),3,dtype=torch.long),attention_mask=torch.ones(len(texts),3))


class NumericLoRATests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        for target,options in (
            ('AutoConfig.from_pretrained',{'return_value':SimpleNamespace(hidden_size=16,use_cache=False)}),
            ('AutoTokenizer.from_pretrained',{'return_value':Tokenizer()}),
            ('AutoModel.from_pretrained',{'side_effect':lambda *a,**k:TinyAttentionBackbone()})):
            mock=patch('battery_smetimes.research_llm.'+target,**options)
            mock.start();self.addCleanup(mock.stop)

    def model(self,rank=0):
        return FrozenSMETimesForecaster('unused',3,device='cpu',dtype=torch.float32,
            rank=4,dropout=0,output_head='integrated_rate',numeric_lora_rank=rank)

    def test_initial_match_gradient_base_freeze_text_stability_and_reload(self):
        control=self.model();model=self.model(4)
        for key,value in adapter_state(control).items():
            torch.testing.assert_close(value,adapter_state(model)[key],rtol=0,atol=0)
        history=torch.randn(2,100,3);length=torch.tensor([10,20]);cards=torch.randn(2,20,16)
        torch.testing.assert_close(control.encode(history,length,cards),model.encode(history,length,cards),rtol=0,atol=0)
        base={k:p.detach().clone() for k,p in model.backbone.named_parameters() if not p.requires_grad}
        text_before=model.encode_cards([['card']])
        encoded=model.encode(history,length,cards)
        prediction=model.decode(encoded,torch.ones(2,9),torch.ones(2,5,3))
        prediction.square().mean().backward()
        gradients={k:p.grad for k,p in model.backbone.named_parameters() if p.requires_grad}
        self.assertTrue(any(float(g.norm())>0 for k,g in gradients.items() if k.endswith('lora_B')))
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients.values()))
        self.assertGreater(float(model.numeric_projection[0].weight.grad.norm()),0)
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters() if not p.requires_grad))
        torch.optim.SGD((p for p in model.parameters() if p.requires_grad),lr=.1).step()
        for k,p in model.backbone.named_parameters():
            if k in base:torch.testing.assert_close(p,base[k],rtol=0,atol=0)
        torch.testing.assert_close(model.encode_cards([['card']]),text_before,rtol=0,atol=0)
        self.assertTrue(model.backbone.layers[-1].self_attn.q_proj.enabled)
        state=adapter_state(model)
        self.assertEqual(sum(k.startswith('backbone.') for k in state),4)
        restored=self.model(4);load_adapter(restored,state)
        torch.testing.assert_close(model.encode(history,length,cards),restored.encode(history,length,cards),rtol=0,atol=0)
        incomplete=dict(state);del incomplete['backbone.layers.0.self_attn.q_proj.lora_A']
        with self.assertRaises(ValueError):load_adapter(restored,incomplete)
        with self.assertRaises(ValueError):load_adapter(control,state)

    def test_text_adapter_switch_restores_even_on_exception(self):
        model=self.model(4)
        layer=model.backbone.layers[-1].self_attn.q_proj
        with self.assertRaises(RuntimeError):
            with model.base_text_encoding():
                self.assertFalse(layer.enabled)
                raise RuntimeError('diagnostic')
        self.assertTrue(layer.enabled)

    def test_lr_grouping_only_slows_lora_and_contains_each_parameter_once(self):
        model=self.model(4)
        optimizer=adapter_optimizer(model,.0005,.03,.1)
        self.assertEqual([g['lr'] for g in optimizer.param_groups],[.0005,.00005])
        ids=[id(p) for g in optimizer.param_groups for p in g['params']]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(set(ids),{id(p) for p in model.parameters() if p.requires_grad})
        self.assertEqual({id(p) for p in optimizer.param_groups[1]['params']},
            {id(p) for p in model.backbone.parameters() if p.requires_grad})
        self.assertEqual(len(adapter_optimizer(self.model(),.0005,.03).param_groups),1)
        with self.assertRaises(ValueError):adapter_optimizer(self.model(),.0005,.03,.1)

    def test_runtime_audit_uses_registry_id_not_metadata_id_and_restores(self):
        model=self.model(4)
        bundle={'network':model,'processor':SimpleNamespace(train_ids=['TRAIN_CELL'])}
        cell={'metadata':{'dataset':'MIT'},'cycle':np.arange(1,11),'raw':np.ones((10,51))}
        originals=adapter_state(model)
        def prediction(bundle,cell,origin,query):
            value=sum(float(p.detach().sum()) for k,p in model.backbone.named_parameters() if k.endswith('lora_B'))
            return np.full(len(query),value)
        with patch('scripts.verify_llm_soh_mvp.predict_llm',side_effect=prediction):
            result=audit_numeric_lora(bundle,cell,10)
        self.assertTrue(result['passed']);self.assertEqual(result['train_cell'],'TRAIN_CELL')
        for key,value in originals.items():torch.testing.assert_close(adapter_state(model)[key],value,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
