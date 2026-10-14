import unittest
from unittest.mock import patch
from types import SimpleNamespace
import torch

from battery_smetimes.llm_query_rate import QueryRateForecaster,integrate_linear_rates
from battery_smetimes.research_llm import adapter_state,load_adapter
from tests.test_research_llm import TinyBackbone


class PhysicalRateTests(unittest.TestCase):
    def test_known_integral_acceleration_slowdown_and_extrapolation(self):
        knots=torch.tensor([0.,1.,2.],dtype=torch.float64)
        rates=torch.tensor([[.02,.02,.02],[0.,.02,.04],[.04,.02,0.]],dtype=torch.float64,requires_grad=True)
        time=torch.tensor([[0.,.5,1.,1.5,2.,3.]]*3,dtype=torch.float64)
        expected=torch.tensor([[0.,.01,.02,.03,.04,.06],
            [0.,.0025,.01,.0225,.04,.08],[0.,.0175,.03,.0375,.04,.04]],dtype=torch.float64)
        value=integrate_linear_rates(rates,knots,time)
        torch.testing.assert_close(value,expected,atol=1e-12,rtol=0)
        value.sum().backward();self.assertTrue(torch.isfinite(rates.grad).all())
        subset=time[:,[5,1,3]]
        torch.testing.assert_close(integrate_linear_rates(rates,knots,subset),value[:,[5,1,3]],rtol=0,atol=0)

    def test_invalid_coordinates_raise(self):
        with self.assertRaises(ValueError):integrate_linear_rates(torch.ones(1,2),torch.tensor([1.,2.]),torch.ones(1,1))
        with self.assertRaises(ValueError):integrate_linear_rates(torch.ones(1,2),torch.tensor([0.,1.]),torch.tensor([[-1.]]))


class QueryDecoderTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        mocks=[patch('battery_smetimes.research_llm.AutoConfig.from_pretrained',return_value=SimpleNamespace(hidden_size=16,use_cache=False)),
            patch('battery_smetimes.research_llm.AutoTokenizer.from_pretrained',return_value=SimpleNamespace(pad_token_id=0)),
            patch('battery_smetimes.research_llm.AutoModel.from_pretrained',side_effect=lambda *a,**k:TinyBackbone())]
        for m in mocks:m.start();self.addCleanup(m.stop)

    def model(self,**kwargs):
        return QueryRateForecaster('test-only',3,device='cpu',dtype=torch.float32,rank=4,dropout=0,rate_width=16,**kwargs)

    def test_normalized_router_receives_unit_variance_and_finite_gradients(self):
        model=self.model(normalize_router=True).eval();seen=[]
        hook=model.gate.register_forward_pre_hook(lambda module,args:seen.append(args[0]))
        self.addCleanup(hook.remove)
        history=torch.randn(1,100,3);cards=torch.randn(1,20,16)
        encoding=model.encode(history,torch.tensor([20]),cards)
        routed=seen[-1]
        torch.testing.assert_close(routed.mean(-1),torch.zeros_like(routed[...,0]),atol=1e-5,rtol=0)
        # LayerNorm's epsilon makes the variance slightly smaller than one.
        self.assertTrue(torch.all((routed.var(-1,unbiased=False)>.99)&(routed.var(-1,unbiased=False)<1.001)))
        queries=torch.tensor([[[0.,1.,0.],[0.,5.,0.]]])
        model.decode(encoding,torch.randn(1,9),queries).sum().backward()
        self.assertTrue(torch.isfinite(model.gate.weight.grad).all())
        self.assertGreater(float(model.gate.weight.grad.norm()),0)

    def test_learned_nodes_receive_gradients_and_preserve_query_invariance(self):
        model=self.model(learned_node_queries=True).eval()
        history=torch.randn(1,100,3);length=torch.tensor([20]);cards=torch.randn(1,20,16)
        encoding=model.encode(history,length,cards);conditions=torch.randn(1,9)
        queries=torch.zeros(1,5,3);queries[:,:,1]=torch.tensor([0.,.3,2.,12.,35.])
        y=model.decode(encoding,conditions,queries)
        torch.testing.assert_close(y[:,[4,1]],model.decode(encoding,conditions,queries[:,[4,1]]),rtol=0,atol=0)
        y.sum().backward()
        self.assertGreater(float(model.node_query.grad.norm()),0)
        self.assertGreater(float(model.numeric_projection[0].weight.grad.norm()),0)
        restored=self.model(learned_node_queries=True).eval();load_adapter(restored,adapter_state(model))
        torch.testing.assert_close(y,restored.decode(restored.encode(history,length,cards),conditions,queries),rtol=0,atol=0)

    def test_mask_gradient_restore_and_query_independence(self):
        model=self.model().eval();history=torch.randn(2,100,3);length=torch.tensor([7,20]);cards=torch.randn(2,20,16)
        conditions=torch.randn(2,9);queries=torch.zeros(2,6,3);queries[:,:,1]=torch.tensor([0.,.2,1.,3.,10.,35.])
        encoded=model.encode(history,length,cards)
        prediction,parameters=model.decode(encoded,conditions,queries,return_parameters=True)
        self.assertTrue(torch.all(parameters['rates']>0))
        torch.testing.assert_close(prediction[:,0],parameters['origin_soh'])
        changed=history.clone();changed[0,7:]=100;changed[1,20:]=100
        torch.testing.assert_close(prediction,model.decode(model.encode(changed,length,cards),conditions,queries))
        torch.testing.assert_close(prediction[:,[4,2,5]],model.decode(encoded,conditions,queries[:,[4,2,5]]),rtol=0,atol=0)
        (prediction-torch.tensor([1.,.98,.96,.9,.8,.5])).square().mean().backward()
        self.assertGreater(float(model.numeric_projection[0].weight.grad.norm()),0)
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
        self.assertGreater(float(model.start_head[-1].weight.grad.norm()),0)
        self.assertGreater(float(model.rate_head[-1].weight.grad.norm()),0)
        restored=self.model().eval();load_adapter(restored,adapter_state(model))
        torch.testing.assert_close(prediction,restored.decode(restored.encode(history,length,cards),conditions,queries),rtol=0,atol=0)


if __name__=='__main__':unittest.main()
