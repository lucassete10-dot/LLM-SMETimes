import unittest
import torch
from battery_smetimes.llm_local_recovery import LocalRecoveryQueryForecaster,interpolate_fixed_nodes
from battery_smetimes.research_llm import adapter_state,load_adapter
from tests import test_llm_query_rate as rate_tests


class FixedRecoveryTests(unittest.TestCase):
    def test_rise_drop_unsorted_subset_and_constant_extension(self):
        knots=torch.tensor([0.,1.,2.]);values=torch.tensor([[0.,.01,-.005]],requires_grad=True)
        time=torch.tensor([[3.,.5,0.,1.5,1.]])
        y=interpolate_fixed_nodes(values,knots,time)
        torch.testing.assert_close(y,torch.tensor([[-.005,.005,0.,.0025,.01]]))
        torch.testing.assert_close(y[:,[3,0]],interpolate_fixed_nodes(values,knots,time[:,[3,0]]),rtol=0,atol=0)
        y.sum().backward();self.assertTrue(torch.isfinite(values.grad).all())


class LocalDecoderTests(rate_tests.QueryDecoderTests):
    def model(self,**kwargs):
        return LocalRecoveryQueryForecaster('test-only',3,device='cpu',dtype=torch.float32,rank=4,dropout=0,
            rate_width=16,local_recovery_spacing_cycles=10,**kwargs)

    def test_bounded_recovery_zero_origin_and_llm_gradient(self):
        model=self.model().eval();history=torch.randn(1,100,3);length=torch.tensor([20]);cards=torch.randn(1,20,16)
        encoding=model.encode(history,length,cards);conditions=torch.randn(1,9);queries=torch.zeros(1,4,3)
        queries[:,:,1]=torch.tensor([0.,.3,4.,35.])
        prediction,parameters=model.decode(encoding,conditions,queries,return_parameters=True)
        local=parameters['local_recovery_values']
        self.assertEqual(float(local[0,0].detach()),0.)
        self.assertTrue(bool((abs(local)<=model.local_recovery_max_soh).all()))
        torch.testing.assert_close(prediction[:,0],parameters['origin_soh'])
        local.square().sum().backward()
        self.assertGreater(float(model.numeric_projection[0].weight.grad.norm()),0)
        if model.local_query is not None:self.assertGreater(float(model.local_query.grad.norm()),0)
        self.assertGreater(float(model.local_head[-1].weight.grad.norm()),0)
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))


class VectorLocalDecoderTests(LocalDecoderTests):
    def model(self,**kwargs):
        return super().model(local_readout='vector',**kwargs)

    def test_each_nonorigin_node_has_independent_finite_gradient(self):
        model=self.model().eval();history=torch.randn(1,100,3);cards=torch.randn(1,20,16)
        encoding=model.encode(history,torch.tensor([20]),cards)
        local=model.local_parameters(encoding,torch.randn(1,9))
        local.sum().backward();gradient=model.local_head[-1].weight.grad
        self.assertEqual(float(gradient[0].abs().sum()),0.)
        self.assertTrue(torch.all(gradient[1:].norm(dim=1)>0))
        self.assertTrue(torch.isfinite(gradient).all())

    def test_optimizer_covers_trainable_parameters_once_and_preserves_scale_on_restore(self):
        from scripts.train_llm_curve_repair import repair_optimizer
        model=self.model();settings={'learning_rate':.001,'local_decoder_lr_scale':5.}
        optimizer=repair_optimizer(model,settings,0.)
        ids=[id(p) for group in optimizer.param_groups for p in group['params']]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(set(ids),{id(p) for p in model.parameters() if p.requires_grad})
        self.assertEqual([g['lr'] for g in optimizer.param_groups],[.001,.005])
        restored=repair_optimizer(model,settings,0.);restored.load_state_dict(optimizer.state_dict())
        self.assertEqual([g['lr_scale'] for g in restored.param_groups],[1.,5.])


if __name__=='__main__':unittest.main()
