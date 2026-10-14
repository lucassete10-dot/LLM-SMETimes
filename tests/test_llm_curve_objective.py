import unittest
import numpy as np
import torch
from battery_smetimes.llm_curve_objective import curve_loss, curve_metrics, training_origin_target
from scripts.verify_llm_curve_metrics import point_metrics


class CurveObjectiveTests(unittest.TestCase):
    def test_local_rate_supervision_is_masked_and_differentiable(self):
        x = torch.arange(101., 501.)[None]
        y = torch.ones_like(x)
        valid = torch.ones_like(x)
        _, perfect = curve_loss(y, y, x, valid, rate_weight=.1)
        self.assertEqual(float(perfect['rate']), 0.)
        p = (1 - (x-100).square() * 1e-6).requires_grad_()
        _, wrong = curve_loss(p, y, x, valid, rate_weight=.1)
        self.assertGreater(float(wrong['rate'].detach()), 0.)
        wrong['rate'].sum().backward()
        self.assertGreater(float(p.grad.norm()), 0.)
        padded_x = torch.cat((x, torch.zeros(1, 20)), 1)
        padded_y = torch.cat((y, torch.full((1, 20), 100.)), 1)
        padded_p = torch.cat((p.detach(), torch.full((1, 20), -100.)), 1)
        padded_valid = torch.cat((valid, torch.zeros(1, 20)), 1)
        _, replay = curve_loss(padded_p, padded_y, padded_x, padded_valid, rate_weight=.1)
        torch.testing.assert_close(wrong['rate'], replay['rate'])

    def test_local_rate_does_not_invent_slopes_across_sparse_or_short_data(self):
        for x in (torch.tensor([[101., 102., 103.]]), torch.tensor([[101., 201., 301.]])):
            y = torch.ones_like(x)
            _, terms = curve_loss(y*.9, y, x, torch.ones_like(x), rate_weight=.1)
            self.assertEqual(float(terms['rate']), 0.)

    def test_independent_metrics_include_irregular_cycle_weighting(self):
        x=np.array([101.,102.,110.]);y=np.array([1.,1.,1.]);p=np.array([1.,1.,.9])
        metrics,weighted=point_metrics(x,y,p)
        self.assertAlmostEqual(metrics['mae_pp'],10/3)
        self.assertAlmostEqual(weighted,40/9)
        self.assertAlmostEqual(metrics['tail_mae_pp'],10)
        self.assertIsNone(metrics['r2'])

    def test_training_start_label_never_uses_future_or_fills_missing(self):
        cell={'label_cycle':np.array([99.,100.,101.]),'soh':np.array([.99,.98,.1])}
        self.assertEqual(training_origin_target(cell,100),(.98,1.))
        cell['soh'][-1]=100
        self.assertEqual(training_origin_target(cell,100),(.98,1.))
        cell['soh'][1]=np.nan
        self.assertEqual(training_origin_target(cell,100),(0.,0.))
        self.assertEqual(training_origin_target(cell,98),(0.,0.))

    def test_exact_curve_passes_and_bad_tail_does_not(self):
        x=np.arange(101,501,dtype=float)
        y=1-.0001*(x-100)-.0000005*(x-100)**2
        self.assertTrue(curve_metrics(x,y,y)["joint_pass"])
        p=y.copy();p[x>400]+=.05
        self.assertFalse(curve_metrics(x,y,p)["joint_pass"])

    def test_false_plateau_and_rise(self):
        x=np.arange(101,501,dtype=float); y=1-.0005*(x-100)
        self.assertTrue(curve_metrics(x,y,np.full_like(y,.9))["false_plateau"])
        self.assertTrue(curve_metrics(x,y,.8+.0005*(x-100))["false_rise"])

    def test_loss_padding_invariance_and_gradient(self):
        x=torch.tensor([[101.,102.,200.,300.,400.,500.,0.,0.]])
        y=torch.tensor([[1.,.99,.97,.94,.9,.85,0.,0.]])
        valid=torch.tensor([[1.,1.,1.,1.,1.,1.,0.,0.]])
        p=(y+.02).clone().requires_grad_()
        a,_=curve_loss(p,y,x,valid,.5,.5)
        yy=y.clone();yy[:,-2:]=100
        b,_=curve_loss(p,yy,x,valid,.5,.5)
        torch.testing.assert_close(a,b)
        a.mean().backward();self.assertGreater(float(p.grad.norm()),0)
        self.assertEqual(float(p.grad[:,-2:].abs().sum()),0)


if __name__=="__main__": unittest.main()
