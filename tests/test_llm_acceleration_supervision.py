import unittest
import torch
from battery_smetimes.llm_acceleration_supervision import sustained_acceleration_loss


class AccelerationTests(unittest.TestCase):
    def tensors(self):
        x = torch.arange(101., 502.)[None]
        y = 1. - x * .0001 - .0003 * torch.relu(x - 300)
        return x, y, torch.ones_like(x)

    def test_exact_curve_and_offset_have_zero_loss(self):
        x,y,v = self.tensors()
        for p in (y,y+.1):
            loss,count = sustained_acceleration_loss(p,y,x,v)
            self.assertLess(float(loss.sum()),1e-12)
            self.assertGreater(count[0],0)

    def test_wrong_acceleration_has_gradient(self):
        x,y,v = self.tensors()
        p = (1.-x*.0001).requires_grad_()
        loss,_ = sustained_acceleration_loss(p,y,x,v)
        self.assertGreater(float(loss.detach()),0)
        loss.sum().backward()
        self.assertGreater(float(p.grad.abs().sum()),0)

    def test_source_shift_invariant(self):
        x,y,v = self.tensors()
        p = 1.-x*.0001
        a,_ = sustained_acceleration_loss(p,y,x,v)
        b,_ = sustained_acceleration_loss(p,y,x+1000,v)
        torch.testing.assert_close(a,b)

    def test_padding_and_sparse_coverage(self):
        x,y,v = self.tensors()
        a,_ = sustained_acceleration_loss(y,y,x,v)
        padded=lambda t: torch.cat((t,torch.full((1,20),float('nan'))),1)
        b,_ = sustained_acceleration_loss(padded(y),padded(y),padded(x),torch.cat((v,torch.zeros(1,20)),1))
        torch.testing.assert_close(a,b)
        loss,count = sustained_acceleration_loss(y[:,::50],y[:,::50],x[:,::50],v[:,::50])
        self.assertEqual(count,[0])
        self.assertEqual(float(loss),0.)

    def test_isolated_spike_is_robust(self):
        x,y,v = self.tensors()
        noisy = y.clone(); noisy[0,150] -= .2
        loss,_ = sustained_acceleration_loss(y,noisy,x,v)
        # A spike may shift a finite-window median by one sample; it must not
        # create a large derivative target. RMS SOH error stays below .01 pp.
        self.assertLess(float(loss),1e-8)


if __name__ == '__main__':
    unittest.main()
