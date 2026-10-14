import unittest
import numpy as np
import torch
from battery_smetimes.monotone_network import CurveNet,predict
from battery_smetimes.warm_monotone import inverse_bias,fit
from tests.test_conditional_drift import cell


class WarmTest(unittest.TestCase):
    def test_initialization_inverse(self):
        prior=np.array([1,.012,.001,4.]);model=CurveNet(2,32)
        with torch.no_grad():model.network[-1].weight.zero_();model.network[-1].bias.copy_(torch.tensor(inverse_bias(prior)))
        np.testing.assert_allclose(model.parameters_for(torch.zeros(1,2)).detach().numpy()[0],prior,atol=1e-6)

    def test_training_only_initialization_and_future_invariance(self):
        cells={}
        for key,value in [('a',1),('b',2),('unused',999)]:
            c=cell(value);c.update(cycle=np.array([101,110,120]),truth=np.array([1,.98,.95]));cells[key]=c
        b=fit(cells,['a','b'],32,.1,1,epochs=2)
        self.assertEqual({r['cell_id'] for r in b['training_parameter_fits']},{'a','b'})
        c=cell(3);p=predict(b,c,[110,120]);poison=dict(c,early_cycle=np.r_[c['early_cycle'],101],early_raw=np.vstack([c['early_raw'],[[np.inf,np.inf]]]),truth=np.inf)
        np.testing.assert_array_equal(p,predict(b,poison,[110,120]))


if __name__=='__main__':unittest.main()
