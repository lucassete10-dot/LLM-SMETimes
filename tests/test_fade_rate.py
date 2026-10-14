import unittest
import numpy as np
from battery_smetimes.fade_rate import fit,predict


class RateTest(unittest.TestCase):
    def cells(self):
        cells={};parameters=[]
        for i in range(3):
            key=str(i);cycle=np.array([101.,150.,200.,300.]);a=1.+i*.01
            cells[key]=dict(early_cycle=np.arange(1.,101.),early_raw=np.tile(np.linspace(3,4,32),(100,1))+i*.1,conditions=(1,1,25),physical_family='test',cycle=cycle,truth=a-.02*cycle/100)
            parameters.append(dict(cell_id=key,anchor=a))
        return cells,parameters

    def test_reconstruction_and_pollution(self):
        cells,parameters=self.cells()
        for learned in (False,True):
            b=fit(cells,list(cells),learned,17,parameters);c=cells['1'];cycle=c['cycle']
            p=predict(b,c,cycle);np.testing.assert_allclose(p,c['truth'],atol=1e-12,rtol=0)
            poison=dict(c,early_cycle=np.r_[c['early_cycle'],101],early_raw=np.vstack([c['early_raw'],np.full((1,32),np.inf)]),truth=np.inf,cycle_life=np.inf)
            np.testing.assert_allclose(p,predict(b,poison,cycle),atol=1e-12,rtol=0)
            with self.assertRaises(ValueError):predict(b,dict(c,physical_family='other'),cycle)
            with self.assertRaises(ValueError):predict(b,c,np.array([100.]))

    def test_training_membership_rejected(self):
        cells,parameters=self.cells()
        with self.assertRaises(ValueError):fit(cells,list(cells),True,17,parameters[::-1])


if __name__=='__main__':unittest.main()
