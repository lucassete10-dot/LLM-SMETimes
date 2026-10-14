import unittest
import numpy as np
from battery_smetimes.waveform_shape import projected_rows,vector


class ShapeTest(unittest.TestCase):
    def cell(self):
        cycle=np.arange(1,101);grid=np.linspace(-1,1,11);basis=np.polynomial.legendre.legvander(grid,3)
        raw=np.zeros((100,32));raw[:,:11]=np.array([basis@np.array([4.,-.1,.02*c/100,.003]) for c in cycle]);raw[:,16:27]=-2.
        return dict(early_cycle=cycle,early_raw=raw,physical_family='test')

    def test_known_shape_and_history_trend(self):
        c=self.cell();cycle,z=projected_rows(c)
        np.testing.assert_allclose(z[:,0],-.1,atol=1e-12);np.testing.assert_allclose(z[:,1],.02*cycle/100,atol=1e-12)
        np.testing.assert_allclose(z[:,2],.003,atol=1e-12);np.testing.assert_allclose(z[:,3:],0,atol=1e-12)
        self.assertAlmostEqual(vector(c)[3*6+1],.02,places=12)

    def test_future_and_label_invariance(self):
        c=self.cell();poison=dict(c,early_cycle=np.r_[c['early_cycle'],101],early_raw=np.vstack([c['early_raw'],np.full((1,32),np.inf)]),truth=np.inf,Capacity=np.inf,cycle_life=np.inf)
        np.testing.assert_array_equal(vector(c),vector(poison))
        with self.assertRaises(ValueError):vector(dict(c,early_raw=c['early_raw'][:,:31]))


if __name__=='__main__':unittest.main()
