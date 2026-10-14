import unittest
import numpy as np
from battery_smetimes.mit_tail_weights import point_weights


class WeightsTest(unittest.TestCase):
    def test_ramp_matches_original(self):
        c=np.arange(101,200)
        expected=np.linspace(1,3,len(c));expected/=expected.sum()
        np.testing.assert_array_equal(point_weights(c,'ramp3'),expected)

    def test_fixed_ratios_and_total(self):
        for scheme,ratio in [('ramp10',10),('ramp30',30),('terminal10',10)]:
            w=point_weights([101,110,120,200],scheme)
            self.assertAlmostEqual(w.sum(),1)
            self.assertAlmostEqual(w[-1]/w[0],ratio)

    def test_terminal_uses_actual_cycle(self):
        w=point_weights([101,102,103,104,200],'terminal10')
        np.testing.assert_allclose(w[:4],np.full(4,1/14))
        self.assertAlmostEqual(w[-1],10/14)

    def test_invalid(self):
        for c in ([],[101],[101,101],[102,101],[101,np.inf]):
            with self.assertRaises(ValueError):point_weights(c,'ramp10')
        with self.assertRaises(ValueError):point_weights([101,102],'unlocked')
