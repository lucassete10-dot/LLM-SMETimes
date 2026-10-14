import unittest
import numpy as np
from battery_smetimes.conditional_drift import encode_fit,encode,query,prefix
from scripts.verify_conditional_drift import manual_scale


def cell(value):
    return dict(physical_family='NCA',early_cycle=np.arange(1,101),early_raw=np.full((100,2),value),conditions=np.array([25,.5]),prefix=np.zeros(10))


class ConditionalDriftTest(unittest.TestCase):
    def test_near_constant_scale_error_bound(self):
        np.testing.assert_array_equal(manual_scale(np.array([1e-32,4.,0.]),np.array([1.,1.,0.]),39),[1.,2.,1.])

    def test_training_only_encoder_and_count_cap(self):
        cells={'a':cell(1),'b':cell(2),'unused':cell(999)}
        encoder=encode_fit(cells,['a','b'],4)
        self.assertEqual(encoder['pca'].n_components_,1)
        np.testing.assert_allclose(encoder['imputer'].statistics_,np.median(np.stack([prefix(cells[k]) for k in ['a','b']]),axis=0))
        self.assertEqual(query(encoder,cell(3),[200,300],2).shape,(2,5))
        with self.assertRaises(ValueError):encode(encoder,dict(cell(3),physical_family='other'))

    def test_future_poison_cannot_change_encoding(self):
        c=cell(2);encoder=encode_fit({'a':cell(1),'b':cell(3)},['a','b'],2)
        p=dict(c,early_cycle=np.r_[c['early_cycle'],101],early_raw=np.vstack([c['early_raw'],[[np.inf,np.inf]]]),truth=np.inf,cycle_life=np.inf)
        np.testing.assert_array_equal(encode(encoder,c),encode(encoder,p))


if __name__=='__main__':unittest.main()
