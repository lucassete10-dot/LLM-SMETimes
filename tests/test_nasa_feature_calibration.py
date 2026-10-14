import unittest
import numpy as np
from battery_smetimes.nasa_feature_calibration import BLOCKS,features,fit,predict

def cell(offset=0.):
    c=np.arange(1,101);raw=np.tile(np.arange(44,dtype=float),(100,1))+c[:,None]*.001+offset
    return dict(cycle=c,raw=raw,label_cycle=c,soh=1-c*.001,physical_family='F',metadata={'discharge_cutoff_v':2.5})

class FeatureCalibrationTests(unittest.TestCase):
    def test_blocks_exclude_integrals(self):
        self.assertEqual([len(BLOCKS[k]) for k in ('V','VI','VIT')],[14,27,42])
        for cols in BLOCKS.values():self.assertFalse(set(cols)&{42,43})
    def test_future_and_integrals_cannot_change_features(self):
        q=cell();p=dict(block='VIT',mode='delta',alpha=100.)
        c,x=features(q,p);bad=dict(q,cycle=np.r_[q['cycle'],101],raw=np.vstack([q['raw'],np.full((1,44),np.inf)]))
        bad['raw'][:,42:]=np.inf;cc,xx=features(bad,p)
        np.testing.assert_array_equal(c,cc);np.testing.assert_array_equal(x,xx)
    def test_delta_removes_signal_offset_not_cutoff(self):
        p=dict(block='V',mode='delta',alpha=100.)
        _,x=features(cell(),p);_,z=features(cell(3.),p)
        np.testing.assert_allclose(x,z,rtol=0,atol=1e-12);self.assertTrue(np.all(x[:,-1]==2.5))
    def test_wrong_family_is_rejected(self):
        q=cell();b=fit({'a':q,'b':cell(.1)},['a','b'],dict(block='V',mode='absolute',alpha=100.))
        with self.assertRaises(ValueError):predict(b,dict(q,physical_family='OTHER'),[101,110])

if __name__=='__main__':unittest.main()
