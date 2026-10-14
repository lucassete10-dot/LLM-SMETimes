import unittest
import numpy as np
from battery_smetimes.xjtu_segment_features import columns,prefix

def query():
    c=np.array([1,5,20,40,60,80,100])
    return dict(cycle=c,raw=np.repeat(c[:,None]/100,46,axis=1),metadata={'dataset':'XJTU','group':'b1','charge_rate_c':2.,'charge_rate_steps_c':None,'diagnostic_discharge_rate_c':1.,'discharge_cutoff_v':2.5})

class SegmentsTests(unittest.TestCase):
    def test_channel_columns(self):
        self.assertEqual([len(columns(k)) for k in ('V','VI','VIT')],[16,30,44])
        for k in ('V','VI','VIT'):self.assertFalse(set(columns(k))&{22,45})
    def test_actual_cycle_slope(self):
        q=query();p=prefix(q,dict(block='V',context='basic'));n=len(columns('V'))
        np.testing.assert_allclose(p[3*n:4*n],1.)
    def test_no_future_or_integral_influence(self):
        q=query();args=dict(block='VIT',context='segments');expected=prefix(q,args)
        q['raw'][:,[22,45]]=np.inf;q['cycle']=np.r_[q['cycle'],101];q['raw']=np.vstack([q['raw'],np.full((1,46),np.inf)])
        np.testing.assert_array_equal(expected,prefix(q,args))
    def test_sparse_bins_are_not_fabricated(self):
        q=query();q['cycle']=np.array([1,2,3,4,5,100]);q['raw']=q['raw'][:6]
        p=prefix(q,dict(block='V',context='segments'));n=len(columns('V'))
        self.assertTrue(np.isnan(p[5*n:7*n]).all())

if __name__=='__main__':unittest.main()
