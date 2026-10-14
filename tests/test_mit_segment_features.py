import unittest
import numpy as np
from battery_smetimes.mit_segment_features import prefix,columns,condition_key

def cell():
    c=np.arange(1,101)
    return dict(cycle=c,raw=np.tile(c[:,None],(1,51))*.01,
        metadata=dict(dataset='MIT',group='batch1',charge_protocol='4C(80%)-4C',
            charge_rate_steps_c=[4,4],diagnostic_discharge_rate_c=4),physical_family='A123')

class FeatureTests(unittest.TestCase):
    def test_q_excluded(self):
        for b in ('V','VI','VIT'):self.assertFalse(set(columns(b))&{38,39})
    def test_batch_not_feature(self):
        c=cell();p=prefix(c,dict(block='VIT',context='segments'))
        c['metadata']['group']='other_batch'
        np.testing.assert_array_equal(p,prefix(c,dict(block='VIT',context='segments')))
    def test_physical_policy_not_storage_suffix(self):
        c=cell();k=condition_key(c);c['metadata']['charge_protocol']+='-newstructure'
        self.assertEqual(k,condition_key(c))
    def test_future_and_q_poison(self):
        c=cell();p=prefix(c,dict(block='VIT',context='segments'))
        c['raw'][:,[38,39]]=np.inf;c['cycle']=np.r_[c['cycle'],9999];c['raw']=np.vstack([c['raw'],np.full((1,51),np.inf)])
        np.testing.assert_array_equal(p,prefix(c,dict(block='VIT',context='segments')))
    def test_actual_source_slope(self):
        c=cell();cols=columns('V');p=prefix(c,dict(block='V',context='basic'))
        np.testing.assert_allclose(p[3*len(cols):4*len(cols)],1)

if __name__=='__main__':unittest.main()
