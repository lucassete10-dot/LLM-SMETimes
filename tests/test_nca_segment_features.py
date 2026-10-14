import unittest
import numpy as np
from battery_smetimes.nca_segment_features import prefix,inputs


def cell():
    c=np.arange(1,101)
    return dict(early_cycle=c,early_raw=np.tile(c[:,None],(1,32))*.01,
        conditions=np.array([25,.5]),physical_family='Zhu2022_dataset1_NCA_18650_3p5Ah_cohort')


class FeatureTests(unittest.TestCase):
    def test_actual_cycle_slope(self):
        p=prefix(cell(),dict(block='V',context='basic'))
        np.testing.assert_allclose(p[48:64],1)

    def test_labels_future_and_current_not_read_by_v(self):
        c=cell();par=dict(block='V',context='segments');p=prefix(c,par)
        c['truth']=np.inf;c['cycle_life']=np.inf;c['cycle']=np.inf;c['prefix']=np.inf
        c['early_raw'][:,16:]=np.inf
        c['early_raw']=np.vstack([c['early_raw'],np.full((1,32),np.inf)])
        c['early_cycle']=np.r_[c['early_cycle'],9999]
        np.testing.assert_array_equal(p,prefix(c,par))

    def test_context_and_vi_shapes(self):
        c=cell();basic=prefix(c,dict(block='V',context='basic'))
        segment=prefix(c,dict(block='V',context='segments'))
        self.assertEqual(len(segment)-len(basic),16*7+4)
        self.assertGreater(len(prefix(c,dict(block='VI',context='segments'))),len(segment))

    def test_unknown_family_and_schema_rejected(self):
        c=cell();c['physical_family']='NASA'
        with self.assertRaises(ValueError):prefix(c,dict(block='V',context='basic'))
        c=cell();c['early_raw']=np.zeros((100,51))
        with self.assertRaises(ValueError):prefix(c,dict(block='V',context='basic'))

    def test_future_request_validation(self):
        for future in ([100],[np.inf],[]):
            with self.assertRaises(ValueError):inputs(cell(),future,dict(block='VI',context='basic'))
