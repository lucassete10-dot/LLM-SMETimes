import unittest
import numpy as np
from battery_smetimes.curve_state_transfer import fit, predict, fade_features


def cell(offset=0):
    cycle=np.arange(1,121,dtype=float)
    raw=np.tile((4-cycle*.003+offset)[:,None],(1,44))
    return dict(cycle=cycle,raw=raw,label_cycle=cycle,soh=1-cycle*.002+offset,
        reference_established_source_cycle=1,physical_family='synthetic_same_family',
        metadata=dict(dataset='NASA',group='NASA',charge_protocol=None,
            charge_rate_c=None,diagnostic_discharge_rate_c=None,discharge_cutoff_v=2.7))


class CurveStateTransferTests(unittest.TestCase):
    def test_prediction_ignores_labels_and_future_signals(self):
        cells={'a':cell(),'b':cell(.01)}
        bundle=fit(cells,['a','b'],dict(state_alpha=10.,fade_alpha=100.))
        query=np.arange(101,121)
        sample=cell(.005)
        original=predict(bundle,sample,query)
        sample.pop('soh'); sample.pop('label_cycle')
        sample['raw'][sample['cycle']>100]=np.inf
        np.testing.assert_array_equal(original,predict(bundle,sample,query))
        self.assertEqual(bundle['train_ids'],['a','b'])
        sample['physical_family']='unseen_family'
        with self.assertRaises(ValueError):
            predict(bundle,sample,query)

    def test_query_features_have_explicit_horizon_interactions(self):
        x=fade_features(np.array([2.,3.]),100,[150,200])
        np.testing.assert_array_equal(x,[[2,3,.5,.25,1,1.5],[2,3,1,1,2,3]])
        with self.assertRaises(ValueError):
            fade_features(np.array([2.]),100,[100])


if __name__=='__main__':
    unittest.main()
