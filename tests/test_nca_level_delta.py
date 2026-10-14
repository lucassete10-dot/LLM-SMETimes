import unittest
import numpy as np
from battery_smetimes.nca_level_delta import predict,fit
from tests.test_nca_segment_features import cell


class Identity:
    def transform(self,x):return np.asarray(x)


class Constant:
    def predict(self,x):return np.full(len(x),.98)


class Changes:
    def predict(self,x):return -.001*np.nansum(x,axis=1)


class LevelDeltaTests(unittest.TestCase):
    def bundle(self):
        return dict(family=cell()['physical_family'],level=Constant(),level_imputer=Identity(),delta=Changes(),delta_imputer=Identity())

    def test_origin_equals_learned_level(self):
        self.assertAlmostEqual(predict(self.bundle(),cell(),[101.])[0],.98)

    def test_query_labels_and_future_raw_ignored(self):
        c=cell();future=[101.,120.,140.];expected=predict(self.bundle(),c,future)
        c.update(truth=np.inf,cycle=np.inf,soh=np.inf,capacity=np.inf,cycle_life=np.inf)
        c['early_cycle']=np.r_[c['early_cycle'],101.]
        c['early_raw']=np.vstack([c['early_raw'],np.full((1,32),np.inf)])
        np.testing.assert_array_equal(expected,predict(self.bundle(),c,future))

    def test_family_and_invalid_horizon_rejected(self):
        c=cell();c['physical_family']='MIT'
        with self.assertRaises(ValueError):predict(self.bundle(),c,[101.])
        for future in ([100.],[np.inf]):
            with self.assertRaises(ValueError):predict(self.bundle(),cell(),future)

    def test_missing_origin_label_not_silently_excluded(self):
        c=cell();c.update(cycle=np.array([102.,103.]),truth=np.array([.98,.97]))
        with self.assertRaises(ValueError):fit({'c':c},['c'],1,42)


if __name__=='__main__':unittest.main()
