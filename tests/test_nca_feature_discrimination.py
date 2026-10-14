import unittest
import numpy as np
from scripts.train_nca_feature_discrimination import features,permutation,predict
from tests.test_nca_segment_features import cell


class Identity:
    def transform(self,x):return x


class Fixed:
    def predict(self,x):return np.tile([1.05,.01,.005],(len(x),1))


class DiscriminationTests(unittest.TestCase):
    def test_features_ignore_labels_and_future(self):
        for arm in ('conditions','prefix','segments','prefix_shuffled'):
            c=cell();old=features(c,arm)
            c.update(truth=np.inf,cycle=np.inf,capacity=np.inf,cycle_life=np.inf)
            c['early_cycle']=np.r_[c['early_cycle'],101];c['early_raw']=np.vstack([c['early_raw'],np.full((1,32),np.inf)])
            np.testing.assert_array_equal(old,features(c,arm))

    def test_shuffle_never_crosses_protocol_or_membership(self):
        cells={str(i):dict(cell(),conditions=np.array([25.,.5 if i<3 else .25])) for i in range(6)}
        ids=['0','1','2','3'];mapping=permutation(cells,ids,20261005,1)
        self.assertEqual(set(mapping),set(ids));self.assertEqual(set(mapping.values()),set(ids))
        for k,v in mapping.items():np.testing.assert_array_equal(cells[k]['conditions'],cells[v]['conditions'])
        self.assertEqual(mapping['3'],'3');self.assertEqual(mapping,permutation(cells,ids,20261005,1))

    def test_inference_shape_and_rejections(self):
        b=dict(family=cell()['physical_family'],arm='conditions',imputer=Identity(),model=Fixed(),scale=np.ones(3))
        h=np.array([.01,1.,2.]);np.testing.assert_allclose(predict(b,cell(),[101.,200.,300.]),1.05-.01*h-.005*h*h)
        for future in ([],[100.],[np.inf]):
            with self.assertRaises(ValueError):predict(b,cell(),future)
        with self.assertRaises(ValueError):predict(b,dict(cell(),physical_family='NASA'),[101.])


if __name__=='__main__':unittest.main()
