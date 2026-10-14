import unittest
import numpy as np
from battery_smetimes.nca_parameter_shape import predict
from tests.test_nca_segment_features import cell


class Identity:
    def transform(self,x):return np.asarray(x)


class Fixed:
    def __init__(self,coef):self.coef=np.asarray(coef)
    def predict(self,x):return np.tile(self.coef,(len(x),1))


class ShapeTests(unittest.TestCase):
    def bundle(self,mode):
        coef=[1.05,.01,.005]+([1.] if mode=='knee' else [])
        return dict(family=cell()['physical_family'],parameters=dict(shape=mode),model=Fixed(coef),imputer=Identity(),scale=np.ones(len(coef)))

    def test_quadratic_coordinates(self):
        c=np.array([101.,200.,300.]);h=(c-100)/100
        np.testing.assert_allclose(predict(self.bundle('quadratic'),cell(),c),1.05-.01*h-.005*h*h)

    def test_knee_preserves_pre_knee_linear_and_accelerates(self):
        c=np.array([150.,200.,300.]);h=(c-100)/100
        p=predict(self.bundle('knee'),cell(),c)
        np.testing.assert_allclose(p,1.05-.01*h-.005*np.maximum(h-1,0)**2)
        self.assertGreater(p[1]-p[2],p[0]-p[1])

    def test_query_labels_and_future_not_used(self):
        c=cell();b=self.bundle('knee');future=[101.,200.,300.];expected=predict(b,c,future)
        c.update(truth=np.inf,soh=np.inf,cycle=np.inf,capacity=np.inf,cycle_life=np.inf)
        c['early_cycle']=np.r_[c['early_cycle'],9999];c['early_raw']=np.vstack([c['early_raw'],np.full((1,32),np.inf)])
        np.testing.assert_array_equal(expected,predict(b,c,future))

    def test_family_and_bad_future_rejected(self):
        c=cell();c['physical_family']='NASA'
        with self.assertRaises(ValueError):predict(self.bundle('quadratic'),c,[101.])
        for mode in ('quadratic','knee'):
            for future in ([100.],[np.inf]):
                with self.assertRaises(ValueError):predict(self.bundle(mode),cell(),future)


if __name__=='__main__':unittest.main()
