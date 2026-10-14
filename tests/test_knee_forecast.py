import unittest
import numpy as np
from battery_smetimes.knee_forecast import fit_knee,trajectory,early_features
from tests.test_curve_state_transfer import cell


class KneeForecastTests(unittest.TestCase):
    def test_training_fit_represents_knee(self):
        x=np.arange(101,1001);coef=np.array([1.02,.002,.008,5.])
        y=trajectory(coef,x);fit=fit_knee(x,y)
        np.testing.assert_allclose(trajectory(fit,x),y,atol=.0002)
        self.assertGreater(fit[0],1)

    def test_window_features_ignore_future_and_labels(self):
        c=cell();before=early_features(c,True)
        c.pop('soh');c.pop('label_cycle');c['raw'][c['cycle']>100]=np.inf
        np.testing.assert_array_equal(before,early_features(c,True))

    def test_parameter_average_is_not_curve_average(self):
        parameters=np.array([[1.,0.,.01,5.],[1.,0.,.03,10.]])
        x=np.array([700.,1500.])
        direct=trajectory(parameters.mean(axis=0),x)
        mixture=np.mean([trajectory(p,x) for p in parameters],axis=0)
        self.assertGreater(abs(direct[-1]-mixture[-1]),.1)


if __name__=='__main__':unittest.main()
