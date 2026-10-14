import unittest
from unittest.mock import patch
import numpy as np
from battery_smetimes.nasa_calibrated_level import predict


class CalibratedLevelTest(unittest.TestCase):
    def test_offset_does_not_change_drop(self):
        def primary(bundle,cell,origin,future):return 1-np.asarray(future)*.001
        with patch('battery_smetimes.nasa_calibrated_level.predict_baseline',side_effect=primary),patch('battery_smetimes.nasa_calibrated_level.calibrated_predict',return_value=np.array([.95])):
            result=predict(dict(primary={},calibration={}),{},[101,151,201])
        np.testing.assert_allclose(result,[.95,.9,.85],rtol=0,atol=1e-12)

    def test_control_no_calibration_call(self):
        with patch('battery_smetimes.nasa_calibrated_level.predict_baseline',return_value=np.array([.9,.8])),patch('battery_smetimes.nasa_calibrated_level.calibrated_predict') as fn:
            np.testing.assert_array_equal(predict(dict(primary={},calibration=None),{},[101,201]),[.9,.8]);fn.assert_not_called()


if __name__=='__main__':unittest.main()
