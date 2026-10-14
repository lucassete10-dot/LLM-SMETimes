import unittest
from battery_smetimes.xjtu_regularized_boosting import model_parameters,predict


class RegularizedBoostingTest(unittest.TestCase):
    def test_no_cycle_early_stopping(self):
        for loss in ('squared_error','absolute_error'):
            p=model_parameters(loss,20261005);self.assertFalse(p['early_stopping']);self.assertEqual(p['loss'],loss);self.assertEqual(p['l2_regularization'],10)

    def test_no_unregistered_loss(self):
        with self.assertRaises(ValueError):model_parameters('other',1)

    def test_family_guard(self):
        with self.assertRaises(ValueError):predict({'family':'A'},{'physical_family':'B'},[101.])


if __name__=='__main__':unittest.main()
