import unittest
import numpy as np
import torch
from battery_smetimes.multiview_pilot import CurveNet, manifest, tensors


class MultiviewContractTests(unittest.TestCase):
    def test_registered_parameter_difference(self):
        dims=manifest()['view_dimensions']
        a=CurveNet(dims,'concat');b=CurveNet(dims,'gate')
        self.assertEqual(sum(p.numel() for p in a.parameters()),10577)
        self.assertEqual(sum(p.numel() for p in b.parameters()),10724)

    def test_capacity_columns_absent(self):
        names=manifest()['raw_columns']
        self.assertEqual(len(names),49)
        self.assertFalse(any('charge_ah' in n or 'energy_wh' in n for n in names))

    def test_unregistered_architecture_rejected(self):
        with self.assertRaises(ValueError):CurveNet(manifest()['view_dimensions'],'other')

    def test_wrong_family_rejected_before_features(self):
        with self.assertRaises(ValueError):tensors(dict(family='A'),dict(physical_family='B'),[101])

    def test_direct_head_has_one_output_per_requested_point(self):
        dims=manifest()['view_dimensions'];net=CurveNet(dims,'gate')
        with torch.no_grad():p=net([torch.zeros((3,d)) for d in dims],torch.zeros((3,3)))
        self.assertEqual(tuple(p.shape),(3,))
        self.assertTrue(np.isfinite(p.numpy()).all())


if __name__=='__main__':unittest.main()
