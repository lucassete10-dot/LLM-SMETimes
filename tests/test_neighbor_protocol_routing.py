import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from battery_smetimes.neighbor_curve import predict

class RoutingTest(unittest.TestCase):
    def bundle(self,local_count):
        identity=SimpleNamespace(transform=lambda x:np.asarray(x))
        refs=[dict(cell_id=str(i),condition='L' if i<local_count else 'O',cycle=np.array([101.,201.]),truth=np.array([1-i*.1,.9-i*.1])) for i in range(4)]
        return dict(family='A',train_ids=[str(i) for i in range(4)],references=refs,z=np.array([[100.],[101.],[102.],[0.]]),imputer=identity,scaler=identity)
    def query(self,b,exclude=None):
        with patch('battery_smetimes.neighbor_curve.prefix',return_value=np.array([0.])),patch('battery_smetimes.neighbor_curve.conditions',return_value='L'):
            return predict(b,dict(physical_family='A'),[101.,201.],1,exclude=exclude)
    def test_three_protocol_peers_take_local(self):
        np.testing.assert_allclose(self.query(self.bundle(3)),[1,.9])
    def test_two_protocol_peers_fallback_global(self):
        np.testing.assert_allclose(self.query(self.bundle(2)),[.7,.6])
    def test_excludes_self_from_eligible_peers(self):
        np.testing.assert_allclose(self.query(self.bundle(0),exclude='3'),[1,.9])

if __name__=='__main__':unittest.main()
