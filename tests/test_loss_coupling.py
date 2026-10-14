import unittest

import torch

from battery_smetimes.models.cycle_time import CycleTimeHealthModel
from battery_smetimes.models.loss_coupling import LossSeparatedHealthModel


class LossCouplingTest(unittest.TestCase):
    def test_identical_values_but_no_future_gradient_into_state_or_encoder(self):
        for backbone in ("gru", "tcn"):
            torch.manual_seed(5)
            joint = CycleTimeHealthModel(6, context_width=4, backbone=backbone).eval()
            torch.manual_seed(5)
            separate = LossSeparatedHealthModel(6, context_width=4, backbone=backbone).eval()
            args = (torch.randn(3, 5, 6), torch.ones(3, 5, dtype=torch.bool),
                    torch.randn(3, 6), (10, 50, 100))
            context = torch.randn(3, 5, 4)
            a = joint(*args, cycle_context=context)
            b = separate(*args, cycle_context=context)
            torch.testing.assert_close(a["current_soh"], b["current_soh"])
            torch.testing.assert_close(a["future_soh"], b["future_soh"])
            self.assertEqual(sum(p.numel() for p in joint.parameters()),
                             sum(p.numel() for p in separate.parameters()))
            b["future_soh"].square().sum().backward()
            self.assertIsNone(separate.core.current_head.weight.grad)
            self.assertIsNone(separate.core.cycle_projection[0].weight.grad)
            self.assertGreater(float(separate.core.rate_head.weight.grad.abs().sum()), 0)
            self.assertGreater(float(separate.core.acceleration_head.weight.grad.abs().sum()), 0)
            a["future_soh"].square().sum().backward()
            self.assertGreater(float(joint.core.current_head.weight.grad.abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()
