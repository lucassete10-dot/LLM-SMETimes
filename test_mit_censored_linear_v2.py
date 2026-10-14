import unittest

import torch

from evaluate_mit_censored_linear_v2 import BoundedLinearRUL, MAX_RUL_CYCLES, objective


class MITCensoredLinearTests(unittest.TestCase):
    def test_linear_output_is_bounded(self):
        model = BoundedLinearRUL(3)
        estimate = torch.expm1(model(torch.tensor([[100., -100., 50.]])))
        self.assertGreaterEqual(estimate.item(), 0.)
        self.assertLessEqual(estimate.item(), MAX_RUL_CYCLES)

    def test_censored_loss_is_one_sided(self):
        lower = torch.tensor([100.])
        event = torch.tensor([False])
        weight = torch.tensor([1.])
        below = objective(torch.tensor([2.]), lower, event, weight, True)
        above = objective(torch.tensor([6.]), lower, event, weight, True)
        self.assertGreater(below.item(), above.item())
        self.assertEqual(above.item(), 0.)


if __name__ == '__main__':
    unittest.main()
