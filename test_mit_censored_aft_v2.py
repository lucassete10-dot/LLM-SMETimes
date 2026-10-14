import unittest

import torch

from evaluate_mit_censored_aft_v2 import censored_likelihood_loss


class MITCensoredAFTTests(unittest.TestCase):
    def test_censored_loss_uses_survival_lower_bound_not_false_endpoint(self):
        bound = torch.tensor([100.])
        weight = torch.tensor([1.])
        event = torch.tensor([False])
        too_short = censored_likelihood_loss(torch.tensor([2.]), bound, event, weight)
        longer = censored_likelihood_loss(torch.tensor([6.]), bound, event, weight)
        self.assertGreater(too_short.item(), longer.item())
        observed = censored_likelihood_loss(torch.tensor([4.]), bound,
                                             torch.tensor([True]), weight)
        self.assertTrue(torch.isfinite(observed))


if __name__ == '__main__':
    unittest.main()
