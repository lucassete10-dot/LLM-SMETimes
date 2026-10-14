import math
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from battery_smetimes.models.survival import (
    conditional_lognormal_nll, conditional_lognormal_quantile,
)
from evaluate_mit_censored_survival_cv import snapshot_rows


class SurvivalModelTest(unittest.TestCase):
    def test_censoring_is_lower_bound_not_fake_event(self):
        mean = torch.tensor([math.log(400.)], requires_grad=True)
        sigma = torch.tensor(.2)
        origin = torch.tensor([100.])
        censor = torch.tensor([700.])
        censored_loss = conditional_lognormal_nll(
            mean, sigma, origin, censor, torch.tensor([False]))
        observed_loss = conditional_lognormal_nll(
            mean, sigma, origin, censor, torch.tensor([True]))
        self.assertTrue(torch.isfinite(censored_loss).all())
        self.assertGreater(float(censored_loss.detach()), 0)
        self.assertNotAlmostEqual(float(censored_loss.detach()),
                                  float(observed_loss.detach()))
        censored_loss.sum().backward()
        self.assertLess(float(mean.grad), 0)  # Longer life improves censor likelihood.

    def test_conditional_intervals_are_ordered_and_after_origin(self):
        mean = torch.tensor([math.log(400.)])
        sigma = torch.tensor(.2)
        origin = torch.tensor([420.])
        low = conditional_lognormal_quantile(mean, sigma, origin, .05)
        median = conditional_lognormal_quantile(mean, sigma, origin, .5)
        high = conditional_lognormal_quantile(mean, sigma, origin, .95)
        self.assertLess(420, float(low))
        self.assertLess(float(low), float(median))
        self.assertLess(float(median), float(high))

    def test_invalid_origin_is_rejected(self):
        with self.assertRaises(ValueError):
            conditional_lognormal_nll(torch.tensor([0.]), torch.tensor(.2),
                                      torch.tensor([10.]), torch.tensor([10.]),
                                      torch.tensor([False]))

    def test_censored_snapshots_retain_only_lower_bound(self):
        cell = SimpleNamespace(
            cycle=np.array([10, 20, 50, 100, 200, 300]),
            soh_available=np.array([False, True, True, True, True, True]),
            capacity_review_flag=np.zeros(6, dtype=bool),
            matrix=lambda variant: np.arange(6, dtype=np.float32)[:, None],
        )
        rows = snapshot_rows({"c": cell}, {"c": {
            "exact80_event_cycle": None, "observed_through_cycle": 300}}, ["c"])
        self.assertTrue(all(not row["event"] for row in rows))
        self.assertTrue(all(row["observed_time"] == 300 for row in rows))
        self.assertTrue(all(row["origin_cycle"] < 300 for row in rows))
        self.assertNotIn(300, [row["origin_cycle"] for row in rows])


if __name__ == "__main__":
    unittest.main()
