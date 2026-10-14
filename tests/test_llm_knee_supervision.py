import unittest
import numpy as np
import torch

from battery_smetimes.llm_knee_supervision import training_knee_target, knee_interval_loss


class KneeSupervisionTests(unittest.TestCase):
    def cell(self, knee=None, last=800):
        x = np.arange(1, last + 1, dtype=float)
        y = 1.02 - .00003 * x
        if knee is not None:
            y -= .0003 * np.maximum(x - knee, 0)
        return {'label_cycle': x, 'soh': y}

    def test_observed_hinge_is_localized(self):
        label = training_knee_target(self.cell(400), 100)
        self.assertTrue(label['valid'])
        self.assertLess(abs(label['lower'] * 100 - 300), 25)
        self.assertLess(abs(label['upper'] * 100 - 300), 25)

    def test_linear_and_unobserved_future_knee_are_masked(self):
        for cell in (self.cell(), self.cell(900)):
            self.assertFalse(training_knee_target(cell, 100)['valid'])

    def test_spike_does_not_create_knee(self):
        cell = self.cell()
        cell['soh'][399] -= .15
        self.assertFalse(training_knee_target(cell, 100)['valid'])

    def test_missing_and_sparse_coverage_are_masked(self):
        self.assertFalse(training_knee_target(self.cell(120, 150), 100)['valid'])
        cell = self.cell(400)
        cell = {k: v[::50] for k, v in cell.items()}
        self.assertFalse(training_knee_target(cell, 100)['valid'])

    def test_historical_labels_do_not_affect_auxiliary_future_target(self):
        cell = self.cell(400)
        expected = training_knee_target(cell, 100)
        cell['soh'][:100] = np.nan
        self.assertEqual(expected, training_knee_target(cell, 100))

    def test_masked_and_in_interval_have_zero_gradient(self):
        p = torch.zeros(2, 7, requires_grad=True)
        lower = torch.tensor([0., 3.])
        upper = torch.tensor([0., 4.])
        loss = knee_interval_loss(p, lower, upper, torch.tensor([0., 1.]))
        self.assertEqual(float(loss.detach().sum()), 0.)
        loss.sum().backward()
        self.assertEqual(float(p.grad.abs().sum()), 0.)

    def test_observed_label_trains_same_knee_parameter(self):
        p = torch.zeros(1, 7, requires_grad=True)
        loss = knee_interval_loss(p, torch.tensor([1.]), torch.tensor([1.]), torch.ones(1))
        loss.sum().backward()
        self.assertGreater(float(p.grad[0, 3]), 0.)
        self.assertEqual(int((p.grad != 0).sum()), 1)


if __name__ == '__main__':
    unittest.main()
