import unittest

import torch

from battery_smetimes.models.temporal import (
    BatteryHealthGRU, BatteryHealthTCN, CausalTCN, WaveformHealthGRU,
    WaveformHealthTCN,
)


class TemporalModelTest(unittest.TestCase):
    def test_physics_decoder_is_nonincreasing_in_horizon(self):
        model = BatteryHealthGRU(138, physics_future=True)
        signal = torch.zeros(3, 16, 138)
        mask = torch.ones(3, 16, dtype=torch.bool)
        metadata = torch.zeros(3, 6)
        output = model(signal, mask, metadata, horizons=(10, 50, 100))
        curve = output["future_soh"]
        self.assertEqual(tuple(curve.shape), (3, 3))
        self.assertTrue(torch.all(curve[:, 0] <= output["current_soh"]))
        self.assertTrue(torch.all(curve[:, 1] <= curve[:, 0]))
        self.assertTrue(torch.all(curve[:, 2] <= curve[:, 1]))
        self.assertLess(sum(p.numel() for p in model.parameters()), 100_000)

    def test_direct_baseline_does_not_expose_future_head(self):
        model = BatteryHealthGRU(5, physics_future=False)
        output = model(torch.zeros(2, 4, 5), torch.ones(2, 4, dtype=torch.bool),
                       torch.zeros(2, 6))
        self.assertNotIn("future_soh", output)

    def test_waveform_cycle_encoder_outputs_physical_curve(self):
        model = WaveformHealthGRU(138, physics_future=True)
        signal = torch.zeros(2, 3, 138)
        mask = torch.ones(2, 3, dtype=torch.bool)
        metadata = torch.zeros(2, 6)
        wave = torch.zeros(2, 3, 2, 4, 121)
        wave[:, :, :, 0, :] = 3.7
        wave[:, :, :, 2, :] = 25
        output = model(signal, mask, metadata, (10, 50, 100), wave)
        self.assertTrue(torch.isfinite(output["current_soh"]).all())
        self.assertTrue(torch.all(output["future_soh"][:, 2] <=
                                  output["future_soh"][:, 0]))
        self.assertLess(sum(p.numel() for p in model.parameters()), 100_000)

    def test_tcn_cannot_see_later_diagnostics(self):
        torch.manual_seed(7)
        model = CausalTCN(hidden=8).eval()
        first = torch.randn(2, 16, 8)
        altered = first.clone()
        altered[:, 8:] += 100
        mask = torch.ones(2, 16, dtype=torch.bool)
        with torch.no_grad():
            original = model(first, mask)
            changed = model(altered, mask)
        torch.testing.assert_close(original[:, :8], changed[:, :8])

    def test_tcn_physics_variants_have_same_output_contract(self):
        signal = torch.zeros(2, 16, 138)
        mask = torch.ones(2, 16, dtype=torch.bool)
        metadata = torch.zeros(2, 6)
        waveform = torch.zeros(2, 16, 2, 4, 121)
        for model, extra in (
            (BatteryHealthTCN(138, physics_future=True), {}),
            (WaveformHealthTCN(138, physics_future=True),
             {"waveform": waveform}),
        ):
            output = model(signal, mask, metadata, horizons=(10, 50, 100),
                           **extra)
            self.assertEqual(tuple(output["future_soh"].shape), (2, 3))
            self.assertTrue(torch.isfinite(output["future_soh"]).all())
            self.assertTrue(torch.all(output["future_soh"][:, 2] <=
                                      output["future_soh"][:, 1]))

    def test_charge_only_waveform_encoder_rejects_discharge_stage(self):
        model = WaveformHealthGRU(69, physics_future=True, stages=1)
        signal = torch.zeros(2, 4, 69)
        mask = torch.ones(2, 4, dtype=torch.bool)
        metadata = torch.zeros(2, 6)
        charge = torch.zeros(2, 4, 1, 4, 121)
        output = model(signal, mask, metadata, (10, 50), charge)
        self.assertEqual(tuple(output["future_soh"].shape), (2, 2))
        with self.assertRaisesRegex(ValueError, "1, 4, 121"):
            model(signal, mask, metadata, (10, 50),
                  torch.zeros(2, 4, 2, 4, 121))


if __name__ == "__main__":
    unittest.main()
