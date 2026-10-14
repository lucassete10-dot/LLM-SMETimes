"""Causality, label isolation, and trained artifact regression checks."""
import unittest
from dataclasses import replace
from pathlib import Path
import joblib
import numpy as np
import torch
from battery_smetimes_regularized import (Features, RegularizedSMETimes, load_data,
                                         strict_prefix, predict_from_features)


class BatteryInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.root = Path('battery_smetimes_regularized_results')
        cls.batteries = load_data(Path(r'D:\LLM\NASA\raw\extracted'))
        cls.semantic = {k: v.float().numpy() for k,v in torch.load(
            next(cls.root.glob('llama_prefix_*.pt')), weights_only=True).items()}
        cls.prep = Features()
        cls.prep.__dict__.update(joblib.load(cls.root/'B0018/preprocessing.joblib'))

    def test_prefix_ignores_later_samples_and_capacity(self):
        time = np.arange(0, 2001, 37.)
        data = {'Time': time, 'Voltage_measured': 4.2-time*.0002,
                'Current_measured': time*0-2, 'Temperature_measured': 24+time*.001,
                'Capacity': 1.8}
        expected = strict_prefix(data)
        for channel in ['Voltage_measured', 'Current_measured', 'Temperature_measured']:
            data[channel][time > 1000] = 9999
        data['Capacity'] = 9999
        np.testing.assert_array_equal(expected, strict_prefix(data))

    def test_no_soh_label_in_inputs(self):
        battery = self.batteries['B0018']
        changed = replace(battery, soh=np.full_like(battery.soh, 99), capacity_ah=battery.capacity_ah*99)
        a = self.prep.transform(battery, self.semantic['B0018'])
        b = self.prep.transform(changed, self.semantic['B0018'])
        for i in range(4):
            torch.testing.assert_close(a[i], b[i])

    def test_future_cycles_do_not_change_previous_predictions(self):
        battery = self.batteries['B0018']
        x = battery.features.copy()
        marks = self.semantic['B0018'].copy()
        original = predict_from_features(self.root/'B0018', x, marks)
        x[70:] = 100
        marks[70:] = 100
        altered = predict_from_features(self.root/'B0018', x, marks)
        np.testing.assert_allclose(original[:70], altered[:70], atol=1e-7)

    def test_reload_matches_saved_predictions(self):
        saved = np.load(self.root/'original_split/predictions.npz')
        for semantic, name in [(True, 'smetimes_fusion'), (False, 'numeric_ablation')]:
            pred = predict_from_features(self.root/'original_split', self.batteries['B0018'].features,
                                         self.semantic['B0018'], semantic)
            np.testing.assert_allclose(pred, saved[name], atol=1e-7)

    def test_outer_scaler_uses_only_training_cells(self):
        x = np.concatenate([self.batteries[k].features for k in ['B0005','B0006','B0007']])
        np.testing.assert_allclose(self.prep.scaler.mean_, x.astype('float64').mean(0), atol=1e-10)

    def test_numeric_ablation_ignores_semantic_marks(self):
        model = RegularizedSMETimes(False).eval()
        data = self.prep.transform(self.batteries['B0018'], self.semantic['B0018'])
        with torch.no_grad():
            a = model(*data[:4])[0]
            data[2] += 999
            b = model(*data[:4])[0]
        torch.testing.assert_close(a, b)


if __name__ == '__main__':
    unittest.main()
