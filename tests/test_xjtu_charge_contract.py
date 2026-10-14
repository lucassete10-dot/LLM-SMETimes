import unittest
import json
from pathlib import Path

import numpy as np

from battery_v2_features import make_inference_features
from prepare_xjtu import FEATURE_NAMES


class XjtuChargeContractTest(unittest.TestCase):
    def test_stage_names_match_the_actual_23_plus_23_storage(self):
        self.assertEqual(len(FEATURE_NAMES), 46)
        self.assertTrue(all(name.startswith("charge_") for name in FEATURE_NAMES[:23]))
        self.assertTrue(all(name.startswith("discharge_") for name in FEATURE_NAMES[23:]))
        self.assertEqual(FEATURE_NAMES[15], "charge_voltage_drop_v")
        self.assertEqual(FEATURE_NAMES[22], "charge_early_throughput_ah")
        self.assertEqual(FEATURE_NAMES[23], "discharge_voltage_V_000s")
        manifest = json.loads((Path(__file__).resolve().parents[1] /
                               "battery_final_models/xjtu_all_batches/training_manifest.json")
                              .read_text(encoding="utf-8"))
        self.assertEqual(manifest["base_feature_names"], FEATURE_NAMES)

    def test_discharge_values_cannot_enter_charge_only_history(self):
        raw = np.arange(3 * 46, dtype=np.float32).reshape(3, 46)
        altered = raw.copy()
        altered[:, 23:] += 1000
        first = make_inference_features(raw, [1, 2, 3], 1)
        second = make_inference_features(altered, [1, 2, 3], 1)
        np.testing.assert_array_equal(first.matrix("charge_signal_history"),
                                      second.matrix("charge_signal_history"))
        self.assertEqual(first.matrix("charge_signal_history").shape, (3, 69))
        self.assertFalse(np.array_equal(first.matrix("signal_history"),
                                        second.matrix("signal_history")))


if __name__ == "__main__":
    unittest.main()
