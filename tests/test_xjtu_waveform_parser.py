import unittest

import numpy as np

from parsers.xjtu import extract_cell_waveforms, stage_waveform


def synthetic_cycle(description="1C discharge"):
    time = np.tile(np.arange(121) / 60, 3)
    return {
        "description": description,
        "relative_time_min": time,
        "voltage_V": np.r_[np.linspace(3.5, 4.1, 121),
                           np.full(121, 4.1), np.linspace(4.0, 3.4, 121)],
        "current_A": np.r_[np.full(121, 1.0), np.zeros(121),
                           np.full(121, -1.0)],
        "temperature_C": np.full(363, 25.0),
    }


class XJTUWaveformParserTest(unittest.TestCase):
    def test_charge_discharge_and_causal_throughput(self):
        cycles = [synthetic_cycle(), synthetic_cycle("other")]
        index, wave = extract_cell_waveforms(cycles, batch=1)
        self.assertEqual(index.tolist(), [1])
        self.assertEqual(wave.shape, (1, 2, 4, 121))
        self.assertAlmostEqual(float(wave[0, 0, 3, -1]), 120 / 3600, places=5)
        self.assertAlmostEqual(float(wave[0, 1, 3, -1]), 120 / 3600, places=5)
        self.assertAlmostEqual(float(wave[0, 0, 3, 0]), 0)

    def test_missing_stage_or_impossible_temperature_is_rejected(self):
        example = synthetic_cycle()
        example["temperature_C"][:] = -273.15
        self.assertIsNone(stage_waveform(example, 0))
        example = synthetic_cycle()
        example["voltage_V"] = example["voltage_V"][:100]
        with self.assertRaises(ValueError):
            stage_waveform(example, 0)


if __name__ == "__main__":
    unittest.main()
