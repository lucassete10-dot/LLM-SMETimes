"""Synthetic saved-curve invariants; no claims of real model performance."""

import copy
from pathlib import Path
import tempfile
import unittest

import matplotlib.pyplot as plt
import numpy as np

from battery_smetimes.research_baselines import regression_metrics
from battery_smetimes.research_transfer_vi120 import stream_sha
from scripts.plot_soh_research_cell_atlas_v5 import curves_for_cell, draw_cell, MODELS, ORIGINS


class CellAtlasTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        seeds = [10, 11, 12, 13, 14]
        self.cycle = np.array([21., 30., 50., 101., 120.])
        self.truth = np.array([1., .98, .99, .85, .80])
        self.predictions = [self.truth + .001*(seed-9) for seed in seeds]
        mape = np.mean([regression_metrics(self.truth, value)["mape_percent"] for value in self.predictions])
        self.row = {"run_id": "toy", "training_seeds": seeds, "by_cell_training_seed_variation": [
            {"cell_id": "cell1", "fold": "MIT_fold1", "mape_percent": {"mean": mape}}]}
        self.atlas = {"rows": {("MIT", "ridge", 20): self.row}, "members": {("MIT", "cell1"): "MIT_fold1"},
            "runs": {"toy": self.run}, "units": {}, "cache": {}, "prediction_sources": {}}
        for seed, prediction in zip(seeds, self.predictions):
            folder = self.run / "MIT_fold1" / str(seed) / "ridge/test"
            folder.mkdir(parents=True)
            np.savez_compressed(folder / "predictions.npz", cell1__20__cycle=self.cycle,
                cell1__20__truth=self.truth, cell1__20__prediction=prediction)
            self.atlas["units"][("toy", "MIT_fold1", seed, "ridge")] = {"status": "passed static artifact/metric audit",
                "artifact_sha256": {str(Path("test") / "predictions.npz"): stream_sha(folder / "predictions.npz")}}

    def test_exact_all_points_seed_mean_sd_and_unsmoothed_capacity_recovery(self):
        curve = curves_for_cell(self.atlas, "MIT", "cell1", 20, "ridge")
        np.testing.assert_array_equal(curve["cycle"], self.cycle)
        np.testing.assert_array_equal(curve["truth"], self.truth)
        self.assertGreater(curve["truth"][2], curve["truth"][1])
        np.testing.assert_array_equal(curve["predictions"], np.asarray(self.predictions))
        np.testing.assert_allclose(curve["sd"], np.std(self.predictions, axis=0, ddof=1))
        self.assertEqual(len(self.atlas["prediction_sources"]), 5)

    def test_audited_bytes_fold_or_report_error_change_rejected(self):
        for kind in ("hash", "fold", "metric"):
            atlas = copy.deepcopy(self.atlas)
            if kind == "hash":
                atlas["units"][("toy", "MIT_fold1", 10, "ridge")]["artifact_sha256"][str(Path("test") / "predictions.npz")] = "changed"
            elif kind == "fold":
                atlas["members"][("MIT", "cell1")] = "MIT_fold2"
            else:
                atlas["rows"][("MIT", "ridge", 20)]["by_cell_training_seed_variation"][0]["mape_percent"]["mean"] += .01
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                curves_for_cell(atlas, "MIT", "cell1", 20, "ridge")

    def test_seed_grid_or_truth_mismatch_is_not_interpolated_away(self):
        path = self.run / "MIT_fold1/14/ridge/test/predictions.npz"
        for field in ("cycle", "truth"):
            arrays = {"cell1__20__cycle": self.cycle.copy(), "cell1__20__truth": self.truth.copy(), "cell1__20__prediction": self.predictions[-1]}
            arrays["cell1__20__"+field][-1] += .01
            np.savez_compressed(path, **arrays)
            atlas = copy.deepcopy(self.atlas)
            atlas["units"][("toy", "MIT_fold1", 14, "ridge")]["artifact_sha256"][str(Path("test") / "predictions.npz")] = stream_sha(path)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "different target"):
                curves_for_cell(atlas, "MIT", "cell1", 20, "ridge")

    def test_missing_eligible_curve_is_explicit_not_a_zero_forecast(self):
        self.row["by_cell_training_seed_variation"] = []
        self.assertIsNone(curves_for_cell(self.atlas, "MIT", "cell1", 20, "ridge"))

    def test_draw_keeps_model_points_and_marks_coverage_gaps(self):
        value = curves_for_cell(self.atlas, "MIT", "cell1", 20, "ridge")
        curves = {(origin, model): None if origin == 20 else value for origin in ORIGINS for model in MODELS}
        fig = draw_cell("XJTU", "toy", curves)
        self.addCleanup(plt.close, fig)
        self.assertEqual(len(fig.axes), 3)
        self.assertIn("Not eligible", fig.axes[0].get_title())
        self.assertTrue(all(not line.get_visible() for line in fig.axes[0].get_ygridlines()))
        self.assertTrue(any("temperature unavailable" in text.get_text() for text in fig.texts))
        for axis in fig.axes[1:]:
            observed = next(line for line in axis.get_lines() if len(line.get_xdata()) == len(self.cycle))
            np.testing.assert_array_equal(observed.get_ydata(), self.truth)


if __name__ == "__main__":
    unittest.main()
