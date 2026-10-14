"""Causality and missing-signal regression tests for the challenger features."""

from types import SimpleNamespace

import numpy as np

from paper_inspired_battery_v2 import time_gated_matrix


def test_time_gated_features_cannot_see_future():
    raw = np.arange(24, dtype=np.float32).reshape(6, 4)
    cell = SimpleNamespace(cycle=np.array([1, 2, 5, 9, 10, 20]), raw=raw, delta=raw - raw[0])
    original = time_gated_matrix(cell)
    changed = raw.copy()
    changed[4:] = 10000
    cell_changed = SimpleNamespace(cycle=cell.cycle, raw=changed, delta=changed - changed[0])
    np.testing.assert_allclose(original[:4], time_gated_matrix(cell_changed)[:4])


def test_time_gated_features_survive_missing_observation():
    raw = np.array([[1., 2.], [np.nan, 3.], [2., 4.]], dtype=np.float32)
    cell = SimpleNamespace(cycle=np.array([1, 3, 10]), raw=raw, delta=raw - raw[0])
    result = time_gated_matrix(cell)
    assert result.shape == (3, 9)
    assert result[1, -1] == .5
    assert np.isfinite(result[2]).all()
