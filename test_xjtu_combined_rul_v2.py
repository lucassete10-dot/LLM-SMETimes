from types import SimpleNamespace

import numpy as np

from evaluate_xjtu_combined_rul_v2 import matrix


def test_combined_features_do_not_read_future_diagnostics_or_usage():
    cycle = np.arange(1, 11)
    raw = np.tile(cycle[:, None], (1, 3)).astype('float32')
    usage = np.tile(cycle[:, None], (1, 16)).astype('float32')
    cell = SimpleNamespace(cell_id='synthetic', cycle=cycle, reference_count=2)
    first = matrix(cell, raw, usage)
    changed_raw, changed_usage = raw.copy(), usage.copy()
    changed_raw[5:] = -1000
    changed_usage[5:] = -1000
    second = matrix(cell, changed_raw, changed_usage)
    assert np.allclose(first[:5], second[:5], equal_nan=True)
