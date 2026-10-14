from types import SimpleNamespace

import numpy as np

from evaluate_mit_checkpoint_signal_rul_v2 import feature


def test_signal_checkpoint_feature_does_not_read_future_soh():
    cycle = np.arange(1, 101)
    cell = SimpleNamespace(cycle=cycle, delta=np.zeros((len(cycle), 11)))
    record = {'charge_protocol': '5.4C(50%)-3C', 'charge_rate_steps_c': [5.4, 3.]}
    predicted_soh = 1-.001*cycle
    first = feature(cell, record, predicted_soh)
    changed = predicted_soh.copy()
    changed[cycle > 50] = -100
    assert np.array_equal(first, feature(cell, record, changed))
