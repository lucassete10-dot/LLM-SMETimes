from types import SimpleNamespace

import numpy as np

from train_mit_checkpoint_rul_v2 import diagnostic_features


def test_diagnostic_features_do_not_read_future_capacity():
    cycle = np.arange(1, 111)
    soh = 1 - cycle * .001
    cell = SimpleNamespace(cycle=cycle, soh=soh,
                           soh_available=np.ones(len(cycle), dtype=bool),
                           delta=np.tile(cycle[:, None], (1, 11)).astype(float))
    record = {'charge_rate_steps_c': [5.4, 3.0],
              'charge_protocol': '5.4C(50%)-3C'}
    before = diagnostic_features(cell, record, 50)
    changed = soh.copy()
    changed[cycle > 50] = -999
    after = diagnostic_features(SimpleNamespace(cycle=cycle, soh=changed,
                                soh_available=cell.soh_available, delta=cell.delta),
                                record, 50)
    assert np.array_equal(before, after)


def test_diagnostic_features_require_all_checkpoint_measurements():
    cycle = np.array([10, 20, 30, 50])
    cell = SimpleNamespace(cycle=cycle, soh=np.ones(4),
                           soh_available=np.ones(4, dtype=bool),
                           delta=np.zeros((4, 11)))
    assert diagnostic_features(cell, {}, 50) is None
