import numpy as np

from evaluate_xjtu_throughput_rul_v2 import STAGE_WIDTH, stage_at_charge


def test_voltage_is_compared_at_equal_early_throughput():
    n = 21
    times = np.arange(n)*30
    def stage(current):
        q = current*times/3600
        voltage = 4-.5*q
        return np.r_[voltage, np.full(n, current), np.full(n, 25.),
                     np.zeros(STAGE_WIDTH-3*n)]
    slow = stage_at_charge(stage(1.))
    fast = stage_at_charge(stage(2.))
    assert np.allclose(slow[:9], fast[:9], atol=1e-6)
    assert slow[27] > fast[27]  # time to 0.08 Ah differs by current.


def test_short_prefix_is_rejected():
    n = 21
    stage = np.r_[np.full(n, 4.), np.full(n, .1), np.full(n, 25.),
                  np.zeros(STAGE_WIDTH-3*n)]
    try:
        stage_at_charge(stage)
    except ValueError:
        return
    raise AssertionError('insufficient observed early throughput was accepted')
