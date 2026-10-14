import numpy as np

from evaluate_mit_policy_rul_v2 import policy_values


def test_mit_policy_parses_two_stage_rates_and_transition():
    values = policy_values({'charge_rate_steps_c': [5.4, 3.0],
                            'charge_protocol': '5.4C(50%)-3C'})
    assert np.allclose(values, [5.4, 3.0, .5, 4.2, -2.4])


def test_mit_policy_recovers_missing_second_stage_suffix():
    values = policy_values({'charge_rate_steps_c': [4.0],
                            'charge_protocol': '4C(31%)-5'})
    assert np.allclose(values[:3], [4.0, 5.0, .31])
