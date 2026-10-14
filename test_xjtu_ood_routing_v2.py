from types import SimpleNamespace

import numpy as np

from evaluate_xjtu_ood_routing_v2 import initial_signature


def test_ood_signature_uses_only_first_15_diagnostics():
    raw = np.arange(20*4, dtype='float32').reshape(20, 4)
    before = initial_signature(SimpleNamespace(raw=raw))
    changed = raw.copy()
    changed[15:] = -1e4
    after = initial_signature(SimpleNamespace(raw=changed))
    assert np.array_equal(before, after)
