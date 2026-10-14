"""Registered loss weights; unchanged raw-label, mask and query semantics."""
import math


def combine_curve_losses(point, start, drops, tail, phase, *, drop_weight=.5, tail_weight=.5):
    for value in (drop_weight, tail_weight):
        if not math.isfinite(value) or value < 0:
            raise ValueError('curve loss weights must be finite and nonnegative')
    if phase not in ('short', 'full'):
        raise ValueError('unknown supervision phase')
    return point + start + drop_weight * drops + (tail_weight * tail if phase == 'full' else 0)
