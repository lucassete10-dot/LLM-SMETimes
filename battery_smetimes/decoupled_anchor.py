"""Signal-only anchor correction preserving an entire relative fade curve."""
import numpy as np
from battery_smetimes.pooled_anchor import anchor
from battery_smetimes.fade_rate import predict as fade


def make(pooled,rate):
    if pooled['family']!=rate['family'] or pooled['train_ids']!=rate['train_ids']:raise ValueError('component family or training membership mismatch')
    if rate['learned_anchor'] or pooled['seed']!=rate['seed']:raise ValueError('requires matched fixed-reference rate component')
    return dict(pooled=pooled,rate=rate,family=rate['family'],train_ids=list(rate['train_ids']))


def predict(bundle,cell,cycle):
    if cell['physical_family']!=bundle['family']:raise ValueError('different physical family')
    a=anchor(bundle['pooled'],cell);origin=fade(bundle['rate'],cell,np.array([101.]))[0]
    p=a+fade(bundle['rate'],cell,cycle)-origin
    if not np.isfinite(p).all():raise ValueError('nonfinite forecast')
    return p
