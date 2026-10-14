"""Global convex blend with rate dynamics, no query-specific fitting."""
import numpy as np
from battery_smetimes.fade_rate import predict as rate_predict
from battery_smetimes.tail_selection import predict as primary_predict


def make(primary,rate,weight):
    if not np.isfinite(weight) or not 0<=weight<=1:raise ValueError('invalid weight')
    if primary['family']!=rate['family'] or primary['train_ids']!=rate['train_ids']:raise ValueError('component family or membership mismatch')
    return dict(primary=primary,rate=rate,weight=float(weight),family=rate['family'],train_ids=rate['train_ids'])


def predict(b,cell,cycle):
    if cell['physical_family']!=b['family']:raise ValueError('different physical family')
    p=primary_predict(b['primary'],cell,cycle);q=rate_predict(b['rate'],cell,cycle)
    y=(1-b['weight'])*p+b['weight']*q
    if not np.isfinite(y).all():raise ValueError('nonfinite forecast')
    return y
