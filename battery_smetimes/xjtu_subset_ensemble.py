"""Fixed whole-curve mean/median combinations of TRAIN-only subset models."""
import numpy as np
from battery_smetimes.xjtu_segment_features import predict as base_predict


def combine(predictions,kind):
    p=np.asarray(predictions,dtype=float)
    if p.ndim!=2 or not np.isfinite(p).all() or not p.shape[0] or not p.shape[1]:raise ValueError('invalid members')
    if kind=='mean':return p.mean(0)
    if kind=='median':return np.median(p,axis=0)
    if kind=='full' and len(p)==1:return p[0]
    raise ValueError('unregistered combination')


def predict(b,cell,cycle):
    if cell['physical_family']!=b['family']:raise ValueError('different family')
    return combine([base_predict(m,cell,cycle) for m in b['members']],b['kind'])
