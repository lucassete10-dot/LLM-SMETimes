"""Predict reference pace from fixed-horizon model forecasts, not query labels."""
import numpy as np


def pace(primary_101,primary_201,reference_101,reference_201):
    a,b,c,d=map(float,(primary_101,primary_201,reference_101,reference_201))
    if not np.isfinite([a,b,c,d]).all():raise ValueError('nonfinite pace anchors')
    p,q=a-b,c-d
    return 1. if p<=.001 or q<=.001 else float(np.clip(p/q,.5,2.))


def map_cycle(cycle,ratio):
    x=np.asarray(cycle,dtype=float)
    if not np.isfinite(x).all() or np.any(x<101) or not .5<=ratio<=2:raise ValueError('invalid cycle or pace')
    return 101+ratio*(x-101)
