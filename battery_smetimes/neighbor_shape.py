"""Borrow degradation shape without transferring a reference's absolute SOH."""
import numpy as np


def mix(primary,reference,primary_anchor,reference_anchor,weight):
    p,q=np.asarray(primary,dtype=float),np.asarray(reference,dtype=float)
    a,b=float(primary_anchor),float(reference_anchor)
    if p.shape!=q.shape or p.ndim!=1 or not p.size:
        raise ValueError('incompatible curve arrays')
    if not 0<=weight<=1 or not np.isfinite(np.r_[p,q,a,b]).all():
        raise ValueError('nonfinite curve or invalid weight')
    return (1-weight)*p+weight*(a+q-b)
