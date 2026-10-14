"""Visible signal statistics for explicitly registered external feature layouts."""
import warnings
import numpy as np


def external_signal_prefix(cell,origin,conditions=True):
    metadata=cell['metadata'];schema=metadata['signal_schema']
    if cell['reference_established_source_cycle']>origin:raise ValueError('label reference not established at origin')
    cycle=np.asarray(cell['cycle']);raw=np.asarray(cell['raw'],dtype=float)
    if raw.shape!=(len(cycle),schema['width']) or np.any(np.diff(cycle)<=0):raise ValueError('external signal schema mismatch')
    take=cycle<=origin;c=cycle[take];x=raw[take]
    if len(c)<schema['minimum_observations']:raise ValueError('insufficient signal history at requested origin')
    if np.isinf(x).any() or not np.isfinite(x).any(axis=1).all():raise ValueError('invalid visible external history')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        initial=np.nanmedian(x[:min(3,len(x))],axis=0);last=np.nanmedian(x[-3:],axis=0)
        xx=x[-10:];tt=c[-10:].astype(float);valid=np.isfinite(xx);n=valid.sum(0)
        tm=(valid*tt[:,None]).sum(0)/np.maximum(n,1);xm=np.nansum(xx,axis=0)/np.maximum(n,1)
        centered=tt[:,None]-tm;den=(centered**2*valid).sum(0)
        slope=np.divide(np.nansum(centered*(xx-xm),axis=0),den,out=np.full(x.shape[1],np.nan),where=(n>=2)&(den>0))*100
    signal=np.r_[last,initial,last-initial,slope,origin/100,c[-1]/100,len(c)/100,(c[-1]-c[0])/100]
    known=np.asarray(metadata['known_conditions'],float)
    if known.shape!=(5,) or np.isinf(known).any():raise ValueError('external known conditions must have five entries')
    return np.r_[signal,known].astype(np.float32) if conditions else signal.astype(np.float32)
