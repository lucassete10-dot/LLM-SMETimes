"""Fixed waveform-shape projections within existing label-free time windows."""
import warnings
import numpy as np
from battery_smetimes.tail_boost import is_nca,inputs


def channel_slices(cell,width):
    if is_nca(cell):
        if width!=32:raise ValueError('NCA raw feature contract mismatch')
        return [slice(0,11),slice(16,27)]
    dataset=cell['metadata']['dataset']
    if dataset=='NASA' and width==44:return [slice(0,11),slice(11,22),slice(22,33)]
    if dataset=='MIT' and width==51:return [slice(0,11),slice(11,22),slice(22,33)]
    if dataset=='XJTU' and width==46:return [slice(a,a+5) for a in (0,5,10,23,28,33)]
    raise ValueError('unsupported waveform shape contract')


def projected_rows(cell):
    cycle=np.asarray(cell['early_cycle'] if is_nca(cell) else cell['cycle'],dtype=float)
    raw=np.asarray(cell['early_raw'] if is_nca(cell) else cell['raw'],dtype=float)
    keep=(cycle>0)&(cycle<=100);cycle=cycle[keep];raw=raw[keep]
    if len(cycle)<2 or not np.isfinite(cycle).all() or np.any(np.diff(cycle)<=0) or np.isinf(raw).any():raise ValueError('invalid early waveform history')
    slices=channel_slices(cell,raw.shape[1]);parts=[]
    for sl in slices:
        values=raw[:,sl];grid=np.linspace(-1,1,values.shape[1]);basis=np.polynomial.legendre.legvander(grid,3)
        coefficients=np.full((len(values),3),np.nan)
        for i,row in enumerate(values):
            finite=np.isfinite(row)
            if finite.sum()>=4:
                values=np.linalg.lstsq(basis[finite],row[finite],rcond=None)[0][1:]
                # Suppress only floating-point projection roundoff, not measured
                # noise or SOH errors. Constant CC channels must remain constant.
                threshold=64*np.finfo(float).eps*max(1.,float(np.max(np.abs(row[finite]))))
                values[np.abs(values)<=threshold]=0
                coefficients[i]=values
        parts.append(coefficients)
    return cycle,np.column_stack(parts)


def slope(cycle,z):
    result=np.full(z.shape[1],np.nan)
    for j in range(z.shape[1]):
        ok=np.isfinite(z[:,j]);t=cycle[ok];y=z[ok,j]
        if len(t)>=2 and np.ptp(t)>0:
            t=t-t.mean();result[j]=100*np.dot(t,y-y.mean())/np.dot(t,t)
    return result


def vector(cell):
    cycle,z=projected_rows(cell)
    reference=5 if is_nca(cell) or cell['metadata']['dataset']=='XJTU' else 1
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        initial=np.nanmedian(z[:reference],axis=0);last=np.nanmedian(z[-3:],axis=0);spread=np.nanstd(z,axis=0)
    return np.r_[initial,last,last-initial,slope(cycle,z),spread,slope(cycle[-10:],z[-10:])]


def matrix(cell,cycle,augmented):
    base=inputs(cell,cycle)
    if not augmented:return base
    return np.column_stack([base,np.tile(vector(cell),(len(cycle),1))])
