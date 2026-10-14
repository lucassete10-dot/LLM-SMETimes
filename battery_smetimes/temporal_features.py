"""Fixed source-cycle windows, never life-relative or capacity-derived."""
import warnings
import numpy as np
from battery_smetimes.tail_boost import inputs, is_nca


def temporal_vector(cell):
    cycle = np.asarray(cell['early_cycle'] if is_nca(cell) else cell['cycle'])
    raw = np.asarray(cell['early_raw'] if is_nca(cell) else cell['raw'], dtype=float)
    take = (cycle > 0) & (cycle <= 100)
    c, z = cycle[take].astype(float), raw[take]
    if not len(c) or np.any(np.isinf(z)): raise ValueError('invalid early signals')
    width = z.shape[1]; pieces = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for start, stop in ((0,25), (25,50), (50,75), (75,100)):
            part = z[(c > start) & (c <= stop)]
            pieces.extend((np.nanmedian(part, axis=0), np.nanstd(part, axis=0)) if len(part) else
                (np.full(width, np.nan), np.full(width, np.nan)))
    slopes = []
    for start, stop in ((0,50), (50,100)):
        keep = (c > start) & (c <= stop); x = z[keep]; t = c[keep]
        slope = np.full(width, np.nan)
        for j in range(width):
            ok = np.isfinite(x[:,j]); tt = t[ok]
            if len(tt) >= 2 and np.ptp(tt) > 0:
                tc = tt-tt.mean(); slope[j] = 100*np.dot(tc, x[ok,j]-x[ok,j].mean())/np.dot(tc,tc)
        slopes.append(slope)
    return np.concatenate(pieces + slopes + [slopes[1]-slopes[0]])


def matrix(cell, future, augmented):
    base = inputs(cell, future)
    if not augmented: return base
    return np.column_stack((base, np.tile(temporal_vector(cell), (len(future),1))))
