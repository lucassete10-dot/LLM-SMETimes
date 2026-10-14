"""A TRAIN donor's knee curve with C1 tangent extension after observed support."""
import numpy as np


def curve(parameters,record_end,cycles):
    c=np.asarray(cycles,dtype=float)
    if c.ndim!=1 or not len(c) or not np.isfinite(c).all() or np.any(c<=100):raise ValueError('invalid future')
    if not np.isfinite(record_end) or record_end<101:raise ValueError('invalid donor support')
    a,r,b,k=parameters;h=(c-100)/100;he=(record_end-100)/100
    value=a-r*h-b*np.maximum(h-k,0)**2
    boundary=a-r*he-b*max(he-k,0)**2
    derivative=(-r-2*b*max(he-k,0))/100
    return np.where(c>record_end,boundary+derivative*(c-record_end),value)
