"""Features from completed *past* 4C discharge traces; no capacity labels."""
import numpy as np
from prepare_mit import discharge_slice

WINDOWS = [(2.0,2.5),(2.5,2.8),(2.8,3.0),(3.0,3.1),(3.1,3.2),(3.2,3.3),(3.3,3.4),(3.4,3.6)]


def cycle_features(t, i, v, temp):
    t,i,v,temp = [np.asarray(x,float).reshape(-1) for x in (t,i,v,temp)]
    if not (t.shape==i.shape==v.shape==temp.shape):
        raise ValueError('signal length mismatch')
    if not np.isfinite(np.c_[t,i,v]).all():
        raise ValueError('nonfinite electrical signal')
    run=discharge_slice(t,i,v)
    if run is None:raise ValueError('missing main discharge')
    t,i,v,temp=[x[run] for x in (t,i,v,temp)]
    dt=np.diff(t)*60
    if np.any(dt < -1e-6):raise ValueError('nonmonotone discharge time')
    dt=np.maximum(dt,0)
    if dt.sum()<=0:raise ValueError('zero duration')
    im=(i[1:]+i[:-1])/2;vm=(v[1:]+v[:-1])/2;tm=(temp[1:]+temp[:-1])/2
    charge=np.abs(im)*dt/3600;w=dt/dt.sum()
    # Integrals use measured current/time, never Qd or summary capacity.
    out=[dt.sum(),charge.sum(),np.sum(charge*vm),np.sum(w*vm),
         np.sum(w*np.abs(im)),np.sum(w*tm),np.nanmax(temp),v[0],v[-1]]
    for lo,hi in WINDOWS:
        take=(vm>=lo)&(vm<hi)
        out.extend([dt[take].sum(),charge[take].sum()])
    return np.asarray(out)
