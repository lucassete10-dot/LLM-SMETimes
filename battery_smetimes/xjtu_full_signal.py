"""Completed past diagnostics only: V/I/t/T features, never capacity labels."""
import numpy as np

WINDOWS = [(3.2,3.4),(3.4,3.6),(3.6,3.8),(3.8,4.0),(4.0,4.2)]


def cycle_features(row):
    t,v,i,temp=[np.asarray(row[k],float).reshape(-1) for k in
        ('relative_time_min','voltage_V','current_A','temperature_C')]
    if not (t.shape==v.shape==i.shape==temp.shape):
        raise ValueError('unequal signal shapes')
    cuts=np.flatnonzero(t==0)
    if len(cuts)<3:raise ValueError('missing charge/discharge stages')
    result=[]
    for stage,sign in [(0,1),(2,-1)]:
        a=cuts[stage];b=cuts[stage+1] if stage+1<len(cuts) else len(t)
        ts,vs,amps,temps=[x[a:b] for x in (t,v,i,temp)]
        if len(ts)<3 or not all(np.isfinite(x).all() for x in (ts,vs,amps,temps)):
            raise ValueError('insufficient or nonfinite stage')
        dt=np.diff(ts)*60
        if np.any(dt<=0):raise ValueError('nonmonotone stage time')
        # No sorting in voltage/time and no bridge across stages.
        vm=(vs[:-1]+vs[1:])/2;im=(amps[:-1]+amps[1:])/2;tm=(temps[:-1]+temps[1:])/2
        active=(amps[:-1]*sign>0)&(amps[1:]*sign>0)
        if dt[active].sum()<.8*dt.sum():raise ValueError('unexpected stage current sign')
        w=dt/dt.sum();mean_v=np.sum(w*vm)
        result.extend([dt.sum(),np.sum(np.abs(im)*dt)/3600,np.sum(w*np.abs(im)),
            mean_v,np.sqrt(np.sum(w*(vm-mean_v)**2)),np.sum(w*tm),temps.max()])
        for lo,hi in WINDOWS:
            take=active&(vm>=lo)&(vm<hi)
            duration=dt[take].sum()
            result.extend([duration,np.sum(np.abs(im[take])*dt[take])/3600,
                float(np.sum(vm[take]*dt[take])/duration) if duration>0 else np.nan])
    return np.asarray(result,float)
