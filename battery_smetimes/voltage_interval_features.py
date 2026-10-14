"""Partial voltage-window charge throughput from observed prefix V/I/t only."""
import numpy as np
from battery_smetimes import continuous_length as point
from battery_smetimes import voltage_change as rich

WINDOWS={'charge':[(3.3,3.4),(3.4,3.5),(3.5,3.58)],
         'discharge':[(3.0,3.1),(3.1,3.2),(3.2,3.3)]}


def cycle_features(row):
    # Explicit whitelist: no capacity arrays, cycle lifetime or SOH labels.
    t=np.asarray(row['time_in_s'],float); v=np.asarray(row['voltage_in_V'],float); i=np.asarray(row['current_in_A'],float)
    if t.ndim!=1 or not(t.shape==v.shape==i.shape) or len(t)<3:
        raise ValueError('invalid signal shape')
    if not np.isfinite(t).all() or not np.isfinite(v).all() or not np.isfinite(i).all() or np.any(np.diff(t)<=0):
        raise ValueError('nonfinite or nonmonotone signal')
    dt=np.diff(t); vm=(v[:-1]+v[1:])/2; im=(i[:-1]+i[1:])/2
    out=[]
    for stage,windows in WINDOWS.items():
        sign=(i[:-1]>.1)&(i[1:]>.1) if stage=='charge' else (i[:-1]<-.1)&(i[1:]<-.1)
        for lo,hi in windows:
            mask=sign&(vm>=lo)&(vm<hi)
            if not mask.any():
                out.extend([0,0,np.nan,np.nan,np.nan,np.nan,0]); continue
            duration=dt[mask].sum(); weights=dt[mask]/duration
            avgi=np.sum(weights*np.abs(im[mask])); avgv=np.sum(weights*vm[mask])
            out.extend([duration,np.sum(abs(im[mask])*dt[mask])/3600,avgi,
                np.sqrt(np.sum(weights*(abs(im[mask])-avgi)**2)),avgv,
                np.sqrt(np.sum(weights*(vm[mask]-avgv)**2)),1])
    return np.array(out,float)


def features(cell,origin):
    cycle=np.asarray(cell['physics_cycle'],float); take=cycle<=origin
    c=cycle[take]; x=np.asarray(cell['physics_raw'],float)[take]
    if len(c)<3: raise ValueError('insufficient valid voltage-window observations')
    if not np.all(np.diff(c)>0) or np.isinf(x).any(): raise ValueError('invalid interval history')
    # Reuse established history summary, but keep the original V/I inputs intact.
    interval=dict(early_cycle=c,early_raw=x,conditions=np.array([]))
    return np.r_[point.features(cell,origin),point.features(interval,origin)]


def inputs(cell,origin,cycle):
    c=np.asarray(cycle,float); h=(c-origin)/100; f=features(cell,origin)
    return np.c_[np.broadcast_to(f,(len(c),len(f))),c/100,h,np.log1p(h)]


def fit(cells,ids,targets,origin,seed):
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    families={cells[k]['physical_family'] for k in ids}
    if len(families)!=1: raise ValueError('mixed family')
    xs,ys,ws=[],[],[]
    for cid in ids:
        c,y=targets[cid]; ii=np.unique(np.linspace(0,len(c)-1,min(96,len(c))).round().astype(int))
        xs.append(inputs(cells[cid],origin,c[ii])); ys.append(y[ii]); ws.append(np.full(len(ii),1/len(ii)))
    x,y,w=np.concatenate(xs),np.concatenate(ys),np.concatenate(ws); w*=len(w)/w.sum()
    imputer=SimpleImputer(strategy='median',keep_empty_features=True); z=imputer.fit_transform(x)
    scaler=StandardScaler().fit(z)
    model=ExtraTreesRegressor(n_estimators=80,min_samples_leaf=1,max_features=.8,random_state=seed,n_jobs=2).fit(scaler.transform(z),y,sample_weight=w)
    return dict(train_ids=list(ids),origin=origin,seed=seed,family=next(iter(families)),imputer=imputer,scaler=scaler,model=model)


def predict(bundle,cell,cycle):
    c=np.asarray(cycle,float)
    if cell['physical_family']!=bundle['family']: raise ValueError('physical family mismatch')
    if c.ndim!=1 or not len(c) or not np.isfinite(c).all() or np.any(c<=bundle['origin']): raise ValueError('invalid future')
    return bundle['model'].predict(bundle['scaler'].transform(bundle['imputer'].transform(inputs(cell,bundle['origin'],c))))
