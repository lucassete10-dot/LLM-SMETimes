"""Signal-inferred time to 90% SOH; train-supervised curve-coordinate alignment."""
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from battery_smetimes import voltage_interval_features as physics


def crossing90(c,y,origin):
    c,y=np.asarray(c,float),np.asarray(y,float)
    # A sustained 5-observation crossing avoids using a one-point capacity spike.
    for j in range(len(y)-4):
        if np.all(y[j:j+5]<=.9):
            remaining=float(c[j]-origin)
            if remaining<=0:raise ValueError('90 percent crossing not future')
            return remaining
    raise ValueError('training cell has censored 90 percent crossing')


def fit(cells,ids,targets,origin,seed):
    ids=list(ids);families={cells[k]['physical_family'] for k in ids}
    if len(families)!=1:raise ValueError('mixed family')
    duration={cid:crossing90(*targets[cid],origin) for cid in ids}
    x=np.stack([physics.features(cells[cid],origin) for cid in ids])
    clock_imputer=SimpleImputer(strategy='median',keep_empty_features=True)
    clock_scaler=StandardScaler().fit(clock_imputer.fit_transform(x))
    clock=ExtraTreesRegressor(n_estimators=20,min_samples_leaf=2,max_features=.8,random_state=seed,n_jobs=2)
    clock.fit(clock_scaler.transform(clock_imputer.transform(x)),np.log([duration[cid] for cid in ids]))
    xs,ys,ws=[],[],[]
    for cid in ids:
        c,y=targets[cid];ii=np.unique(np.linspace(0,len(c)-1,min(96,len(c))).round().astype(int))
        phase=origin+(c[ii]-origin)*100/duration[cid]
        xs.append(physics.inputs(cells[cid],origin,phase));ys.append(y[ii]);ws.append(np.full(len(ii),1/len(ii)))
    xx,yy,w=np.concatenate(xs),np.concatenate(ys),np.concatenate(ws);w*=len(w)/w.sum()
    imputer=SimpleImputer(strategy='median',keep_empty_features=True);z=imputer.fit_transform(xx)
    scaler=StandardScaler().fit(z)
    curve=ExtraTreesRegressor(n_estimators=60,min_samples_leaf=1,max_features=.8,random_state=seed,n_jobs=2)
    curve.fit(scaler.transform(z),yy,sample_weight=w)
    return dict(train_ids=ids,origin=origin,seed=seed,family=next(iter(families)),training_durations=duration,
        clock=clock,clock_imputer=clock_imputer,clock_scaler=clock_scaler,
        model=curve,imputer=imputer,scaler=scaler,budget=dict(clock_trees=20,curve_trees=60,total_trees=80),
        limitation='Training uses observed 90%-crossing supervision; inference uses a predicted crossing. This teacher-inference mismatch is evaluated, not concealed.')


def inferred_duration(bundle,cell):
    f=physics.features(cell,bundle['origin'])[None]
    return float(np.exp(bundle['clock'].predict(bundle['clock_scaler'].transform(bundle['clock_imputer'].transform(f)))[0]))


def predict(bundle,cell,cycle):
    c=np.asarray(cycle,float);origin=bundle['origin']
    if cell['physical_family']!=bundle['family']:raise ValueError('physical family mismatch')
    if c.ndim!=1 or not len(c) or not np.isfinite(c).all() or np.any(c<=origin):raise ValueError('invalid future')
    phase=origin+(c-origin)*100/inferred_duration(bundle,cell)
    return bundle['model'].predict(bundle['scaler'].transform(bundle['imputer'].transform(physics.inputs(cell,origin,phase))))
