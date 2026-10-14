"""TRAIN-only curve-parameter supervision, early V/I-only inference."""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesRegressor
from battery_smetimes.tail_boost import inputs
from battery_smetimes.curve_fidelity import shape_parameters
from battery_smetimes.knee_forecast import fit_knee,trajectory


def fit(cells,ids,parameters,seed):
    ids=list(ids);family=cells[ids[0]]['physical_family']
    if any(cells[k]['physical_family']!=family for k in ids):raise ValueError('different family')
    if parameters not in [dict(shape=s,leaf=l) for s in ('quadratic','knee') for l in (1,3)]:raise ValueError('unregistered candidate')
    x=np.stack([inputs(cells[k],[101.])[0] for k in ids])
    targets=[]
    for k in ids:
        c,y=cells[k]['cycle'],cells[k]['truth']
        targets.append(fit_knee(c,y) if parameters['shape']=='knee' else shape_parameters(c,y,2))
    y=np.stack(targets);scale=np.maximum(y.std(axis=0),.001)
    imputer=SimpleImputer(strategy='median',keep_empty_features=True)
    model=ExtraTreesRegressor(n_estimators=128,min_samples_leaf=parameters['leaf'],random_state=seed,n_jobs=2).fit(imputer.fit_transform(x),y/scale)
    return dict(model=model,imputer=imputer,scale=scale,training_coefficients=y,parameters=parameters,
        family=family,train_ids=ids,seed=seed)


def predict(b,cell,future):
    if cell['physical_family']!=b['family']:raise ValueError('different physical family')
    # Validate requested cycle units even though coefficient head only uses prefix.
    inputs(cell,future)
    a=b['model'].predict(b['imputer'].transform(inputs(cell,[101.])))[0]*b['scale']
    if b['parameters']['shape']=='knee':p=trajectory(a,future)
    else:
        h=(np.asarray(future,dtype=float)-100)/100;p=a[0]-a[1]*h-a[2]*h*h
    if not np.isfinite(p).all():raise ValueError('nonfinite forecast')
    return p
