"""TRAIN curve coefficients supervised by first100 VIT segments only."""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesRegressor
from battery_smetimes.xjtu_segment_features import prefix
from battery_smetimes.curve_fidelity import shape_parameters, targets, condition_key

FEATURES=dict(block='VIT',context='segments')

def evaluate_coefficients(coefficients,future):
    c=np.asarray(future,dtype=float)
    if c.ndim!=1 or not len(c) or not np.isfinite(c).all() or np.any(c<=100):
        raise ValueError('invalid future')
    a=np.asarray(coefficients,dtype=float)
    if a.shape not in ((2,),(3,)) or not np.isfinite(a).all():
        raise ValueError('invalid coefficients')
    h=(c-100)/100
    return a[0]-a[1]*h-(a[2]*h*h if len(a)==3 else 0)

def fit(cells,ids,parameters,seed,local=True):
    ids=list(ids)
    if parameters not in [dict(degree=d,leaf=l) for d in (1,2) for l in (1,2)]:
        raise ValueError('unregistered candidate')
    families={cells[k]['physical_family'] for k in ids}
    if len(families)!=1:raise ValueError('different physical family')
    x=np.stack([prefix(cells[k],FEATURES) for k in ids])
    y=np.stack([shape_parameters(*targets(cells[k]),parameters['degree']) for k in ids])
    scale=np.maximum(np.std(y,axis=0),.001)
    imputer=SimpleImputer(strategy='median',keep_empty_features=True)
    z=imputer.fit_transform(x)
    model=ExtraTreesRegressor(n_estimators=128,min_samples_leaf=parameters['leaf'],
        max_depth=None,random_state=seed,n_jobs=2).fit(z,y/scale)
    groups={}
    for k in ids:groups.setdefault(condition_key(cells[k]),[]).append(k)
    locals={k:fit(cells,v,parameters,seed,False) for k,v in groups.items() if local and len(v)>=6}
    return dict(model=model,imputer=imputer,target_scale=scale,training_shape_parameters=y,
        parameters=parameters,seed=seed,train_ids=ids,family=next(iter(families)),local=locals)

def predict(bundle,cell,future):
    if cell['physical_family']!=bundle['family']:raise ValueError('different physical family')
    key=condition_key(cell)
    if key in bundle['local']:return predict(bundle['local'][key],cell,future)
    x=prefix(cell,FEATURES)[None]
    a=bundle['model'].predict(bundle['imputer'].transform(x))[0]*bundle['target_scale']
    return evaluate_coefficients(a,future)
