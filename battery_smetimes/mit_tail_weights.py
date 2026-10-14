"""Fixed alternative within-cell loss weights; unchanged MIT V input contract."""
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from battery_smetimes.mit_segment_features import inputs,condition_key,predict
from battery_smetimes.curve_fidelity import targets


def point_weights(cycle,scheme):
    c=np.asarray(cycle,dtype=float)
    if c.ndim!=1 or len(c)<2 or not np.isfinite(c).all() or np.any(np.diff(c)<=0):
        raise ValueError('invalid training cycles')
    if scheme in ('ramp3','ramp10','ramp30'):
        w=np.linspace(1,float(scheme[4:]),len(c))
    elif scheme=='terminal10':
        w=np.where(c>=c[0]+.75*(c[-1]-c[0]),10.,1.)
    else:raise ValueError('unknown fixed weight scheme')
    return w/w.sum()


def training_arrays(cells,ids,parameters):
    xs=[];ys=[];ws=[]
    for key in ids:
        c,y=targets(cells[key]);idx=np.unique(np.linspace(0,len(y)-1,min(96,len(y))).round().astype(int))
        xs.append(inputs(cells[key],c[idx],parameters));ys.append(y[idx])
        ws.append(point_weights(c[idx],parameters['scheme']))
    x,y,w=np.concatenate(xs),np.concatenate(ys),np.concatenate(ws)
    w*=len(w)/w.sum()
    return x,y,w


def fit(cells,ids,parameters,seed,local=True):
    ids=list(ids);families={cells[k]['physical_family'] for k in ids}
    if len(families)!=1:raise ValueError('different family')
    x,y,w=training_arrays(cells,ids,parameters)
    imputer=SimpleImputer(strategy='median',keep_empty_features=True)
    z=imputer.fit_transform(x)
    model=ExtraTreesRegressor(n_estimators=128,min_samples_leaf=1,max_depth=None,
        random_state=seed,n_jobs=2).fit(z,y,sample_weight=w)
    models={}
    if local:
        groups={}
        for k in ids:groups.setdefault(condition_key(cells[k]),[]).append(k)
        for key,members in groups.items():
            if len(members)>=6:models[key]=fit(cells,members,parameters,seed,False)
    return dict(model=model,imputer=imputer,local=models,parameters=parameters,
        seed=seed,train_ids=ids,family=next(iter(families)))
