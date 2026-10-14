"""Separate signal-only cycle-101 level and future change, NCA only."""
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor,HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from battery_smetimes.tail_boost import inputs


def fit(cells,ids,tail_weight,seed):
    ids=list(ids);families={cells[k]['physical_family'] for k in ids}
    if len(families)!=1:raise ValueError('different family')
    ax=[];ay=[];xs=[];ys=[];ws=[]
    for k in ids:
        cell=cells[k];c=np.asarray(cell['cycle']);y=np.asarray(cell['truth'])
        if c[0]!=101 or c.shape!=y.shape or not np.isfinite(y).all():raise ValueError('cycle-101 training label required')
        ax.append(inputs(cell,np.array([101.]))[0]);ay.append(y[0])
        ii=np.unique(np.linspace(0,len(y)-1,min(96,len(y))).round().astype(int))
        xs.append(inputs(cell,c[ii]));ys.append(y[ii]-y[0])
        w=np.linspace(1,tail_weight,len(ii));ws.append(w/w.sum())
    ai=SimpleImputer(strategy='median',keep_empty_features=True);az=ai.fit_transform(np.asarray(ax))
    level=ExtraTreesRegressor(n_estimators=128,min_samples_leaf=3,max_depth=8,random_state=seed,n_jobs=2).fit(az,ay)
    di=SimpleImputer(strategy='median',keep_empty_features=True);dz=di.fit_transform(np.concatenate(xs));w=np.concatenate(ws);w*=len(w)/w.sum()
    delta=HistGradientBoostingRegressor(max_iter=120,learning_rate=.05,max_leaf_nodes=15,min_samples_leaf=20,
        l2_regularization=1,max_bins=64,early_stopping=False,random_state=seed).fit(dz,np.concatenate(ys),sample_weight=w)
    return dict(level=level,level_imputer=ai,delta=delta,delta_imputer=di,train_ids=ids,
        family=next(iter(families)),tail_weight=tail_weight,seed=seed)


def predict(b,cell,future):
    if cell['physical_family']!=b['family']:raise ValueError('different physical family')
    origin=inputs(cell,np.array([101.]));x=inputs(cell,future)
    level=b['level'].predict(b['level_imputer'].transform(origin))[0]
    d=b['delta'].predict(b['delta_imputer'].transform(x))
    d0=b['delta'].predict(b['delta_imputer'].transform(origin))[0]
    p=level+d-d0
    if not np.isfinite(p).all():raise ValueError('invalid prediction')
    return p
