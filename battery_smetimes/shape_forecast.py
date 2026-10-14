"""Matched signal-only forecasts with explicit waveform-shape projections."""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from battery_smetimes.waveform_shape import matrix
from battery_smetimes.tail_boost import labels,predict as tree_predict


def training_arrays(cells,ids):
    xs=[];ys=[];ws=[]
    for key in ids:
        x,y=labels(cells[key]);idx=np.unique(np.linspace(0,len(y)-1,min(96,len(y))).round().astype(int))
        xs.append(matrix(cells[key],x[idx],True));ys.append(y[idx]);w=np.linspace(1,3,len(idx));ws.append(w/w.sum())
    x,y,w=np.concatenate(xs),np.concatenate(ys),np.concatenate(ws);w*=len(w)/w.sum()
    return x,y,w


def fit(cells,ids,kind,alpha,seed):
    ids=list(ids);families={cells[k]['physical_family'] for k in ids}
    if len(families)!=1:raise ValueError('different physical families')
    x,y,w=training_arrays(cells,ids);imputer=SimpleImputer(strategy='median',keep_empty_features=True);z=imputer.fit_transform(x.astype(float));scaler=None
    if kind=='extra_trees':model=ExtraTreesRegressor(n_estimators=128,min_samples_leaf=1,max_depth=None,random_state=seed,n_jobs=2)
    elif kind=='ridge':
        scaler=StandardScaler();offset=z[0].copy();scaler.fit(z-offset,sample_weight=w);scaler.mean_+=offset
        if not np.isfinite(scaler.var_).all() or np.any(scaler.var_<0):raise ValueError('invalid training variance')
        z=scaler.transform(z);model=Ridge(alpha=alpha,solver='svd')
    else:raise ValueError('unknown model')
    model.fit(z,y,sample_weight=w)
    return dict(mode='shape',model=model,imputer=imputer,scaler=scaler,train_ids=ids,family=next(iter(families)),kind=kind,alpha=alpha,seed=seed)


def predict(b,cell,cycle):
    if b['mode']=='control':return tree_predict(b['base'],cell,cycle)
    if cell['physical_family']!=b['family']:raise ValueError('different physical family')
    x=b['imputer'].transform(matrix(cell,cycle,True))
    if b['scaler'] is not None:x=b['scaler'].transform(x)
    p=b['model'].predict(x)
    if not np.isfinite(p).all():raise ValueError('nonfinite forecast')
    return p
