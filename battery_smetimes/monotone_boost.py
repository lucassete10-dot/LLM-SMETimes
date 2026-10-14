"""Matched HGB models with/without source-cycle monotonic constraints."""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from battery_smetimes.conditional_drift import prefix
from battery_smetimes.crossfit_drift import basis
from battery_smetimes.tail_boost import labels,predict as tree_predict


def inputs(imputer,cell,cycle):
    h=basis(cycle,1)[:,0]
    x=imputer.transform(prefix(cell)[None])[0]
    return np.column_stack([np.tile(x,(len(h),1)),h,np.log1p(h),h*h])


def fit(cells,ids,l2,monotone,seed):
    if len({cells[k]['physical_family'] for k in ids})!=1:raise ValueError('different physical families')
    imputer=SimpleImputer(strategy='median',keep_empty_features=True)
    imputer.fit(np.stack([prefix(cells[k]) for k in ids]));xs=[];ys=[];ws=[]
    for key in ids:
        cycle,y=labels(cells[key]);idx=np.unique(np.linspace(0,len(y)-1,min(96,len(y))).round().astype(int))
        xs.append(inputs(imputer,cells[key],cycle[idx]));ys.append(y[idx]);w=np.linspace(1,3,len(idx));ws.append(w/w.sum())
    x,y,w=np.concatenate(xs),np.concatenate(ys),np.concatenate(ws);w*=len(w)/w.sum()
    constraints=np.r_[np.zeros(x.shape[1]-3,dtype=int),np.full(3,-1 if monotone else 0,dtype=int)]
    model=HistGradientBoostingRegressor(max_iter=120,learning_rate=.05,max_leaf_nodes=31,min_samples_leaf=20,
        l2_regularization=l2,max_bins=64,early_stopping=False,random_state=seed,monotonic_cst=constraints)
    model.fit(x,y,sample_weight=w)
    return dict(mode='boost',model=model,imputer=imputer,train_ids=list(ids),family=cells[ids[0]]['physical_family'],
        l2=l2,monotone=monotone,seed=seed)


def predict(b,cell,cycle):
    if b['mode']=='control':return tree_predict(b['base'],cell,cycle)
    if cell['physical_family']!=b['family']:raise ValueError('different physical family')
    p=b['model'].predict(inputs(b['imputer'],cell,cycle))
    if not np.isfinite(p).all():raise ValueError('nonfinite forecast')
    return p
