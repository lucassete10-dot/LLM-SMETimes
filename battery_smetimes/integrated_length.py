"""Causal signal-conditioned nonnegative rate integration; no terminal clipping."""
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from battery_smetimes.continuous_length import point_inputs, fit as fit_anchor, predict as anchor_predict
from battery_smetimes.xjtu_integrated_rate_v2 import rate_labels


def training_arrays(cells,ids,targets,origin,block,window):
    xs=[];ys=[];ws=[]
    for key in ids:
        c,r=rate_labels(*targets[key],window)
        xs.append(point_inputs(cells[key],origin,c,block));ys.append(r)
        ws.append(np.full(len(r),1/len(r)))
    x,y,w=np.concatenate(xs),np.concatenate(ys),np.concatenate(ws)
    return x,y,w*len(w)/w.sum()


def fit(cells,ids,targets,origin,block,window,leaf,seed,anchor):
    ids=list(ids);families={cells[k]['physical_family'] for k in ids}
    if len(families)!=1:raise ValueError('incompatible physical families')
    x,y,w=training_arrays(cells,ids,targets,origin,block,window)
    imp=SimpleImputer(strategy='median',keep_empty_features=True);z=imp.fit_transform(x)
    sc=StandardScaler().fit(z)
    model=ExtraTreesRegressor(n_estimators=80,min_samples_leaf=leaf,max_features=.8,random_state=seed,n_jobs=2)
    model.fit(sc.transform(z),y,sample_weight=w)
    start=min(float(targets[k][0][0]) for k in ids)
    if start<=origin:raise ValueError('anchor is not future')
    return dict(kind='integrated_rate',block=block,window=window,leaf=leaf,seed=seed,origin=origin,
                train_ids=ids,family=next(iter(families)),imputer=imp,scaler=sc,model=model,anchor=anchor,
                anchor_cycle=start,training_rate_targets=y,training_rate_count=len(y))


def predict(b,cell,cycle):
    if cell['physical_family']!=b['family']:raise ValueError('physical family mismatch')
    c=np.asarray(cycle,float);start=b['anchor_cycle']
    if c.ndim!=1 or not len(c) or not np.isfinite(c).all() or np.any(c<start):raise ValueError('invalid future grid')
    anchor=anchor_predict(b['anchor'],cell,np.array([start]))[0]
    edges=np.r_[start,np.arange(np.floor(start)+1,np.ceil(c.max())+1)]
    if len(edges)==1:return np.full_like(c,anchor)
    mid=(edges[:-1]+edges[1:])/2
    x=point_inputs(cell,b['origin'],mid,b['block'])
    rates=b['model'].predict(b['scaler'].transform(b['imputer'].transform(x)))
    if not np.isfinite(rates).all() or np.any(rates<0):raise ValueError('invalid rate predictions')
    drop=np.r_[0,np.cumsum(rates*np.diff(edges)/100)]
    return anchor-np.interp(c,edges,drop)
