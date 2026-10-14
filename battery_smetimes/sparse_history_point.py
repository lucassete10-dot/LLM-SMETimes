"""Observed-only prefix statistics allowing one or two actual reference tests."""
import warnings
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from battery_smetimes import continuous_length as original


def features(cell,origin,block='VIT'):
    cycle,raw,condition=original.signal_history(cell);take=cycle<=origin
    c=np.asarray(cycle[take],float);x=np.asarray(raw[take],float)
    if len(c)>=3:return original.features(cell,origin,block)
    if not len(c):raise ValueError('no visible signal observations')
    if not np.isfinite(c).all() or np.any(np.diff(c)<=0) or np.isinf(x).any():raise ValueError('invalid visible signals')
    if block!='VIT':raise ValueError('sparse probe only supports VIT')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        initial=np.nanmedian(x[:5],axis=0);last=np.nanmedian(x[-3:],axis=0)
        valid=np.isfinite(x);counts=valid.sum(axis=0)
        tm=(valid*c[:,None]).sum(axis=0)/np.maximum(counts,1);xm=np.nansum(x,axis=0)/np.maximum(counts,1)
        centered=c[:,None]-tm;den=(centered**2*valid).sum(axis=0)
        slope=np.divide(np.nansum(centered*(x-xm),axis=0),den,out=np.full(x.shape[1],np.nan),where=(counts>=2)&(den>0))*100
        segments=[];segment_counts=[]
        for lo,hi in zip(np.linspace(0,origin,5)[:-1],np.linspace(0,origin,5)[1:]):
            mask=(c>lo)&(c<=hi);segment_counts.append(mask.sum())
            segments.append(np.nanmedian(x[mask],axis=0) if mask.any() else np.full(x.shape[1],np.nan))
    # The schema is unchanged. A single observation has a missing trend, not an invented zero slope.
    return np.r_[initial,last,last-initial,slope,np.ravel(segments),np.ravel(np.diff(segments,axis=0)),np.array(segment_counts)/origin,condition]


def point_inputs(cell,origin,cycle,block='VIT'):
    c=np.asarray(cycle,float);h=(c-origin)/100;p=features(cell,origin,block)
    return np.c_[np.broadcast_to(p,(len(c),len(p))),c/100,h,np.log1p(h)]


def fit(cells,ids,origin,kind,block,leaf,seed,targets):
    if kind!='point' or block!='VIT':raise ValueError('only point/VIT supported')
    ids=list(ids);families={cells[k]['physical_family'] for k in ids}
    if len(families)!=1:raise ValueError('physical families differ')
    xs,ys,ws=[],[],[]
    for cid in ids:
        c,y=targets[cid];ii=np.unique(np.linspace(0,len(c)-1,min(96,len(c))).round().astype(int))
        xs.append(point_inputs(cells[cid],origin,c[ii]));ys.append(y[ii]);ws.append(np.full(len(ii),1/len(ii)))
    x,y,w=np.concatenate(xs),np.concatenate(ys),np.concatenate(ws);w*=len(w)/w.sum()
    imputer=SimpleImputer(strategy='median',keep_empty_features=True);z=imputer.fit_transform(x)
    scaler=StandardScaler().fit(z)
    model=ExtraTreesRegressor(n_estimators=80,min_samples_leaf=leaf,max_features=.8,random_state=seed,n_jobs=2).fit(scaler.transform(z),y,sample_weight=w)
    return dict(train_ids=ids,origin=origin,seed=seed,family=next(iter(families)),imputer=imputer,scaler=scaler,model=model,
        feature_backend='sparse_observed_only',minimum_observations=1,kind='point',block='VIT',leaf=leaf)


def predict(bundle,cell,cycle):
    c=np.asarray(cycle,float);origin=bundle['origin']
    if cell['physical_family']!=bundle['family']:raise ValueError('physical family mismatch')
    if c.ndim!=1 or not len(c) or not np.isfinite(c).all() or np.any(c<=origin):raise ValueError('invalid future')
    return bundle['model'].predict(bundle['scaler'].transform(bundle['imputer'].transform(point_inputs(cell,origin,c))))
