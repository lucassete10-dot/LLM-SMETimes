"""Train-supervised degradation clock, inferred only from early signals at deployment."""
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from battery_smetimes import continuous_length as point
from battery_smetimes.condition_length import key


def group(cell):
    if 'metadata' in cell:return key(cell)
    return repr(np.asarray(cell['conditions'],float).tolist())


def rate_label(c,y):
    m=(c>=101)&(c<=150);c,y=c[m],y[m]
    if len(c)<3 or c[-1]-c[0]<20:raise ValueError('insufficient observed pace supervision')
    x=c-c.mean()
    return float(-100*x@(y-y.mean())/(x@x))


def fit(cells,ids,targets,origin,seed):
    ids=list(ids);raw={};eligible=[]
    for cid in ids:
        try:raw[cid]=rate_label(*targets[cid]);eligible.append(cid)
        except ValueError:pass
    if len(eligible)<3:raise ValueError('insufficient pace training cells')
    floor=.0005
    positive={cid:max(floor,raw[cid]) for cid in eligible}
    refs={g:float(np.median([positive[k] for k in eligible if group(cells[k])==g])) for g in {group(cells[k]) for k in eligible}}
    fallback=float(np.median(list(positive.values())))
    encoder=OneHotEncoder(sparse_output=False,handle_unknown='ignore')
    known=encoder.fit_transform([[group(cells[k])] for k in eligible])
    x=np.c_[np.stack([point.features(cells[k],origin,'VIT') for k in eligible]),known]
    imp=SimpleImputer(strategy='median',keep_empty_features=True);z=imp.fit_transform(x);sc=StandardScaler().fit(z)
    model=ExtraTreesRegressor(n_estimators=80,min_samples_leaf=2,max_features=.8,random_state=seed,n_jobs=2)
    model.fit(sc.transform(z),np.log([positive[k] for k in eligible]))
    transformed={};factors={}
    for cid in ids:
        ref=refs.get(group(cells[cid]),fallback)
        factor=float(np.clip(positive.get(cid,ref)/ref,.25,4))
        c,y=targets[cid];transformed[cid]=(origin+(c-origin)*factor,y);factors[cid]=factor
    shape=point.fit(cells,ids,origin,'point','VIT',1,seed,transformed)
    return dict(shape=shape,train_ids=ids,pace_train_ids=eligible,raw_rates=raw,reference_rates=refs,
        fallback_rate=fallback,floor=floor,factor_bounds=[.25,4],training_factors=factors,encoder=encoder,
        imputer=imp,scaler=sc,model=model,origin=origin,family=shape['family'])


def predict(b,cell,cycles):
    c=np.asarray(cycles,float)
    if cell['physical_family']!=b['family']:raise ValueError('physical family mismatch')
    if c.ndim!=1 or not len(c) or not np.isfinite(c).all() or np.any(c<=b['origin']):raise ValueError('invalid future positions')
    x=np.c_[point.features(cell,b['origin'],'VIT')[None],b['encoder'].transform([[group(cell)]])]
    rate=float(np.exp(b['model'].predict(b['scaler'].transform(b['imputer'].transform(x)))[0]))
    ref=b['reference_rates'].get(group(cell),b['fallback_rate'])
    factor=float(np.clip(rate/ref,*b['factor_bounds']))
    return point.predict(b['shape'],cell,b['origin']+(c-b['origin'])*factor)
