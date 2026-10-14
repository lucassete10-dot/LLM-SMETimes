"""Nonnegative fixed-knot rate increments with a finite asymptotic slope."""
import numpy as np
from scipy.optimize import nnls
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from battery_smetimes import continuous_length as point
from battery_smetimes import condition_length as protocol

KNOTS = np.array([0., .5, 1., 2., 4., 8., 12.])
WIDTH = .25


def basis(cycle, origin):
    h=(np.asarray(cycle,float)-origin)/100
    return np.c_[h, WIDTH*np.logaddexp(0,(h[:,None]-KNOTS)/WIDTH)]


def supervision(c,y,origin,strength):
    c,y=np.asarray(c,float),np.asarray(y,float)
    ii=np.unique(np.linspace(0,len(c)-1,min(128,len(c))).round().astype(int))
    a=basis(c[ii],origin)-basis([c[0]],origin)
    delta=y[0]-y[ii]
    # Fixed regularization penalizes unsupported rate increments; no horizon-dependent cap.
    penalty=np.diag(np.r_[0.,np.ones(len(KNOTS))])
    coef,_=nnls(np.r_[a/np.sqrt(len(ii)),np.sqrt(strength)*penalty],np.r_[delta/np.sqrt(len(ii)),np.zeros(a.shape[1])])
    error=float(np.sqrt(np.mean((a@coef-delta)**2)))
    return coef,error


def fit(cells,ids,targets,origin,seed,strength,local=False):
    ids=list(ids); anchor=point.fit(cells,ids,origin,'point','VIT',1,seed,targets)
    x=np.stack([point.features(cells[k],origin) for k in ids]);ys=[];fits=[]
    for k in ids:
        coef,error=supervision(*targets[k],origin,strength)
        ys.append(coef);fits.append(dict(cell_id=k,coefficients=coef.tolist(),fit_rmse_soh=error))
    y=np.stack(ys);scale=np.maximum(y.std(0),1e-4)
    imp=SimpleImputer(strategy='median',keep_empty_features=True);z=imp.fit_transform(x)
    sc=StandardScaler().fit(z)
    model=ExtraTreesRegressor(n_estimators=80,min_samples_leaf=1,max_features=.8,random_state=seed,n_jobs=2).fit(sc.transform(z),y/scale)
    groups={}
    if local:
        for k in ids:groups.setdefault(protocol.key(cells[k]),[]).append(k)
    children={g:fit(cells,ks,targets,origin,seed,strength,False) for g,ks in groups.items() if len(ks)>=3}
    return dict(kind='smooth_rate_curve',origin=origin,seed=seed,strength=strength,train_ids=ids,
        family=anchor['family'],anchor=anchor,anchor_cycle=min(targets[k][0][0] for k in ids),
        imputer=imp,scaler=sc,model=model,target_scale=scale,training_shape_fits=fits,local=children,
        training_group_counts={g:len(ks) for g,ks in groups.items()})


def predict(b,cell,cycle):
    if cell['physical_family']!=b['family']:raise ValueError('incompatible physical family')
    if b['local']:
        child=b['local'].get(protocol.key(cell))
        if child is not None:return predict(child,cell,cycle)
    c=np.asarray(cycle,float);start=b['anchor_cycle']
    if c.ndim!=1 or not len(c) or not np.isfinite(c).all() or np.any(c<start):raise ValueError('invalid future grid')
    z=b['scaler'].transform(b['imputer'].transform(point.features(cell,b['origin'])[None]))
    coef=b['model'].predict(z)[0]*b['target_scale']
    if np.any(coef<0) or not np.isfinite(coef).all():raise ValueError('invalid predicted coefficients')
    anchor=point.predict(b['anchor'],cell,np.array([start]))[0]
    return anchor-(basis(c,b['origin'])-basis([start],b['origin']))@coef
