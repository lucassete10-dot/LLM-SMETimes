"""Sparse RPT pulse-voltage history; never infer measurements between RPTs."""
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def features(cell,origin):
    c=np.asarray(cell['source_cycle'],float); v=np.asarray(cell['pulse_voltage'],float)
    take=c<=origin; c,v=c[take],v[take]
    if not len(c) or not np.isfinite(c).all() or np.any(np.diff(c)<=0) or not np.isfinite(v).all():
        raise ValueError('invalid visible pulse history')
    v=v.reshape(len(c),6,101); t=np.arange(101,dtype=float); tc=t-t.mean()
    z=np.concatenate([v[:,:,0,None],v[:,:,100,None],v[:,:,[1,10,50,100]]-v[:,:,0,None],
        v.mean(axis=2)[:,:,None],v.std(axis=2)[:,:,None],
        (np.sum(v*tc,axis=2)/np.sum(tc*tc))[:,:,None]],axis=2).reshape(len(c),-1)
    delta=z[-1]-z[0]
    slope=delta/max(c[-1]-c[0],1)*100 if len(c)>1 else np.zeros(z.shape[1])
    g=int(cell['group_id']); assert 1<=g<=11
    return np.r_[z[0],z[-1],delta,slope,len(c),len(c)>1,c[-1]/100,np.eye(11)[g-1]]


def inputs(cell,origin,cycle,block='pulse'):
    c=np.asarray(cycle,float); h=(c-origin)/100
    f=features(cell,origin) if block=='pulse' else np.eye(11)[int(cell['group_id'])-1]
    return np.c_[np.broadcast_to(f,(len(c),len(f))),c/100,h,np.log1p(h)]


def fit(cells,ids,targets,origin,seed,block='pulse'):
    families={cells[k]['physical_family'] for k in ids}
    if len(families)!=1: raise ValueError('physical family mismatch')
    xs,ys,ws=[],[],[]
    for cid in ids:
        c,y=targets[cid]
        xs.append(inputs(cells[cid],origin,c,block)); ys.append(y); ws.append(np.full(len(c),1/len(c)))
    x,y,w=np.concatenate(xs),np.concatenate(ys),np.concatenate(ws); w*=len(w)/w.sum()
    imputer=SimpleImputer(strategy='median',keep_empty_features=True); z=imputer.fit_transform(x)
    scaler=StandardScaler().fit(z)
    model=ExtraTreesRegressor(n_estimators=80,min_samples_leaf=1,max_features=.8,random_state=seed,n_jobs=2).fit(scaler.transform(z),y,sample_weight=w)
    return dict(origin=origin,seed=seed,block=block,family=next(iter(families)),train_ids=list(ids),imputer=imputer,scaler=scaler,model=model)


def predict(bundle,cell,cycle):
    if cell['physical_family']!=bundle['family']: raise ValueError('physical family mismatch')
    c=np.asarray(cycle,float)
    if c.ndim!=1 or not len(c) or not np.isfinite(c).all() or np.any(c<=bundle['origin']): raise ValueError('invalid future cycle')
    return bundle['model'].predict(bundle['scaler'].transform(bundle['imputer'].transform(inputs(cell,bundle['origin'],c,bundle['block']))))
