"""Explicit V/VI/VIT calibration; no query labels or partial-Ah/Wh channels."""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,HuberRegressor

BLOCKS={'V':list(range(11))+[33,34,35],
        'VI':list(range(22))+[33,34,35,36,37],
        'VIT':list(range(42))}

def features(cell,parameters):
    cycle=np.asarray(cell['cycle']);take=cycle<=100;c=cycle[take]
    x=np.asarray(cell['raw'])[take][:,BLOCKS[parameters['block']]].astype(float)
    if len(c)<60 or not np.isfinite(c).all() or np.any(np.diff(c)<=0):raise ValueError('invalid early cycles')
    if np.isinf(x).any():raise ValueError('invalid early signals')
    if parameters['mode']=='delta':x=x-np.nanmedian(x[:5],axis=0)
    elif parameters['mode']!='absolute':raise ValueError('invalid feature mode')
    return c,np.column_stack([x,np.full(len(c),float(cell['metadata']['discharge_cutoff_v']))])

def supervision(cells,ids,parameters):
    xs=[];ys=[];ws=[];members=[]
    for key in ids:
        cell=cells[key];c,x=features(cell,parameters)
        lookup={int(n):float(y) for n,y in zip(cell['label_cycle'],cell['soh']) if n<=100 and np.isfinite(y) and y>0}
        keep=np.array([int(n) in lookup for n in c]);y=np.array([lookup[int(n)] for n in c[keep]])
        if not len(y):raise ValueError('no train supervision')
        xs.append(x[keep]);ys.append(y);ws.append(np.full(len(y),1/len(y)))
        members.extend(dict(cell_id=key,cycle=int(n)) for n in c[keep])
    x,y,w=np.concatenate(xs),np.concatenate(ys),np.concatenate(ws);w*=len(w)/w.sum()
    return x,y,w,members

def fit(cells,ids,parameters):
    families={cells[k]['physical_family'] for k in ids}
    if len(families)!=1:raise ValueError('different physical families')
    x,y,w,members=supervision(cells,ids,parameters)
    imputer=SimpleImputer(strategy='median',keep_empty_features=True).fit(x);z=imputer.transform(x);offset=z[0].copy()
    scaler=StandardScaler().fit(z-offset,sample_weight=w);scaler.mean_+=offset
    model=Ridge(alpha=parameters['alpha'],solver='svd').fit(scaler.transform(z),y,sample_weight=w)
    return dict(model=model,imputer=imputer,scaler=scaler,parameters=parameters,train_ids=list(ids),family=next(iter(families)),training_members=members)

def early_health(bundle,cell):
    if cell['physical_family']!=bundle['family']:raise ValueError('different physical family')
    c,x=features(cell,bundle['parameters'])
    return c,bundle['model'].predict(bundle['scaler'].transform(bundle['imputer'].transform(x)))

def predict(bundle,cell,future):
    future=np.asarray(future,dtype=float)
    if future.ndim!=1 or not len(future) or not np.isfinite(future).all() or np.any(future<=100):raise ValueError('invalid future')
    c,y=early_health(bundle,cell);keep=c>70
    trend=HuberRegressor(alpha=.01,max_iter=500).fit(((c[keep]-100)/100)[:,None],y[keep])
    return trend.predict(((future-100)/100)[:,None])
