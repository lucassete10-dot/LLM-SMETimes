"""Cycle-wise VI/T calibration and label-free forecast of inferred health trend."""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,HuberRegressor

COLS=[5,10,34,35,38,40]


def features(cell,mode):
    take=np.asarray(cell['cycle'])<=100;c=np.asarray(cell['cycle'])[take];x=np.asarray(cell['raw'])[take][:,COLS].astype(float)
    if len(c)<60:raise ValueError('insufficient early cycles')
    if mode=='delta':x=x-np.nanmedian(x[:5],axis=0)
    elif mode!='absolute':raise ValueError('unknown mode')
    cutoff=float(cell['metadata']['discharge_cutoff_v'])
    return c,np.column_stack([x,np.full(len(c),cutoff)])


def supervision(cells,ids,mode):
    xs=[];ys=[];ws=[];members=[]
    for key in ids:
        cell=cells[key];c,x=features(cell,mode);lookup={int(n):float(y) for n,y in zip(cell['label_cycle'],cell['soh']) if n<=100 and np.isfinite(y) and y>0};take=np.array([int(n) in lookup for n in c]);y=np.array([lookup[int(n)] for n in c[take]])
        if not len(y):raise ValueError('no early train labels')
        xs.append(x[take]);ys.append(y);ws.append(np.full(len(y),1/len(y)));members.extend(dict(cell_id=key,cycle=int(n)) for n in c[take])
    x,y,w=np.concatenate(xs),np.concatenate(ys),np.concatenate(ws);w*=len(w)/w.sum()
    return x,y,w,members


def fit(cells,ids,parameters):
    families={cells[k]['physical_family'] for k in ids}
    if len(families)!=1:raise ValueError('different physical families')
    x,y,w,members=supervision(cells,ids,parameters['mode']);imputer=SimpleImputer(strategy='median',keep_empty_features=True).fit(x);z=imputer.transform(x);offset=z[0].copy();scaler=StandardScaler().fit(z-offset,sample_weight=w);scaler.mean_+=offset
    model=Ridge(alpha=parameters['alpha'],solver='svd').fit(scaler.transform(z),y,sample_weight=w)
    return dict(model=model,imputer=imputer,scaler=scaler,parameters=parameters,train_ids=list(ids),family=next(iter(families)),training_members=members)


def early_health(bundle,cell):
    if cell['physical_family']!=bundle['family']:raise ValueError('different physical family')
    c,x=features(cell,bundle['parameters']['mode']);y=bundle['model'].predict(bundle['scaler'].transform(bundle['imputer'].transform(x)))
    return c,y


def predict(bundle,cell,future):
    c,y=early_health(bundle,cell);take=c>100-bundle['parameters']['window'];t=(c[take]-100)/100
    model=HuberRegressor(alpha=.01,max_iter=500).fit(t[:,None],y[take]);future=np.asarray(future,dtype=float)
    if future.ndim!=1 or not np.isfinite(future).all() or np.any(future<=100):raise ValueError('invalid future')
    return model.predict(((future-100)/100)[:,None])
