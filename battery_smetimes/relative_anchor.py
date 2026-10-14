"""Within-cell early relative signal calibration, with frozen fade dynamics."""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from battery_smetimes.pooled_anchor import early
from battery_smetimes.fade_rate import predict as fade


def signals(cell,compact):
    cycle,raw=early(cell);reference=np.nanmedian(raw[:5],axis=0)
    cols=np.array([5,10,34,35,38,40]) if compact else np.arange(44)
    return cycle,(raw-reference)[:,cols]


def training(cells,ids,compact):
    xs=[];ys=[];ws=[];membership=[]
    for key in ids:
        cell=cells[key];cycle,x=signals(cell,compact)
        lookup={float(c):float(y) for c,y in zip(cell['label_cycle'],cell['soh']) if 0<c<=100 and np.isfinite(y) and y>0}
        take=np.array([float(c) in lookup for c in cycle]);c=cycle[take];y=np.array([lookup[float(n)] for n in c]);assert len(y)>=3
        xs.append(x[take]);ys.append(y);ws.append(np.full(len(y),1/len(y)));membership.append(dict(cell_id=key,cycles=c.tolist(),rows=len(y)))
    x,y,w=np.concatenate(xs),np.concatenate(ys),np.concatenate(ws);w*=len(w)/w.sum()
    return x,y,w,membership


def fit(cells,ids,compact,alpha,seed,rate):
    ids=list(ids);families={cells[k]['physical_family'] for k in ids}
    if len(families)!=1 or rate['train_ids']!=ids or rate['family'] not in families or rate['learned_anchor']:raise ValueError('component family or membership mismatch')
    x,y,w,membership=training(cells,ids,compact);imputer=SimpleImputer(strategy='median',keep_empty_features=True);z=imputer.fit_transform(x)
    offset=z[0].copy();scaler=StandardScaler().fit(z-offset,sample_weight=w);scaler.mean_+=offset;model=Ridge(alpha=alpha,solver='svd').fit(scaler.transform(z),y,sample_weight=w)
    return dict(model=model,imputer=imputer,scaler=scaler,compact=compact,alpha=alpha,seed=seed,train_ids=ids,family=next(iter(families)),rate=rate,supervision=membership)


def anchor(b,cell):
    if cell['physical_family']!=b['family']:raise ValueError('different physical family')
    _,x=signals(cell,b['compact']);z=b['scaler'].transform(b['imputer'].transform(x[-3:]))
    return float(np.median(b['model'].predict(z)))


def predict(b,cell,cycle):
    p=anchor(b,cell)+fade(b['rate'],cell,cycle)-fade(b['rate'],cell,np.array([101.]))[0]
    if not np.isfinite(p).all():raise ValueError('nonfinite forecast')
    return p
