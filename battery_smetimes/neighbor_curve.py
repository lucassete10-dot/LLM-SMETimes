"""Whole-cell reference curves selected using early signals and known protocol."""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from battery_smetimes.conditional_drift import prefix
from battery_smetimes.tail_boost import labels,conditions

def extrapolate(c,y,future):
    c,y,future=[np.asarray(v,dtype=float) for v in (c,y,future)]
    if len(c)<2 or not np.all(np.diff(c)>0):raise ValueError('invalid reference')
    p=np.interp(future,c,y);n=min(20,len(c))
    for mask,cc,yy,anchor_c,anchor_y in [(future<c[0],c[:n],y[:n],c[0],y[0]),(future>c[-1],c[-n:],y[-n:],c[-1],y[-1])]:
        centered=cc-cc.mean();slope=float(centered@(yy-yy.mean())/(centered@centered));p[mask]=anchor_y+slope*(future[mask]-anchor_c)
    return p

def fit(cells,ids):
    ids=list(ids);families={cells[k]['physical_family'] for k in ids}
    if len(families)!=1:raise ValueError('different physical family')
    raw=np.stack([prefix(cells[k]) for k in ids]);imputer=SimpleImputer(strategy='median',keep_empty_features=True);x=imputer.fit_transform(raw);scaler=StandardScaler();z=scaler.fit_transform(x)
    refs=[]
    for k in ids:
        c,y=labels(cells[k]);refs.append(dict(cell_id=k,condition=conditions(cells[k]),cycle=c,truth=y))
    return dict(train_ids=ids,family=next(iter(families)),imputer=imputer,scaler=scaler,z=z,references=refs)

def predict(bundle,cell,future,k,exclude=None):
    if cell['physical_family']!=bundle['family']:raise ValueError('different physical family')
    future=np.asarray(future,dtype=float)
    if future.ndim!=1 or not len(future) or not np.isfinite(future).all() or np.any(future<=100):raise ValueError('invalid future')
    eligible=[i for i,r in enumerate(bundle['references']) if r['cell_id']!=exclude]
    local=[i for i in eligible if bundle['references'][i]['condition']==conditions(cell)]
    pool=local if len(local)>=3 else eligible
    if not pool:raise ValueError('no reference peers')
    q=bundle['scaler'].transform(bundle['imputer'].transform(prefix(cell)[None]))[0]
    order=sorted(pool,key=lambda i:(float(np.linalg.norm(q-bundle['z'][i])),bundle['train_ids'][i]))[:k]
    values=[extrapolate(bundle['references'][i]['cycle'],bundle['references'][i]['truth'],future) for i in order]
    p=np.mean(values,axis=0)
    if not np.isfinite(p).all():raise ValueError('nonfinite reference forecast')
    return p
