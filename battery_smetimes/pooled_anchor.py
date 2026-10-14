"""NASA signal-only pooled early anchor followed by future fade dynamics."""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from battery_smetimes.fade_rate import fit as rate_fit
from battery_smetimes.tail_boost import inputs


def early(cell):
    c=np.asarray(cell['cycle'],dtype=float);x=np.asarray(cell['raw'],dtype=float);take=(c>0)&(c<=100);c=c[take];x=x[take]
    if x.ndim!=2 or x.shape[1]!=44 or len(c)<3 or not np.isfinite(c).all() or np.any(np.diff(c)<=0) or np.isinf(x).any():raise ValueError('invalid NASA early signals')
    return c,x


def supervision(cell):
    c,x=early(cell);lookup={float(n):float(y) for n,y in zip(cell['label_cycle'],cell['soh']) if 0<n<=100 and np.isfinite(y) and y>0}
    take=np.array([float(n) in lookup for n in c]);cc=c[take];xx=x[take];y=np.array([lookup[float(n)] for n in cc])
    if len(y)<3:raise ValueError('insufficient training anchor labels')
    return cc,xx,y,float(np.median(y[-3:]))


def fit(cells,ids,kind,alpha,seed):
    ids=list(ids);families={cells[k]['physical_family'] for k in ids}
    if len(families)!=1:raise ValueError('different physical families')
    xs=[];ys=[];weights=[];anchors=[];membership=[]
    for key in ids:
        c,x,y,a=supervision(cells[key]);xs.append(x);ys.append(y);weights.append(np.full(len(y),1/len(y)));anchors.append(a);membership.append(dict(cell_id=key,cycles=c.tolist(),rows=len(y)))
    x,y,w=np.concatenate(xs),np.concatenate(ys),np.concatenate(weights);w*=len(w)/w.sum()
    imputer=SimpleImputer(strategy='median',keep_empty_features=True);z=imputer.fit_transform(x);scaler=None
    if kind=='ridge':
        offset=z[0].copy();scaler=StandardScaler().fit(z-offset,sample_weight=w);scaler.mean_+=offset;z=scaler.transform(z);model=Ridge(alpha=alpha,solver='svd')
    elif kind=='extra_trees':model=ExtraTreesRegressor(n_estimators=128,min_samples_leaf=5,max_depth=None,random_state=seed,n_jobs=2)
    else:raise ValueError('unknown anchor model')
    model.fit(z,y,sample_weight=w)
    # Reuse rate learning only; its temporary per-cell anchor tree is not queried.
    dynamic=rate_fit(cells,ids,True,seed,[dict(cell_id=k,anchor=a) for k,a in zip(ids,anchors)])
    return dict(kind=kind,alpha=alpha,seed=seed,train_ids=ids,family=next(iter(families)),model=model,imputer=imputer,scaler=scaler,dynamic=dynamic,anchors=anchors,supervision=membership)


def anchor(bundle,cell):
    if cell['physical_family']!=bundle['family']:raise ValueError('different physical family')
    _,x=early(cell);z=bundle['imputer'].transform(x[-3:])
    if bundle['scaler'] is not None:z=bundle['scaler'].transform(z)
    return float(np.median(bundle['model'].predict(z)))


def predict(bundle,cell,cycle):
    a=anchor(bundle,cell);cycle=np.asarray(cycle,dtype=float)
    if cycle.ndim!=1 or not len(cycle) or not np.isfinite(cycle).all() or np.any(cycle<=100):raise ValueError('invalid future coordinates')
    d=bundle['dynamic'];r=d['model'].predict(d['imputer'].transform(inputs(cell,cycle)));p=a-(1+(cycle-100)/100)*r
    if not np.isfinite(p).all():raise ValueError('nonfinite forecast')
    return p
