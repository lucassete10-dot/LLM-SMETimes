"""Only replace OLS last10 slope with pairwise median in existing VIT prefix."""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesRegressor
from battery_smetimes.xjtu_segment_features import prefix as original_prefix,columns
from battery_smetimes.curve_fidelity import condition_key,targets


def prefix(cell,parameters):
    p=original_prefix(cell,parameters).copy();ix=columns(parameters['block'])
    keep=cell['cycle']<=100;t=cell['cycle'][keep][-10:].astype(float);x=cell['raw'][keep][-10:][:,ix].astype(float);slopes=[]
    for j in range(len(ix)):
        good=np.isfinite(x[:,j]);c=t[good];v=x[good,j]
        pairs=[100*(v[b]-v[a])/(c[b]-c[a]) for a in range(len(c)) for b in range(a+1,len(c))]
        slopes.append(float(np.median(pairs)) if pairs else np.nan)
    p[3*len(ix):4*len(ix)]=slopes
    return p


def inputs(cell,future,parameters):
    c=np.asarray(future,dtype=float)
    if c.ndim!=1 or not len(c) or not np.isfinite(c).all() or np.any(c<=100):raise ValueError('invalid future')
    p=prefix(cell,parameters);h=c-100
    return np.column_stack([np.tile(p,(len(c),1)),c/100,h/100,np.log1p(h)])


def fit(cells,ids,parameters,seed,local=True):
    ids=list(ids);families={cells[k]['physical_family'] for k in ids}
    if len(families)!=1:raise ValueError('different family')
    xs=[];ys=[];weights=[]
    for k in ids:
        c,y=targets(cells[k]);ii=np.unique(np.linspace(0,len(c)-1,min(96,len(c))).round().astype(int))
        xs.append(inputs(cells[k],c[ii],parameters));ys.append(y[ii]);w=np.linspace(1,3,len(ii));weights.append(w/w.sum())
    x,y,w=np.concatenate(xs),np.concatenate(ys),np.concatenate(weights);w*=len(w)/w.sum()
    imputer=SimpleImputer(strategy='median',keep_empty_features=True);z=imputer.fit_transform(x)
    model=ExtraTreesRegressor(n_estimators=128,min_samples_leaf=1,max_depth=None,random_state=seed,n_jobs=2).fit(z,y,sample_weight=w)
    groups={}
    for k in ids:groups.setdefault(condition_key(cells[k]),[]).append(k)
    locals={k:fit(cells,v,parameters,seed,False) for k,v in groups.items() if local and len(v)>=6}
    return dict(imputer=imputer,model=model,local=locals,train_ids=ids,parameters=parameters,seed=seed,family=next(iter(families)),trend='theil_sen')


def predict(bundle,cell,future):
    if cell['physical_family']!=bundle['family']:raise ValueError('different family')
    key=condition_key(cell)
    if key in bundle['local']:return predict(bundle['local'][key],cell,future)
    return bundle['model'].predict(bundle['imputer'].transform(inputs(cell,future,bundle['parameters'])))
