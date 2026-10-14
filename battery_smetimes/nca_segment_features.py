"""NCA family-specific V/VI source-cycle context, without query capacity."""
import warnings
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesRegressor


def prefix(cell,parameters):
    if not cell['physical_family'].startswith('Zhu2022_dataset1_NCA_'):raise ValueError('NCA family required')
    c0=np.asarray(cell['early_cycle']);x0=np.asarray(cell['early_raw'])
    if x0.ndim!=2 or x0.shape[1]!=32:raise ValueError('NCA 32-channel schema required')
    take=c0<=100;c=c0[take].astype(float)
    if parameters['block'] not in ('V','VI'):raise ValueError('unknown signal block')
    x=x0[take,:16 if parameters['block']=='V' else 32].astype(float)
    if len(c)<6 or not np.isfinite(c).all() or np.any(np.diff(c)<=0) or np.isinf(x).any():raise ValueError('invalid early signals')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        first=np.nanmedian(x[:5],axis=0);last=np.nanmedian(x[-3:],axis=0)
        xx=x[-10:];tt=c[-10:];valid=np.isfinite(xx);count=valid.sum(axis=0)
        mt=(valid*tt[:,None]).sum(axis=0)/np.maximum(count,1);mx=np.nansum(xx,axis=0)/np.maximum(count,1)
        centered=tt[:,None]-mt;den=(centered*centered*valid).sum(axis=0)
        slope=np.divide(np.nansum(centered*(xx-mx),axis=0),den,out=np.full(x.shape[1],np.nan),where=(count>=2)&(den>0))*100
        parts=[first,last,last-first,slope]
        if parameters['context']=='segments':
            med=[];counts=[]
            for start in (0,25,50,75):
                mask=(c>start)&(c<=start+25);counts.append(mask.sum())
                med.append(np.nanmedian(x[mask],axis=0) if mask.any() else np.full(x.shape[1],np.nan))
            parts+=med+[med[j+1]-med[j] for j in range(3)]+[np.asarray(counts)/100]
        elif parameters['context']!='basic':raise ValueError('unknown context')
    return np.concatenate(parts+[np.asarray(cell['conditions']),np.array([c[0],c[-1],len(c),c[-1]-c[0]])/100])


def inputs(cell,future,parameters):
    c=np.asarray(future,dtype=float)
    if c.ndim!=1 or not len(c) or not np.isfinite(c).all() or np.any(c<=100):raise ValueError('invalid future source cycles')
    p=np.tile(prefix(cell,parameters),(len(c),1));h=(c-100)/100
    return np.c_[p,h,h*h,np.log1p(h),p*h[:,None]]


def fit(cells,ids,parameters,seed,local=True):
    ids=list(ids);families={cells[k]['physical_family'] for k in ids}
    if len(families)!=1:raise ValueError('different family')
    xs=[];ys=[];ws=[]
    for k in ids:
        c,y=cells[k]['cycle'],cells[k]['truth'];ii=np.unique(np.linspace(0,len(y)-1,min(96,len(y))).round().astype(int))
        xs.append(inputs(cells[k],c[ii],parameters));ys.append(y[ii]);w=np.linspace(1,3,len(ii));ws.append(w/w.sum())
    x=np.concatenate(xs);y=np.concatenate(ys);w=np.concatenate(ws);w*=len(w)/w.sum()
    imputer=SimpleImputer(strategy='median',keep_empty_features=True);z=imputer.fit_transform(x)
    model=ExtraTreesRegressor(n_estimators=128,min_samples_leaf=1,random_state=seed,n_jobs=2).fit(z,y,sample_weight=w)
    locals={}
    if local:
        groups={}
        for k in ids:groups.setdefault(tuple(cells[k]['conditions']),[]).append(k)
        for condition,members in groups.items():
            if len(members)>=6:locals[condition]=fit(cells,members,parameters,seed,False)
    return dict(model=model,imputer=imputer,local=locals,parameters=parameters,seed=seed,train_ids=ids,family=next(iter(families)))


def predict(bundle,cell,future):
    if cell['physical_family']!=bundle['family']:raise ValueError('different family')
    condition=tuple(cell['conditions'])
    if condition in bundle['local']:return predict(bundle['local'][condition],cell,future)
    return bundle['model'].predict(bundle['imputer'].transform(inputs(cell,future,bundle['parameters'])))
