"""XJTU V/VI/VIT and actual-source-cycle segments, no partial-Ah channels."""
import warnings
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from prepare_xjtu import FEATURE_NAMES
from battery_smetimes.forecast import metadata_vector,type_key
from battery_smetimes.curve_fidelity import condition_key,targets

def columns(block):
    terms={'V':('voltage',),'VI':('voltage','current'),'VIT':('voltage','current','temperature')}[block]
    return [i for i,n in enumerate(FEATURE_NAMES) if any(t in n for t in terms)]

def prefix(cell,parameters):
    c0=np.asarray(cell['cycle']);take=c0<=100;c=c0[take].astype(float)
    x=np.asarray(cell['raw'])[take][:,columns(parameters['block'])].astype(float)
    if len(c)<6 or not np.isfinite(c).all() or np.any(np.diff(c)<=0) or np.isinf(x).any():raise ValueError('invalid early history')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        first=np.nanmedian(x[:5],axis=0);last=np.nanmedian(x[-3:],axis=0)
        xx=x[-10:];tt=c[-10:];valid=np.isfinite(xx);count=valid.sum(axis=0)
        mt=(valid*tt[:,None]).sum(axis=0)/np.maximum(count,1);mx=np.nansum(xx,axis=0)/np.maximum(count,1)
        tc=tt[:,None]-mt;den=(tc*tc*valid).sum(axis=0)
        slope=np.divide(np.nansum(tc*(xx-mx),axis=0),den,out=np.full(x.shape[1],np.nan),where=(count>=2)&(den>0))*100
        parts=[first,last,last-first,slope]
        if parameters['context']=='segments':
            med=[];counts=[]
            for lower,upper in ((0,25),(25,50),(50,75),(75,100)):
                mask=(c>lower)&(c<=upper);counts.append(mask.sum())
                med.append(np.nanmedian(x[mask],axis=0) if mask.any() else np.full(x.shape[1],np.nan))
            parts.extend(med);parts.extend([med[j+1]-med[j] for j in range(3)]);parts.append(np.array(counts)/100)
        elif parameters['context']!='basic':raise ValueError('unknown context')
    meta=cell['metadata'];parts.extend([np.array([c[0],c[-1],len(c),c[-1]-c[0]])/100,metadata_vector(meta,[type_key(meta)])[-5:]])
    return np.concatenate(parts)

def inputs(cell,future,parameters):
    f=np.asarray(future,dtype=float)
    if f.ndim!=1 or not len(f) or np.any(f<=100) or not np.isfinite(f).all():raise ValueError('invalid future')
    p=prefix(cell,parameters);h=f-100
    return np.column_stack([np.tile(p,(len(f),1)),f/100,h/100,np.log1p(h)])

def training_arrays(cells,ids,parameters):
    xs=[];ys=[];ws=[]
    for key in ids:
        c,y=targets(cells[key]);idx=np.unique(np.linspace(0,len(y)-1,min(96,len(y))).round().astype(int))
        xs.append(inputs(cells[key],c[idx],parameters));ys.append(y[idx]);w=np.linspace(1,3,len(idx));ws.append(w/w.sum())
    x,y,w=np.concatenate(xs),np.concatenate(ys),np.concatenate(ws);w*=len(w)/w.sum()
    return x,y,w

def fit(cells,ids,parameters,seed,local=True):
    ids=list(ids);families={cells[k]['physical_family'] for k in ids}
    if len(families)!=1:raise ValueError('different family')
    x,y,w=training_arrays(cells,ids,parameters);imputer=SimpleImputer(strategy='median',keep_empty_features=True);z=imputer.fit_transform(x)
    model=ExtraTreesRegressor(n_estimators=128,min_samples_leaf=1,max_depth=None,random_state=seed,n_jobs=2).fit(z,y,sample_weight=w)
    models={}
    if local:
        groups={}
        for key in ids:groups.setdefault(condition_key(cells[key]),[]).append(key)
        for k,members in groups.items():
            if len(members)>=6:models[k]=fit(cells,members,parameters,seed,local=False)
    return dict(model=model,imputer=imputer,local=models,parameters=parameters,seed=seed,train_ids=ids,family=next(iter(families)))

def predict(bundle,cell,future):
    if cell['physical_family']!=bundle['family']:raise ValueError('different family')
    key=condition_key(cell)
    if key in bundle['local']:return predict(bundle['local'][key],cell,future)
    return bundle['model'].predict(bundle['imputer'].transform(inputs(cell,future,bundle['parameters'])))
