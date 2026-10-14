"""First100 segment-conditioned local decline rates, anchored at cycle101."""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesRegressor
from battery_smetimes.xjtu_segment_features import inputs,predict as anchor_predict
from battery_smetimes.curve_fidelity import targets,condition_key

FEATURES=dict(block='VIT',context='segments')

def rate_labels(c,y,window):
    c=np.asarray(c,dtype=float);y=np.asarray(y,dtype=float)
    indices=np.unique(np.linspace(0,len(c)-1,min(96,len(c))).round().astype(int))
    centers=[];rates=[]
    for ix in indices:
        mask=abs(c-c[ix])<=window/2
        if mask.sum()<3:
            if len(c)<3:continue
            nearest=np.argsort(abs(c-c[ix]),kind='stable')[:3]
            mask=np.zeros(len(c),dtype=bool);mask[nearest]=True
        t=c[mask]-c[mask].mean();den=t@t
        if den<=0:continue
        centers.append(c[ix]);rates.append(max(0.,-float(t@(y[mask]-y[mask].mean())/den)*100))
    if not rates:raise ValueError('no valid TRAIN rate labels')
    return np.array(centers),np.array(rates)

def arrays(cells,ids,parameters):
    xs=[];ys=[];ws=[]
    for key in ids:
        c,r=rate_labels(*targets(cells[key]),parameters['window'])
        xs.append(inputs(cells[key],c,FEATURES));ys.append(r)
        w=np.linspace(1,3,len(c));ws.append(w/w.sum())
    x,y,w=np.concatenate(xs),np.concatenate(ys),np.concatenate(ws);w*=len(w)/w.sum()
    return x,y,w

def fit_rates(cells,ids,parameters,seed,local=True):
    ids=list(ids)
    if parameters not in [dict(window=w,leaf=l) for w in (10,25) for l in (1,2)]:raise ValueError('unregistered candidate')
    families={cells[k]['physical_family'] for k in ids}
    if len(families)!=1:raise ValueError('different family')
    x,y,w=arrays(cells,ids,parameters);imputer=SimpleImputer(strategy='median',keep_empty_features=True)
    model=ExtraTreesRegressor(n_estimators=128,min_samples_leaf=parameters['leaf'],max_depth=None,random_state=seed,n_jobs=2).fit(imputer.fit_transform(x),y,sample_weight=w)
    groups={}
    for k in ids:groups.setdefault(condition_key(cells[k]),[]).append(k)
    children={k:fit_rates(cells,v,parameters,seed,False) for k,v in groups.items() if local and len(v)>=6}
    return dict(model=model,imputer=imputer,local=children,train_ids=ids,family=next(iter(families)),parameters=parameters,seed=seed,training_rate_targets=y)

def rates(bundle,cell,c):
    if cell['physical_family']!=bundle['family']:raise ValueError('different family')
    key=condition_key(cell)
    if key in bundle['local']:return rates(bundle['local'][key],cell,c)
    return bundle['model'].predict(bundle['imputer'].transform(inputs(cell,c,FEATURES)))

def integrated_drop(future,rate_function):
    c=np.asarray(future,dtype=float)
    if c.ndim!=1 or not len(c) or np.any(c<101) or not np.isfinite(c).all():raise ValueError('invalid future')
    edges=np.arange(101,int(np.ceil(c.max()))+1,dtype=float)
    if len(edges)==1:return np.zeros_like(c)
    mid=(edges[:-1]+edges[1:])/2;r=np.asarray(rate_function(mid))
    if r.shape!=mid.shape or not np.isfinite(r).all():raise ValueError('invalid rates')
    drop=np.r_[0.,np.cumsum(r/100)]
    return np.interp(c,edges,drop)

def predict(bundle,cell,future):
    if cell['physical_family']!=bundle['family']:raise ValueError('different family')
    a=anchor_predict(bundle['anchor'],cell,[101.])[0]
    return a-integrated_drop(future,lambda c:rates(bundle['rate'],cell,c))
