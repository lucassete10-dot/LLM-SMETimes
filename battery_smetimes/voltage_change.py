"""Explicit cross-cycle partial-voltage shape changes; not full-discharge Q(V)."""
import warnings
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from battery_smetimes import continuous_length as point


def features(cell,origin):
    base=point.features(cell,origin,'VIT')
    if 'early_cycle' in cell:
        c=np.asarray(cell['early_cycle']);raw=np.asarray(cell['early_raw'])
        if raw.shape[1]!=32:raise ValueError('unknown early signal schema')
        groups=[raw[c<=origin,:11]]
    else:
        c=np.asarray(cell['cycle']);raw=np.asarray(cell['raw']);dataset=cell['metadata']['dataset']
        if dataset=='MIT':
            from prepare_mit import ENHANCED_NAMES as names
            groups=[raw[c<=origin][:,[i for i,n in enumerate(names) if n.startswith('voltage_v_t')]]]
        elif dataset=='XJTU':
            from prepare_xjtu import FEATURE_NAMES as names
            groups=[raw[c<=origin][:,[i for i,n in enumerate(names) if n.startswith(stage+'_voltage_V_')]] for stage in ('charge','discharge')]
        else:raise ValueError('unknown voltage schema')
    extra=[]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        for x in groups:
            if x.ndim!=2 or x.shape[1]<3 or len(x)<3:raise ValueError('insufficient partial voltage grid')
            first=np.nanmedian(x[:min(5,len(x))],axis=0);last=np.nanmedian(x[-3:],axis=0);d=last-first
            q=np.nanpercentile(d,[10,50,90])
            extra.extend([np.nanmean(d),np.nanstd(d),np.nanvar(d),np.nanmin(d),np.nanmax(d),*q,
                np.sqrt(np.nanmean(d*d)),np.nanmean(np.abs(np.diff(d))),d[-1]-d[0],
                np.nanmean(np.nanstd(x,axis=0)),np.nanmean(np.abs(np.diff(x,axis=0)))])
    return np.r_[base,extra]


def inputs(cell,origin,cycle):
    c=np.asarray(cycle,float);h=(c-origin)/100;f=features(cell,origin)
    return np.c_[np.broadcast_to(f,(len(c),len(f))),c/100,h,np.log1p(h)]


def fit(cells,ids,targets,origin,seed):
    families={cells[k]['physical_family'] for k in ids}
    if len(families)!=1:raise ValueError('mixed physical family')
    xs=[];ys=[];ws=[]
    for k in ids:
        c,y=targets[k];ii=np.unique(np.linspace(0,len(c)-1,min(96,len(c))).round().astype(int))
        xs.append(inputs(cells[k],origin,c[ii]));ys.append(y[ii]);ws.append(np.full(len(ii),1/len(ii)))
    x=np.concatenate(xs);y=np.concatenate(ys);w=np.concatenate(ws);w*=len(w)/w.sum()
    imp=SimpleImputer(strategy='median',keep_empty_features=True);z=imp.fit_transform(x);sc=StandardScaler().fit(z)
    model=ExtraTreesRegressor(n_estimators=80,min_samples_leaf=1,max_features=.8,random_state=seed,n_jobs=2)
    model.fit(sc.transform(z),y,sample_weight=w)
    return dict(train_ids=list(ids),origin=origin,seed=seed,family=next(iter(families)),imputer=imp,scaler=sc,model=model)


def predict(b,cell,cycle):
    c=np.asarray(cycle,float)
    if cell['physical_family']!=b['family']:raise ValueError('physical family mismatch')
    if c.ndim!=1 or not len(c) or not np.isfinite(c).all() or np.any(c<=b['origin']):raise ValueError('invalid future')
    return b['model'].predict(b['scaler'].transform(b['imputer'].transform(inputs(cell,b['origin'],c))))
