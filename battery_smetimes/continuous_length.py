"""Causal V/I/T history with train-cell smooth acceleration supervision."""
import warnings
import numpy as np
from scipy.optimize import nnls
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from battery_smetimes.forecast import metadata_vector, type_key


def signal_history(cell):
    if 'early_cycle' in cell:
        return np.asarray(cell['early_cycle']), np.asarray(cell['early_raw']), np.array(cell['conditions'],float)
    dataset=cell['metadata']['dataset']
    if dataset=='MIT':
        from prepare_mit import ENHANCED_NAMES as names
    elif dataset=='XJTU':
        from prepare_xjtu import FEATURE_NAMES as names
    else:raise ValueError('unsupported cohort')
    idx=[i for i,n in enumerate(names) if any(s in n for s in ('voltage','current','temperature'))]
    m=cell['metadata']
    return np.asarray(cell['cycle']),np.asarray(cell['raw'])[:,idx],metadata_vector(m,[type_key(m)])[-5:]


def features(cell,origin,block='VIT'):
    cycle,raw,condition=signal_history(cell)
    take=cycle<=origin;c=cycle[take].astype(float);x=raw[take].astype(float)
    if len(c)<3:raise ValueError('insufficient visible signal observations')
    if not np.isfinite(c).all() or np.any(np.diff(c)<=0) or np.isinf(x).any():raise ValueError('invalid visible signals')
    if block=='condition':return condition
    if block=='V':
        if 'early_cycle' in cell:x=x[:,:16]
        else:
            dataset=cell['metadata']['dataset']
            if dataset=='MIT':from prepare_mit import ENHANCED_NAMES as names
            else:from prepare_xjtu import FEATURE_NAMES as names
            allowed=[n for n in names if any(s in n for s in ('voltage','current','temperature'))]
            x=x[:,[i for i,n in enumerate(allowed) if 'voltage' in n]]
    elif block!='VIT':raise ValueError('unknown block')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        initial=np.nanmedian(x[:min(5,len(x))],axis=0);last=np.nanmedian(x[-3:],axis=0)
        t=c[-10:];xx=x[-10:];valid=np.isfinite(xx);count=valid.sum(axis=0)
        tm=(valid*t[:,None]).sum(axis=0)/np.maximum(count,1);xm=np.nansum(xx,axis=0)/np.maximum(count,1)
        centered=t[:,None]-tm;den=(centered**2*valid).sum(axis=0)
        slope=np.divide(np.nansum(centered*(xx-xm),axis=0),den,out=np.full(x.shape[1],np.nan),where=(count>=2)&(den>0))*100
        segments=[];counts=[]
        for lo,hi in zip(np.linspace(0,origin,5)[:-1],np.linspace(0,origin,5)[1:]):
            mask=(c>lo)&(c<=hi);counts.append(mask.sum())
            segments.append(np.nanmedian(x[mask],axis=0) if mask.any() else np.full(x.shape[1],np.nan))
    return np.r_[initial,last,last-initial,slope,np.ravel(segments),np.ravel(np.diff(segments,axis=0)),np.array(counts)/origin,condition]


def basis(cycle,origin,tau,width=.2):
    h=(np.asarray(cycle,dtype=float)-origin)/100
    soft=width*np.logaddexp(0,(h-tau)/width)
    zero=width*np.logaddexp(0,-tau/width)
    return np.c_[np.ones(len(h)),-h,-(soft**2-zero**2)]


def shape_target(cycle,y,origin):
    c,y=np.asarray(cycle,float),np.asarray(y,float)
    ii=np.unique(np.linspace(0,len(c)-1,min(128,len(c))).round().astype(int));c,y=c[ii],y[ii]
    horizon=max((c[-1]-origin)/100,.01)
    best=None
    # Training-cell horizon selects a supervision search range only.
    for tau in np.unique(np.r_[0,np.linspace(0,horizon,41)]):
        A=basis(c,origin,tau);coef,_=nnls(A,y)
        residual=A@coef-y;score=float(np.mean(residual**2))
        if best is None or score<best[0]:best=(score,np.r_[coef,tau])
    return best[1],float(np.sqrt(best[0]))


def point_inputs(cell,origin,cycle,block):
    c=np.asarray(cycle,float);h=(c-origin)/100;p=features(cell,origin,block)
    return np.c_[np.broadcast_to(p,(len(c),len(p))),c/100,h,np.log1p(h)]


def fit(cells,ids,origin,kind,block,leaf,seed,targets):
    ids=list(ids);families={cells[k]['physical_family'] for k in ids}
    if len(families)!=1:raise ValueError('physical families differ')
    rows=[];truth=[];weights=[];fits=[]
    if kind=='continuous':
        for key in ids:
            c,y=targets[key];coef,error=shape_target(c,y,origin)
            rows.append(features(cells[key],origin,block));truth.append(coef)
            fits.append(dict(cell_id=key,parameters=coef.tolist(),fit_rmse_soh=error))
        x=np.stack(rows);y=np.stack(truth);target_scale=np.maximum(y.std(axis=0),1e-4);y=y/target_scale;w=None
    else:
        for key in ids:
            c,y0=targets[key];ii=np.unique(np.linspace(0,len(c)-1,min(96,len(c))).round().astype(int))
            rows.append(point_inputs(cells[key],origin,c[ii],block));truth.append(y0[ii]);weights.append(np.full(len(ii),1/len(ii)))
        x=np.concatenate(rows);y=np.concatenate(truth);w=np.concatenate(weights);w*=len(w)/w.sum();target_scale=None
    imputer=SimpleImputer(strategy='median',keep_empty_features=True);z=imputer.fit_transform(x)
    scaler=StandardScaler().fit(z);z=scaler.transform(z)
    model=ExtraTreesRegressor(n_estimators=80,min_samples_leaf=leaf,max_features=.8,random_state=seed,n_jobs=2).fit(z,y,sample_weight=w)
    return dict(kind=kind,block=block,leaf=leaf,origin=origin,seed=seed,train_ids=ids,family=next(iter(families)),
        imputer=imputer,scaler=scaler,model=model,target_scale=target_scale,training_shape_fits=fits)


def predict(b,cell,cycle):
    if cell['physical_family']!=b['family']:raise ValueError('physical family mismatch')
    c=np.asarray(cycle,float)
    if c.ndim!=1 or not len(c) or not np.isfinite(c).all() or np.any(c<=b['origin']):raise ValueError('invalid requested future')
    if b['kind']=='continuous':
        x=features(cell,b['origin'],b['block'])[None]
        params=b['model'].predict(b['scaler'].transform(b['imputer'].transform(x)))[0]*b['target_scale']
        return basis(c,b['origin'],params[3])@params[:3]
    x=point_inputs(cell,b['origin'],c,b['block'])
    return b['model'].predict(b['scaler'].transform(b['imputer'].transform(x)))
