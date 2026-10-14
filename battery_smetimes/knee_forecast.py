"""Training-only knee supervision and label-free early-signal inference."""
import warnings
import numpy as np
from scipy.optimize import nnls
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from battery_smetimes.research_baselines import signal_prefix
from battery_smetimes.curve_fidelity import targets


def early_features(cell, windows=False):
    prefix=signal_prefix(cell,100)
    if not windows:
        return prefix.astype(float)
    cycle=np.asarray(cell['cycle']); raw=np.asarray(cell['raw'],dtype=float)
    blocks=[]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        for lo in range(0,100,20):
            take=(cycle>lo)&(cycle<=lo+20)
            blocks.append(np.nanmedian(raw[take],axis=0) if take.any() else np.full(raw.shape[1],np.nan))
    blocks=np.stack(blocks)
    return np.r_[prefix,blocks.ravel(),np.diff(blocks,axis=0).ravel()]


def fit_knee(cycle,y):
    cycle,y=np.asarray(cycle,dtype=float),np.asarray(y,dtype=float)
    if len(y)<2 or np.any(cycle<=100) or not np.isfinite(y).all():
        raise ValueError('invalid training targets')
    idx=np.unique(np.linspace(0,len(y)-1,min(len(y),96)).round().astype(int))
    h=(cycle[idx]-100)/100; y=y[idx]
    w=np.sqrt(1+3*h/h.max()); penalty=np.eye(3)*np.sqrt(.001)
    best=None
    for tau in np.arange(0,30.001,.25):
        basis=np.column_stack([np.ones(len(h)),-h,-np.maximum(h-tau,0)**2])
        design=np.vstack([basis*w[:,None],penalty])
        target=np.r_[y*w,penalty@np.array([1.,0.,0.])]
        coef,residual=nnls(design,target)
        if best is None or residual<best[0]:
            best=(residual,np.r_[coef,tau])
    return best[1]


def trajectory(parameters,cycle):
    a,b,c,tau=parameters
    h=(np.asarray(cycle,dtype=float)-100)/100
    if h.ndim!=1 or not len(h) or not np.isfinite(h).all() or np.any(h<=0):
        raise ValueError('invalid requested future')
    return a-b*h-c*np.maximum(h-tau,0)**2


def fit(cells,ids,seed,parameters):
    family=cells[ids[0]]['physical_family']
    if any(cells[k]['physical_family']!=family for k in ids):
        raise ValueError('different physical families')
    x=np.stack([early_features(cells[k],parameters['windows']) for k in ids])
    y=np.stack([fit_knee(*targets(cells[k])) for k in ids])
    scale=np.maximum(y.std(axis=0),.001)
    imputer=SimpleImputer(strategy='median',keep_empty_features=True)
    model=ExtraTreesRegressor(n_estimators=128,min_samples_leaf=parameters['leaf'],
        random_state=seed,n_jobs=2).fit(imputer.fit_transform(x),y/scale)
    return dict(family=family,train_ids=list(ids),parameters=dict(parameters),
        training_knee_parameters=y,target_scale=scale,imputer=imputer,model=model)


def predict(bundle,cell,cycle):
    if cell['physical_family']!=bundle['family']:
        raise ValueError('different physical family')
    x=early_features(cell,bundle['parameters']['windows'])
    coef=bundle['model'].predict(bundle['imputer'].transform(x[None]))[0]*bundle['target_scale']
    p=trajectory(coef,cycle)
    if not np.isfinite(p).all():
        raise ValueError('nonfinite prediction')
    return p
