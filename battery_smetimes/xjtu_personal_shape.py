"""Personalized regularized saturating-curvature shapes from first100 VIT."""
import numpy as np
from scipy.optimize import lsq_linear
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from battery_smetimes.xjtu_segment_features import prefix
from battery_smetimes.curve_fidelity import targets

FEATURES=dict(block='VIT',context='segments')


def basis(cycle):
    c=np.asarray(cycle,dtype=float)
    if c.ndim!=1 or not len(c) or not np.isfinite(c).all() or np.any(c<=100):raise ValueError('invalid future')
    h=(c-100)/100
    return np.column_stack([np.ones(len(c)),-h,-h*h/(1+h)])


def target(c,y):
    c=np.asarray(c,dtype=float);y=np.asarray(y,dtype=float)
    ii=np.unique(np.linspace(0,len(y)-1,min(96,len(y))).round().astype(int))
    w=np.linspace(1,3,len(ii));w=np.sqrt(.5*w/w.sum())
    x=np.vstack([basis(c[ii])*w[:,None],basis(c[-1:])*np.sqrt(.5),np.eye(3)*np.sqrt(.001)])
    z=np.r_[y[ii]*w,y[-1]*np.sqrt(.5),np.sqrt(.001),0,0]
    result=lsq_linear(x,z,bounds=([.2,0,0],[1.3,5,5]),tol=1e-12,max_iter=500)
    if not result.success:raise ValueError('shape target fit failed')
    return result.x


def fit(cells,ids,shape_targets,alpha):
    ids=list(ids);families={cells[k]['physical_family'] for k in ids}
    if len(families)!=1:raise ValueError('different family')
    x=np.stack([prefix(cells[k],FEATURES) for k in ids]);y=np.stack([shape_targets[k] for k in ids])
    imp=SimpleImputer(strategy='median',keep_empty_features=True);z=imp.fit_transform(x)
    scale=StandardScaler();z=scale.fit_transform(z)
    model=None if alpha is None else Ridge(alpha=alpha).fit(z,y)
    return dict(imputer=imp,scaler=scale,model=model,mean=y.mean(axis=0),lower=y.min(axis=0),upper=y.max(axis=0),
        alpha=alpha,train_ids=ids,family=next(iter(families)))


def coefficients(bundle,cell):
    if cell['physical_family']!=bundle['family']:raise ValueError('different family')
    if bundle['model'] is None:return bundle['mean']
    z=bundle['scaler'].transform(bundle['imputer'].transform(prefix(cell,FEATURES)[None]))
    return np.clip(bundle['model'].predict(z)[0],bundle['lower'],bundle['upper'])


def predict(bundle,cell,cycle):return basis(cycle)@coefficients(bundle,cell)
