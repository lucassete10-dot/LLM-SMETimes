"""Train-only smooth-knee parameters predicted from low-rank early signals."""
import numpy as np
from scipy.special import expit
from sklearn.linear_model import Ridge
from battery_smetimes.conditional_drift import encode_fit,encode
from battery_smetimes.warm_monotone import derive_prior
from battery_smetimes.tail_boost import predict as tree_predict
from battery_smetimes.crossfit_drift import basis

FIELDS=('anchor','rate','acceleration','knee_h')


def transformed(rows,logarithmic):
    p=np.array([[r[f] for f in FIELDS] for r in rows],dtype=float)
    if not np.isfinite(p).all() or np.any(p[:,1:]<0):raise ValueError('invalid training parameters')
    fraction=np.clip((p[:,0]-.5)/.75,1e-6,1-1e-6)
    p[:,0]=np.log(fraction/(1-fraction))
    if logarithmic:p[:,1:]=np.log(np.maximum(p[:,1:],1e-6))
    return p


def parameters_from(u,logarithmic):
    u=np.asarray(u,dtype=float);p=u.copy();p[...,0]=.5+.75*expit(u[...,0])
    with np.errstate(over='raise',invalid='raise'):
        p[...,1:]=np.exp(u[...,1:]) if logarithmic else np.maximum(u[...,1:],0)
    if not np.isfinite(p).all():raise ValueError('nonfinite predicted parameters')
    return p


def trajectory(p,cycle):
    h=basis(cycle,1)[:,0];a,b,c,knee=p
    hinge=.1*np.logaddexp(0,(h-knee)/.1)
    y=a-b*h-c*hinge**2
    if not np.isfinite(y).all():raise ValueError('nonfinite SOH')
    return y


def fit(cells,ids,alpha,logarithmic,seed,parameter_fits=None):
    ids=list(ids);encoder=encode_fit(cells,ids,2)
    rows=derive_prior(cells,ids)[1] if parameter_fits is None else parameter_fits
    if [r['cell_id'] for r in rows]!=ids:raise ValueError('training parameter membership mismatch')
    x=np.stack([encode(encoder,cells[k]) for k in ids]);y=transformed(rows,logarithmic)
    mean=y.mean(axis=0);scale=np.maximum(y.std(axis=0),1e-6)
    model=Ridge(alpha=alpha,solver='svd').fit(x,(y-mean)/scale)
    return dict(mode='parameters',encoder=encoder,model=model,target_mean=mean,target_scale=scale,training_parameter_fits=rows,train_ids=ids,family=encoder['family'],alpha=alpha,logarithmic=logarithmic,seed=seed)


def predict(b,cell,cycle):
    if b['mode']=='control':return tree_predict(b['base'],cell,cycle)
    x=encode(b['encoder'],cell)[None]
    u=b['model'].predict(x)[0]*b['target_scale']+b['target_mean']
    return trajectory(parameters_from(u,b['logarithmic']),cycle)
