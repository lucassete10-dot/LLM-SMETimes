"""Bounded two-parameter TRAIN-OOF gain, not query-lifespan normalization."""
import numpy as np
from scipy.optimize import lsq_linear
from battery_smetimes.xjtu_segment_features import predict as base_predict


def phi(cycle):
    c=np.asarray(cycle,dtype=float)
    if c.ndim!=1 or not len(c) or not np.isfinite(c).all() or np.any(c<=100):raise ValueError('invalid future')
    h=(c-100)/100
    return h/(1+h)


def fit_coefficients(records,penalty):
    xs=[];ys=[]
    for r in records:
        c,y,p=[np.asarray(r[f]) for f in ('cycle','truth','prediction')]
        ii=np.unique(np.linspace(0,len(y)-1,min(96,len(y))).round().astype(int))
        w=np.linspace(1,3,len(ii));w=np.sqrt(w/(w.sum()*len(records)))
        dp=p[0]-p[ii];dy=y[0]-y[ii]
        xs.append(np.column_stack([dp,dp*phi(c[ii])])*w[:,None]);ys.append(dy*w)
    design=np.vstack([np.concatenate(xs),np.eye(2)*np.sqrt(penalty)])
    target=np.r_[np.concatenate(ys),np.sqrt(penalty),0.]
    result=lsq_linear(design,target,bounds=([.5,0],[1.5,.5]),tol=1e-12,max_iter=500)
    if not result.success or not np.isfinite(result.x).all():raise ValueError('calibration solver failed')
    return result.x,dict(cost=float(result.cost),optimality=float(result.optimality),status=int(result.status),
        message=result.message,weighted_design_rows=len(target))


def predict(b,cell,cycle):
    if cell['physical_family']!=b['base']['family']:raise ValueError('different family')
    f=phi(cycle);p=base_predict(b['base'],cell,cycle);a=base_predict(b['base'],cell,[101.])[0]
    gain=b['coefficients'][0]+b['coefficients'][1]*f
    return a+gain*(p-a)
