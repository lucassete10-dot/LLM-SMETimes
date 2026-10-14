"""Training-cell OOF drift correction; inference has no labels or true life."""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from battery_smetimes.tail_boost import predict as base_predict


def split_ids(ids, seed):
    ids=list(ids)
    if len(ids)<2 or len(ids)!=len(set(ids)):raise ValueError('need distinct training cells')
    order=np.random.default_rng(seed).permutation(len(ids))
    groups=[[ids[i] for i in part] for part in np.array_split(order,2)]
    return [dict(train=[k for k in ids if k not in held],held=held) for held in groups]


def basis(cycle, degree):
    cycle=np.asarray(cycle,dtype=float)
    if cycle.ndim!=1 or not len(cycle) or not np.isfinite(cycle).all() or np.any(cycle<=100):
        raise ValueError('request must be finite future source cycles')
    h=(cycle-100)/100
    return np.column_stack([h**power for power in range(1,degree+1)])


def fit_head(cycle, residual, weights, degree, alpha):
    z=basis(cycle,degree);scaler=StandardScaler();z=scaler.fit_transform(z)
    model=Ridge(alpha=alpha);model.fit(z,residual,sample_weight=weights)
    return dict(model=model,scaler=scaler,degree=degree,alpha=alpha)


def predict(bundle,cell,cycle):
    p=base_predict(bundle['base'],cell,cycle)
    if bundle['head'] is not None:
        head=bundle['head'];p=p+head['model'].predict(head['scaler'].transform(basis(cycle,head['degree'])))
    if not np.isfinite(p).all():raise ValueError('nonfinite prediction')
    return p
