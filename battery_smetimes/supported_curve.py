"""Exact leaf-weight donor curves with train-support linear tails and separate anchor."""
import numpy as np
from scipy.special import expit
from battery_smetimes.continuous_length import features,basis,predict as point_predict


def retrofit(shape,anchor,cells,targets):
    ids=shape['train_ids'];assert anchor['train_ids']==ids
    x=np.stack([features(cells[k],shape['origin'],shape['block']) for k in ids])
    z=shape['scaler'].transform(shape['imputer'].transform(x))
    return dict(kind='supported_curve',shape=shape,anchor=anchor,train_ids=ids,family=shape['family'],
        training_leaves=shape['model'].apply(z),parameters=np.array([r['parameters'] for r in shape['training_shape_fits']]),
        support_end=np.array([targets[k][0][-1] for k in ids]),anchor_cycle=min(targets[k][0][0] for k in ids))


def weights(b,cell):
    s=b['shape'];x=features(cell,s['origin'],s['block'])[None];z=s['scaler'].transform(s['imputer'].transform(x))
    leaves=s['model'].apply(z)[0];w=np.zeros(len(b['train_ids']))
    for j,leaf in enumerate(leaves):
        hit=b['training_leaves'][:,j]==leaf
        if not hit.any():raise ValueError('missing train support')
        w[hit]+=1/hit.sum()/len(leaves)
    return w


def donor_curve(c,origin,params,end):
    c=np.asarray(c,float);bounded=np.minimum(c,end)
    y=basis(bounded,origin,params[3])@params[:3]
    h=(end-origin)/100;soft=.2*np.logaddexp(0,(h-params[3])/.2)
    slope=(-params[1]-2*params[2]*soft*expit((h-params[3])/.2))/100
    return y+slope*np.maximum(c-end,0)


def predict(b,cell,cycle):
    if cell['physical_family']!=b['family']:raise ValueError('physical family mismatch')
    c=np.asarray(cycle,float);start=b['anchor_cycle']
    if c.ndim!=1 or not len(c) or not np.isfinite(c).all() or np.any(c<start):raise ValueError('invalid future')
    w=weights(b,cell);a=point_predict(b['anchor'],cell,np.array([start]))[0];delta=np.zeros_like(c)
    for i in np.flatnonzero(w):
        params=b['parameters'][i];end=b['support_end'][i]
        delta+=w[i]*(donor_curve(c,b['shape']['origin'],params,end)-donor_curve([start],b['shape']['origin'],params,end)[0])
    return a+delta
