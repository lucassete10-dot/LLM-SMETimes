"""Pace-supervised same-family retrieval with bounded-curvature extension."""
import numpy as np
from battery_smetimes.mit_pace_forecast import head,FEATURES
from battery_smetimes.mit_segment_features import prefix,condition_key
from battery_smetimes.mit_fixed_origin_dynamics import origin
from battery_smetimes.mit_template_tangent import curve


def latent(b,cell):
    x=prefix(cell,FEATURES)[None]
    value=b['model'].predict(b['scaler'].transform(b['imputer'].transform(x)))[0]
    return float(np.clip(value,b['lower'],b['upper']))


def fit(cache,cells,alpha):
    ids=cache['train_ids'];families={cells[k]['physical_family'] for k in ids}
    if len(families)!=1 or cache['family'] not in families:raise ValueError('different family')
    if [t['cell_id'] for t in cache['templates']]!=ids:raise ValueError('template membership')
    pace=head(cells,ids,alpha)
    return dict(train_ids=ids,family=cache['family'],templates=cache['templates'],frozen=cache['frozen'],
        pace_head=pace,donor_coordinates=np.array([latent(pace,cells[k]) for k in ids]),alpha=alpha,neighbors=3)


def route(b,cell):
    if cell['physical_family']!=b['family']:raise ValueError('different family')
    local=[i for i,t in enumerate(b['templates']) if t['condition']==condition_key(cell)]
    pool=local if len(local)>=6 else list(range(len(b['train_ids'])))
    query=latent(b['pace_head'],cell);d=abs(b['donor_coordinates'][pool]-query)
    pick=np.argsort(d,kind='stable')[:min(3,len(pool))];ii=np.array(pool)[pick];w=1/(d[pick]+.001);w/=w.sum()
    return ii,d[pick],w,query,'same_condition' if len(local)>=6 else 'family_fallback'


def predict(b,cell,future):
    c=np.asarray(future,dtype=float)
    if c.ndim!=1 or not len(c) or not np.isfinite(c).all() or np.any(c<=100):raise ValueError('invalid future')
    ii,_,w,_,_=route(b,cell);changes=[]
    for j in ii:
        t=b['templates'][j];changes.append(curve(t['parameters'],t['record_end'],c)-curve(t['parameters'],t['record_end'],[101.])[0])
    p=origin(b['frozen'],cell)+np.sum(np.stack(changes)*w[:,None],axis=0)
    if not np.isfinite(p).all():raise ValueError('invalid prediction')
    return p


def routing_record(b,cell):
    ii,d,w,q,scope=route(b,cell)
    return dict(neighbor_ids=[b['train_ids'][i] for i in ii],distances=d.tolist(),weights=w.tolist(),
        predicted_logpace=q,scope=scope)
