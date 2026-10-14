"""Same-family TRAIN templates routed only by first100 signal features."""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from battery_smetimes.mit_segment_features import prefix,condition_key
from battery_smetimes.mit_fixed_origin_dynamics import origin,PARAMETERS
from battery_smetimes.curve_fidelity import targets
from battery_smetimes.knee_forecast import fit_knee,trajectory


def fit(cells,ids,frozen):
    ids=list(ids);families={cells[k]['physical_family'] for k in ids}
    if len(families)!=1 or frozen['feature']['train_ids']!=ids:raise ValueError('origin membership/family')
    raw=np.stack([prefix(cells[k],PARAMETERS) for k in ids])
    imputer=SimpleImputer(strategy='median',keep_empty_features=True);scaler=StandardScaler()
    z=scaler.fit_transform(imputer.fit_transform(raw));templates=[]
    for k in ids:
        c,y=targets(cells[k]);assert c[0]==101
        templates.append(dict(cell_id=k,condition=condition_key(cells[k]),parameters=fit_knee(c,y),record_end=float(c[-1])))
    return dict(train_ids=ids,family=next(iter(families)),imputer=imputer,scaler=scaler,z=z,
        templates=templates,frozen=frozen,parameters=PARAMETERS)


def route(b,cell):
    if cell['physical_family']!=b['family']:raise ValueError('different family')
    key=condition_key(cell)
    local=[i for i,t in enumerate(b['templates']) if t['condition']==key]
    pool=local if len(local)>=6 else list(range(len(b['templates'])))
    x=b['scaler'].transform(b['imputer'].transform(prefix(cell,b['parameters'])[None]))[0]
    d=np.sqrt(np.mean((b['z'][pool]-x)**2,axis=1))
    order=np.argsort(d,kind='stable')[:min(b['neighbors'],len(pool))]
    ii=np.array(pool)[order];dist=d[order];w=1/(dist+.001);w/=w.sum()
    return ii,dist,w,'same_condition' if len(local)>=6 else 'family_fallback'


def predict(b,cell,future):
    c=np.asarray(future,dtype=float)
    if c.ndim!=1 or not len(c) or not np.isfinite(c).all() or np.any(c<=100):raise ValueError('invalid future')
    ii,_,w,_=route(b,cell);changes=[]
    for i in ii:
        p=b['templates'][i]['parameters']
        changes.append(trajectory(p,c)-trajectory(p,[101.])[0])
    out=origin(b['frozen'],cell)+np.sum(np.stack(changes)*w[:,None],axis=0)
    if not np.isfinite(out).all():raise ValueError('invalid forecast')
    return out


def routing_record(b,cell,future):
    ii,d,w,scope=route(b,cell);end=float(np.max(future))
    return dict(scope=scope,neighbor_ids=[b['train_ids'][i] for i in ii],distances=d.tolist(),weights=w.tolist(),
        neighbor_record_ends=[b['templates'][i]['record_end'] for i in ii],
        fraction_neighbors_extrapolated_at_query_end=float(np.mean([end>b['templates'][i]['record_end'] for i in ii])))
