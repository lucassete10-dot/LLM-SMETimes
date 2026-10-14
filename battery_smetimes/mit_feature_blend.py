"""Locked uniform-weight blend within one MIT physical family."""
from sklearn.base import BaseEstimator
from battery_smetimes.mit_segment_features import fit as feature_fit,predict as feature_predict
from battery_smetimes.tail_selection import fit as legacy_fit,predict as legacy_predict

PARAMETERS=dict(block='V',context='segments')
WEIGHTS=(0.,.25,.5,.75,1.)

def fit(cells,ids,choice,weight,seed):
    ids=list(ids)
    if weight not in WEIGHTS:raise ValueError('unregistered weight')
    families={cells[k]['physical_family'] for k in ids}
    if len(families)!=1 or any(cells[k]['metadata']['dataset']!='MIT' for k in ids):raise ValueError('MIT physical family required')
    old=legacy_fit(cells,ids,choice,seed) if weight<1 else None
    new=feature_fit(cells,ids,PARAMETERS,seed) if weight>0 else None
    return dict(mode='mit_feature_blend',old=old,new=new,new_weight=weight,train_ids=ids,
        family=next(iter(families)),seed=seed,parameters=PARAMETERS)

def predict(bundle,cell,future):
    if cell['physical_family']!=bundle['family']:raise ValueError('different physical family')
    w=bundle['new_weight']
    if w not in WEIGHTS:raise ValueError('unregistered weight')
    if w==0:return legacy_predict(bundle['old'],cell,future)
    if w==1:return feature_predict(bundle['new'],cell,future)
    return (1-w)*legacy_predict(bundle['old'],cell,future)+w*feature_predict(bundle['new'],cell,future)

def estimator_components(bundle):
    """Count whole fitted predictors, not individual trees or imputers."""
    seen=set()
    def count(v):
        if id(v) in seen:return 0
        seen.add(id(v))
        if isinstance(v,BaseEstimator) and hasattr(v,'predict') and hasattr(v,'fit'):return 1
        if isinstance(v,dict):return sum(count(x) for x in v.values())
        if isinstance(v,(tuple,list)):return sum(count(x) for x in v)
        return 0
    return count(bundle)
