"""Signal-only transformed fade-rate tree with fixed or predicted anchor."""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesRegressor
from battery_smetimes.tail_boost import inputs,train_arrays,predict as old_predict
from battery_smetimes.conditional_drift import prefix


def tree(seed):
    return ExtraTreesRegressor(n_estimators=128,min_samples_leaf=1,max_depth=None,random_state=seed,n_jobs=2)


def fit(cells,ids,learned_anchor,seed,parameter_fits):
    ids=list(ids);families={cells[k]['physical_family'] for k in ids}
    if len(families)!=1:raise ValueError('different physical families')
    if [r['cell_id'] for r in parameter_fits]!=ids:raise ValueError('training parameter membership mismatch')
    anchors=np.array([r['anchor'] for r in parameter_fits]) if learned_anchor else np.ones(len(ids))
    if not np.isfinite(anchors).all():raise ValueError('invalid anchors')
    x,y,w=train_arrays(cells,ids,3);from battery_smetimes.tail_boost import labels
    cycle=[];row_anchor=[]
    for key,a in zip(ids,anchors):
        c,_=labels(cells[key]);idx=np.unique(np.linspace(0,len(c)-1,min(96,len(c))).round().astype(int))
        cycle.extend(c[idx]);row_anchor.extend([a]*len(idx))
    denominator=1+(np.asarray(cycle)-100)/100
    rate=(np.asarray(row_anchor)-y)/denominator
    imputer=SimpleImputer(strategy='median',keep_empty_features=True);model=tree(seed).fit(imputer.fit_transform(x),rate,sample_weight=w)
    anchor_imputer=None;anchor_model=None
    if learned_anchor:
        z=np.stack([prefix(cells[k]) for k in ids]);anchor_imputer=SimpleImputer(strategy='median',keep_empty_features=True)
        anchor_model=tree(seed).fit(anchor_imputer.fit_transform(z),anchors)
    return dict(mode='rate',model=model,imputer=imputer,anchor_model=anchor_model,anchor_imputer=anchor_imputer,training_anchors=anchors,train_ids=ids,family=next(iter(families)),learned_anchor=learned_anchor,seed=seed)


def anchor(bundle,cell):
    if cell['physical_family']!=bundle['family']:raise ValueError('different physical family')
    if not bundle['learned_anchor']:return 1.
    return float(bundle['anchor_model'].predict(bundle['anchor_imputer'].transform(prefix(cell)[None]))[0])


def predict(bundle,cell,cycle):
    if bundle['mode']=='control':return old_predict(bundle['base'],cell,cycle)
    a=anchor(bundle,cell);cycle=np.asarray(cycle,dtype=float)
    if cycle.ndim!=1 or not len(cycle) or not np.isfinite(cycle).all() or np.any(cycle<=100):raise ValueError('invalid future coordinates')
    rate=bundle['model'].predict(bundle['imputer'].transform(inputs(cell,cycle)))
    p=a-(1+(cycle-100)/100)*rate
    if not np.isfinite(p).all():raise ValueError('nonfinite forecast')
    return p
