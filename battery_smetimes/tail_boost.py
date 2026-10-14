"""Tail-weighted whole-curve regressors with cell/family isolation."""
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor,HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from battery_smetimes.curve_fidelity import targets,condition_key
from battery_smetimes.research_baselines import signal_prefix
from battery_smetimes.forecast import query_matrix
from scripts.prepare_zhu2022_nca import prefix
from scripts.train_zhu2022_nca import matrix as nca_matrix


def is_nca(cell):return 'early_raw' in cell


def conditions(cell):
    return tuple(cell['conditions']) if is_nca(cell) else condition_key(cell)


def labels(cell):
    return (cell['cycle'],cell['truth']) if is_nca(cell) else targets(cell)


def inputs(cell,future):
    if is_nca(cell):
        cycle=np.asarray(cell['early_cycle']);raw=np.asarray(cell['early_raw']);take=cycle<=100
        clean=dict(cell,prefix=prefix(cycle[take],raw[take],cell['conditions']))
        return nca_matrix(clean,future)
    return query_matrix(signal_prefix(cell,100),100,future).astype(float)


def train_arrays(cells,ids,tail_weight):
    xs=[];ys=[];weights=[]
    for key in ids:
        x,y=labels(cells[key]);idx=np.unique(np.linspace(0,len(y)-1,min(96,len(y))).round().astype(int))
        xs.append(inputs(cells[key],x[idx]));ys.append(y[idx])
        w=np.linspace(1,tail_weight,len(idx));weights.append(w/w.sum())
    x,y,w=np.concatenate(xs),np.concatenate(ys),np.concatenate(weights)
    w*=len(w)/w.sum()
    return x,y,w


def fit(cells,ids,parameters,seed):
    ids=list(ids);family=cells[ids[0]]['physical_family']
    if any(cells[k]['physical_family']!=family for k in ids):raise ValueError('different physical families')
    x,y,w=train_arrays(cells,ids,parameters['tail_weight'])
    imputer=SimpleImputer(strategy='median',keep_empty_features=True);z=imputer.fit_transform(x.astype(float))
    if parameters['model']=='extra_trees':
        estimator=ExtraTreesRegressor(n_estimators=128,min_samples_leaf=parameters['leaf'],max_depth=None,
            random_state=seed,n_jobs=2)
    else:
        estimator=HistGradientBoostingRegressor(max_iter=120,learning_rate=.05,max_leaf_nodes=parameters['max_leaves'],
            min_samples_leaf=20,l2_regularization=1,max_bins=64,early_stopping=False,random_state=seed)
    estimator.fit(z,y,sample_weight=w)
    local={}
    if parameters['local']:
        groups={}
        for key in ids:groups.setdefault(conditions(cells[key]),[]).append(key)
        for protocol,members in groups.items():
            if len(members)>=6:local[protocol]=fit(cells,members,dict(parameters,local=False),seed)
    return dict(model=estimator,imputer=imputer,local=local,train_ids=ids,
        family=family,parameters=parameters,seed=seed)


def predict(bundle,cell,future):
    if cell['physical_family']!=bundle['family']:raise ValueError('different physical family')
    if conditions(cell) in bundle['local']:return predict(bundle['local'][conditions(cell)],cell,future)
    p=bundle['model'].predict(bundle['imputer'].transform(inputs(cell,future)))
    if not np.isfinite(p).all():raise ValueError('nonfinite prediction')
    return p
