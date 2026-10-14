"""Feature-randomized trees; paired fixed early temporal statistics."""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesRegressor
from battery_smetimes.tail_boost import labels,predict as tree_predict
from battery_smetimes.temporal_features import matrix


def training_arrays(cells,ids,augmented):
    xs=[];ys=[];ws=[]
    for key in ids:
        c,y=labels(cells[key]);idx=np.unique(np.linspace(0,len(y)-1,min(96,len(y))).round().astype(int))
        xs.append(matrix(cells[key],c[idx],augmented));ys.append(y[idx])
        w=np.linspace(1,3,len(idx));ws.append(w/w.sum())
    x,y,w=np.concatenate(xs),np.concatenate(ys),np.concatenate(ws);w*=len(w)/w.sum()
    return x,y,w


def fit(cells,ids,max_features,augmented,seed):
    ids=list(ids);families={cells[k]['physical_family'] for k in ids}
    if len(families)!=1:raise ValueError('different physical families')
    x,y,w=training_arrays(cells,ids,augmented)
    imputer=SimpleImputer(strategy='median',keep_empty_features=True);z=imputer.fit_transform(x.astype(float))
    model=ExtraTreesRegressor(n_estimators=128,min_samples_leaf=1,max_depth=None,max_features=max_features,random_state=seed,n_jobs=2)
    model.fit(z,y,sample_weight=w)
    return dict(mode='forest',model=model,imputer=imputer,train_ids=ids,family=next(iter(families)),max_features=max_features,augmented=augmented,seed=seed)


def predict(b,cell,cycle):
    if b['mode']=='control':return tree_predict(b['base'],cell,cycle)
    if cell['physical_family']!=b['family']:raise ValueError('different physical family')
    p=b['model'].predict(b['imputer'].transform(matrix(cell,cycle,b['augmented'])))
    if not np.isfinite(p).all():raise ValueError('nonfinite forecast')
    return p
