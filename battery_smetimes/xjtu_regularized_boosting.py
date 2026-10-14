"""Fixed-loss regularized direct prediction on existing early VIT segments."""
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from battery_smetimes.xjtu_segment_features import training_arrays,inputs
from battery_smetimes.curve_fidelity import condition_key

FEATURES=dict(block='VIT',context='segments')


def model_parameters(loss,seed):
    if loss not in ('squared_error','absolute_error'):raise ValueError('unregistered loss')
    return dict(loss=loss,max_iter=200,learning_rate=.05,max_leaf_nodes=15,max_depth=None,
        min_samples_leaf=20,l2_regularization=10,max_bins=64,early_stopping=False,random_state=seed)


def fit(cells,ids,loss,seed,local=True):
    ids=list(ids);families={cells[k]['physical_family'] for k in ids}
    if len(families)!=1:raise ValueError('different family')
    x,y,w=training_arrays(cells,ids,FEATURES);imputer=SimpleImputer(strategy='median',keep_empty_features=True);z=imputer.fit_transform(x)
    model=HistGradientBoostingRegressor(**model_parameters(loss,seed)).fit(z,y,sample_weight=w)
    groups={}
    for k in ids:groups.setdefault(condition_key(cells[k]),[]).append(k)
    locals={k:fit(cells,v,loss,seed,False) for k,v in groups.items() if local and len(v)>=6}
    return dict(imputer=imputer,model=model,local=locals,train_ids=ids,loss=loss,seed=seed,family=next(iter(families)))


def predict(b,cell,cycle):
    if cell['physical_family']!=b['family']:raise ValueError('different family')
    key=condition_key(cell)
    if key in b['local']:return predict(b['local'][key],cell,cycle)
    return b['model'].predict(b['imputer'].transform(inputs(cell,cycle,FEATURES)))
