"""Training-target-aware PLS mapping from early signals to smooth-knee parameters."""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from battery_smetimes.conditional_drift import prefix
from battery_smetimes.parameter_ridge import transformed,parameters_from,trajectory
from battery_smetimes.tail_boost import predict as tree_predict


def fit(cells,ids,components,logarithmic,seed,parameter_fits):
    ids=list(ids);families={cells[k]['physical_family'] for k in ids}
    if len(families)!=1:raise ValueError('different physical families')
    if [r['cell_id'] for r in parameter_fits]!=ids:raise ValueError('training parameter membership mismatch')
    raw=np.stack([prefix(cells[k]) for k in ids]);imputer=SimpleImputer(strategy='median',keep_empty_features=True)
    x=imputer.fit_transform(raw);scaler=StandardScaler();x=scaler.fit_transform(x)
    y=transformed(parameter_fits,logarithmic);mean=y.mean(axis=0);scale=np.maximum(y.std(axis=0),1e-6)
    count=min(components,len(ids)-1,x.shape[1])
    if count<1:raise ValueError('insufficient training cells')
    model=PLSRegression(n_components=count,scale=False,max_iter=1000,tol=1e-8).fit(x,(y-mean)/scale)
    return dict(mode='pls',model=model,imputer=imputer,scaler=scaler,target_mean=mean,target_scale=scale,training_parameter_fits=parameter_fits,train_ids=ids,family=next(iter(families)),components=components,logarithmic=logarithmic,seed=seed)


def predicted_parameters(b,cell):
    if cell['physical_family']!=b['family']:raise ValueError('different physical family')
    x=b['scaler'].transform(b['imputer'].transform(prefix(cell)[None]))
    return parameters_from(b['model'].predict(x)[0]*b['target_scale']+b['target_mean'],b['logarithmic'])


def predict(b,cell,cycle):
    if b['mode']=='control':return tree_predict(b['base'],cell,cycle)
    return trajectory(predicted_parameters(b,cell),cycle)
