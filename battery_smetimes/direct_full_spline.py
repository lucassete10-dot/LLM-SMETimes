"""Full early-signal interactions, avoiding unsupervised PCA information loss."""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from battery_smetimes.conditional_drift import prefix
from battery_smetimes.direct_spline import time_basis
from battery_smetimes.tail_boost import labels,predict as tree_predict


def query(encoder,cell,cycle):
    if cell['physical_family']!=encoder['family']:raise ValueError('different physical family')
    z=encoder['scaler'].transform(encoder['imputer'].transform(prefix(cell)[None]))[0]
    x=time_basis(cycle)
    return np.column_stack([x,np.tile(z,(len(x),1)),*(x[:,i,None]*z[None] for i in range(x.shape[1]))])


def training_arrays(cells,ids,encoder):
    xs=[];ys=[];ws=[]
    for key in ids:
        x,y=labels(cells[key]);idx=np.unique(np.linspace(0,len(y)-1,min(96,len(y))).round().astype(int))
        xs.append(query(encoder,cells[key],x[idx]));ys.append(y[idx]);w=np.linspace(1,3,len(idx));ws.append(w/w.sum())
    x,y,w=np.concatenate(xs),np.concatenate(ys),np.concatenate(ws);w*=len(w)/w.sum()
    return x,y,w


def fit(cells,ids,alpha):
    families={cells[k]['physical_family'] for k in ids}
    if len(families)!=1:raise ValueError('different physical families')
    raw=np.stack([prefix(cells[k]) for k in ids]);imputer=SimpleImputer(strategy='median',keep_empty_features=True)
    scaler=StandardScaler();scaler.fit(imputer.fit_transform(raw))
    encoder=dict(imputer=imputer,scaler=scaler,family=next(iter(families)),train_ids=list(ids))
    x,y,w=training_arrays(cells,ids,encoder);query_scaler=StandardScaler()
    # Center before weighted variance accumulation to avoid cancellation in
    # nearly constant signal/time-interaction columns; inference mean restored.
    offset=x[0].copy();query_scaler.fit(x-offset,sample_weight=w);query_scaler.mean_+=offset
    z=query_scaler.transform(x)
    model=Ridge(alpha=alpha,solver='lsqr',tol=1e-8,max_iter=5000);model.fit(z,y,sample_weight=w)
    return dict(mode='full_spline',encoder=encoder,scaler=query_scaler,model=model,train_ids=list(ids),alpha=alpha)


def predict(b,cell,cycle):
    if b['mode']=='control':return tree_predict(b['base'],cell,cycle)
    p=b['model'].predict(b['scaler'].transform(query(b['encoder'],cell,cycle)))
    if not np.isfinite(p).all():raise ValueError('nonfinite prediction')
    return p
