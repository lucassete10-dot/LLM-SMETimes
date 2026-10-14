"""Low-rank signal-conditioned OOF residual predictor."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from battery_smetimes.tail_boost import inputs,predict as base_predict
from battery_smetimes.crossfit_drift import basis


def prefix(cell):
    return inputs(cell,np.array([101.]))[0]


def encode_fit(cells,ids,components):
    families={cells[k]['physical_family'] for k in ids}
    if len(families)!=1:raise ValueError('different physical families')
    x=np.stack([prefix(cells[k]) for k in ids])
    imputer=SimpleImputer(strategy='median',keep_empty_features=True);x=imputer.fit_transform(x)
    scaler=StandardScaler();x=scaler.fit_transform(x)
    count=min(components,len(ids)-1,x.shape[1])
    if count<1:raise ValueError('insufficient training cells')
    pca=PCA(n_components=count,svd_solver='full',whiten=False);pca.fit(x)
    return dict(imputer=imputer,scaler=scaler,pca=pca,train_ids=list(ids),family=next(iter(families)))


def encode(encoder,cell):
    if cell['physical_family']!=encoder['family']:raise ValueError('different physical family')
    return encoder['pca'].transform(encoder['scaler'].transform(encoder['imputer'].transform(prefix(cell)[None])))[0]


def query(encoder,cell,cycle,degree):
    h=basis(cycle,degree);z=encode(encoder,cell)
    return np.column_stack([h,np.tile(z,(len(h),1)),*(h[:,i,None]*z[None] for i in range(degree))])


def fit_head(cells,ids,residuals,components,degree,alpha=100):
    encoder=encode_fit(cells,ids,components);xs=[];ys=[];ws=[]
    for key in ids:
        item=residuals[key];x=item['cycle'];r=item['truth']-item['prediction']
        xs.append(query(encoder,cells[key],x,degree));ys.append(r)
        w=np.linspace(1,3,len(x));ws.append(w/w.sum())
    x,y,w=np.concatenate(xs),np.concatenate(ys),np.concatenate(ws);w*=len(w)/w.sum()
    scaler=StandardScaler();z=scaler.fit_transform(x)
    model=Ridge(alpha=alpha);model.fit(z,y,sample_weight=w)
    return dict(encoder=encoder,scaler=scaler,model=model,degree=degree,components=components,
        alpha=alpha,train_ids=list(ids))


def predict(bundle,cell,cycle):
    p=base_predict(bundle['base'],cell,cycle)
    if bundle['head'] is not None:
        head=bundle['head'];p=p+head['model'].predict(head['scaler'].transform(query(head['encoder'],cell,cycle,head['degree'])))
    if not np.isfinite(p).all():raise ValueError('nonfinite forecast')
    return p
