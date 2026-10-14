"""Direct whole-curve regression with signal-conditioned fixed hinges."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from battery_smetimes.conditional_drift import encode_fit,encode
from battery_smetimes.crossfit_drift import basis
from battery_smetimes.tail_boost import labels,predict as tree_predict

KNOTS=(.25,.5,1.,2.,4.,8.)


def time_basis(cycle):
    h=basis(cycle,1)[:,0]
    return np.column_stack([h,np.log1p(h),*(np.maximum(h-t,0)**2 for t in KNOTS)])


def query(encoder,cell,cycle):
    x=time_basis(cycle);z=encode(encoder,cell)
    return np.column_stack([x,np.tile(z,(len(x),1)),*(x[:,i,None]*z[None] for i in range(x.shape[1]))])


def training_arrays(cells,ids,encoder):
    xs=[];ys=[];ws=[]
    for key in ids:
        x,y=labels(cells[key]);idx=np.unique(np.linspace(0,len(y)-1,min(96,len(y))).round().astype(int))
        xs.append(query(encoder,cells[key],x[idx]));ys.append(y[idx]);w=np.linspace(1,3,len(idx));ws.append(w/w.sum())
    x,y,w=np.concatenate(xs),np.concatenate(ys),np.concatenate(ws);w*=len(w)/w.sum()
    return x,y,w


def fit(cells,ids,components,alpha):
    encoder=encode_fit(cells,ids,components);x,y,w=training_arrays(cells,ids,encoder)
    scaler=StandardScaler();z=scaler.fit_transform(x,sample_weight=w)
    model=Ridge(alpha=alpha,solver='svd');model.fit(z,y,sample_weight=w)
    return dict(mode='spline',encoder=encoder,scaler=scaler,model=model,train_ids=list(ids),
        components=components,alpha=alpha,knots=KNOTS)


def predict(bundle,cell,cycle):
    if bundle['mode']=='control':return tree_predict(bundle['base'],cell,cycle)
    p=bundle['model'].predict(bundle['scaler'].transform(query(bundle['encoder'],cell,cycle)))
    if not np.isfinite(p).all():raise ValueError('nonfinite prediction')
    return p
