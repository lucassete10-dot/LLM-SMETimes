"""First100-conditioned bounded relative residuals, with fixed cycle101 anchor."""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from battery_smetimes.xjtu_segment_features import prefix,predict as base_predict

FEATURES=dict(block='VIT',context='segments')


def basis(cycle):
    c=np.asarray(cycle,dtype=float)
    if c.ndim!=1 or not len(c) or not np.isfinite(c).all() or np.any(c<=100):raise ValueError('invalid future')
    h=(c-101)/100
    return np.column_stack([h,h*h/(1+h)])


def target(record):
    c,y,p=[np.asarray(record[f]) for f in ('cycle','truth','prediction')]
    ii=np.unique(np.linspace(0,len(y)-1,min(96,len(y))).round().astype(int))
    w=np.linspace(1,3,len(ii));w=.5*w/w.sum()
    x=basis(c[ii])-basis(c[:1])[0];r=(y[ii]-y[0])-(p[ii]-p[0])
    xe=basis(c[-1:])[0]-basis(c[:1])[0];re=(y[-1]-y[0])-(p[-1]-p[0])
    lhs=x.T@(w[:,None]*x)+.5*np.outer(xe,xe)+.01*np.eye(2)
    rhs=x.T@(w*r)+.5*xe*re
    return np.clip(np.linalg.solve(lhs,rhs),-.1,.1)


def fit(base,cells,ids,records,alpha):
    ids=list(ids)
    if set(records)!=set(ids) or any(cells[k]['physical_family']!=base['family'] for k in ids):raise ValueError('invalid training family/targets')
    y=np.stack([target(records[k]) for k in ids]);x=np.stack([prefix(cells[k],FEATURES) for k in ids])
    imputer=SimpleImputer(strategy='median',keep_empty_features=True);z=imputer.fit_transform(x)
    scaler=StandardScaler();z=scaler.fit_transform(z);model=Ridge(alpha=alpha).fit(z,y)
    return dict(base=base,imputer=imputer,scaler=scaler,model=model,supervision=y,
        lower=y.min(0),upper=y.max(0),alpha=alpha,train_ids=ids,family=base['family'])


def correction(b,cell,cycle):
    if cell['physical_family']!=b['family']:raise ValueError('different family')
    z=b['scaler'].transform(b['imputer'].transform(prefix(cell,FEATURES)[None]))
    a=np.clip(b['model'].predict(z)[0],b['lower'],b['upper'])
    return .04*np.tanh((basis(cycle)@a)/.04)


def predict(b,cell,cycle):return base_predict(b['base'],cell,cycle)+correction(b,cell,cycle)
