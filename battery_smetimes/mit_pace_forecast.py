"""OOF early-feature pace conditioning; no query capacity or lifetime input."""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold
from battery_smetimes.mit_segment_features import prefix,inputs
from battery_smetimes.curve_fidelity import targets

FEATURES=dict(block='V',context='segments')


def pace_label(cell):
    c,y=targets(cell)
    smooth=np.array([np.median(y[max(0,j-2):min(len(y),j+3)]) for j in range(len(y))])
    start=float(np.median(y[:5]));hit=np.flatnonzero(smooth<=start-.05)
    return float(c[hit[0]]-100) if len(hit) else None


def head(cells,ids,alpha):
    observed=[(k,pace_label(cells[k])) for k in ids]
    labeled=[(k,v) for k,v in observed if v is not None]
    if len(labeled)<10:raise ValueError('insufficient observed pace supervision')
    x=np.stack([prefix(cells[k],FEATURES) for k,v in labeled]);y=np.log([v for k,v in labeled])
    imputer=SimpleImputer(strategy='median',keep_empty_features=True);z=imputer.fit_transform(x)
    scaler=StandardScaler().fit(z);model=Ridge(alpha=alpha).fit(scaler.transform(z),y)
    return dict(imputer=imputer,scaler=scaler,model=model,lower=float(y.min()),upper=float(y.max()),
        train_ids=list(ids),labeled_ids=[k for k,v in labeled],unobserved_ids=[k for k,v in observed if v is None])


def pace_predict(bundle,cell):
    x=prefix(cell,FEATURES)[None,:]
    value=bundle['model'].predict(bundle['scaler'].transform(bundle['imputer'].transform(x)))[0]
    return float(np.exp(np.clip(value,bundle['lower'],bundle['upper'])))


def curve_inputs(cell,cycles,pace):
    x=inputs(cell,cycles,FEATURES)
    return np.column_stack([x,(np.asarray(cycles)-100)/pace])


def fit(cells,ids,alpha,seed):
    ids=list(ids);families={cells[k]['physical_family'] for k in ids}
    if len(families)!=1:raise ValueError('different family')
    oof={};folds=[]
    for tr,va in KFold(5,shuffle=True,random_state=seed).split(ids):
        train=[ids[j] for j in tr];held=[ids[j] for j in va]
        h=head(cells,train,alpha)
        assert not set(train)&set(held)
        oof.update({k:pace_predict(h,cells[k]) for k in held})
        folds.append(dict(train=train,held=held,head=h))
    final=head(cells,ids,alpha);xs=[];ys=[];ws=[]
    for k in ids:
        c,y=targets(cells[k]);ii=np.unique(np.linspace(0,len(y)-1,min(96,len(y))).round().astype(int))
        xs.append(curve_inputs(cells[k],c[ii],oof[k]));ys.append(y[ii])
        w=np.linspace(1,3,len(ii));ws.append(w/w.sum())
    x=np.concatenate(xs);y=np.concatenate(ys);w=np.concatenate(ws);w*=len(w)/w.sum()
    imputer=SimpleImputer(strategy='median',keep_empty_features=True);z=imputer.fit_transform(x)
    model=ExtraTreesRegressor(n_estimators=128,min_samples_leaf=1,random_state=seed,n_jobs=2).fit(z,y,sample_weight=w)
    return dict(pace_head=final,oof_folds=folds,oof_paces=oof,curve_imputer=imputer,
        curve_model=model,alpha=alpha,seed=seed,train_ids=ids,family=next(iter(families)))


def predict(bundle,cell,cycles):
    if cell['physical_family']!=bundle['family']:raise ValueError('different family')
    pace=pace_predict(bundle['pace_head'],cell)
    return bundle['curve_model'].predict(bundle['curve_imputer'].transform(curve_inputs(cell,cycles,pace)))
