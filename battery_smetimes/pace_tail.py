"""Continuous tail beyond known-protocol training support, conditioned on early signals."""
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from battery_smetimes import condition_length as local
from battery_smetimes.continuous_length import features


def fit(cells, ids, targets, origin, seed):
    base=local.fit(cells,ids,targets,origin,'point',seed)
    eligible=[]; labels=[]; support={}
    for cid in ids:
        c,y=targets[cid]; group=local.key(cells[cid])
        support[group]=max(support.get(group,origin),float(c[-1]))
        m=(c>=101)&(c<=150); cc,yy=c[m],y[m]
        if len(cc)>=3 and cc[-1]-cc[0]>=20:
            centered=cc-cc.mean()
            labels.append(float(-100*centered@(yy-yy.mean())/(centered@centered)));eligible.append(cid)
    if len(eligible)<3:raise ValueError('insufficient pace supervision')
    encoder=OneHotEncoder(sparse_output=False,handle_unknown='ignore')
    known=encoder.fit_transform([[local.key(cells[cid])] for cid in eligible])
    x=np.c_[np.stack([features(cells[cid],origin,'VIT') for cid in eligible]),known]
    imp=SimpleImputer(strategy='median',keep_empty_features=True); z=imp.fit_transform(x)
    sc=StandardScaler().fit(z)
    model=ExtraTreesRegressor(n_estimators=80,min_samples_leaf=2,max_features=.8,random_state=seed,n_jobs=2)
    model.fit(sc.transform(z),labels)
    return dict(base=base,origin=origin,train_ids=list(ids),pace_train_ids=eligible,pace_labels=labels,
        support=support,encoder=encoder,imputer=imp,scaler=sc,model=model,family=base['family'])


def predict(b,cell,cycles,extend=True):
    c=np.asarray(cycles,float)
    p=local.predict(b['base'],cell,c)
    if not extend:return p
    end=b['support'].get(local.key(cell))
    if end is None:return p  # No extrapolation policy asserted for unseen protocol.
    m=c>end
    if m.any():
        x=np.c_[features(cell,b['origin'],'VIT')[None],b['encoder'].transform([[local.key(cell)]])]
        rate=max(0.,float(b['model'].predict(b['scaler'].transform(b['imputer'].transform(x)))[0]))
        anchor=local.predict(b['base'],cell,np.array([end]))[0]
        p[m]=anchor-rate*(c[m]-end)/100
    return p
