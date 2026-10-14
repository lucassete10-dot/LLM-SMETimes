"""Training-cell current-state anchor plus early-signal future-fade regression."""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from battery_smetimes.research_baselines import signal_prefix


def fade_features(prefix, origin, future):
    future = np.asarray(future, dtype=float)
    if future.ndim != 1 or not len(future) or not np.isfinite(future).all() or np.any(future <= origin):
        raise ValueError('invalid future query')
    h = (future-origin)/100
    p = np.broadcast_to(np.asarray(prefix, dtype=float), (len(h),len(prefix)))
    return np.column_stack([p, h, h*h, p*h[:,None]])


def fit_head(x, y, weight, alpha):
    imputer = SimpleImputer(strategy='median', keep_empty_features=True)
    z = imputer.fit_transform(np.asarray(x, dtype=float))
    scaler = StandardScaler()
    offset = z[0].copy()
    scaler.fit(z-offset, sample_weight=weight)
    scaler.mean_ += offset
    model = Ridge(alpha=alpha, solver='svd').fit(scaler.transform(z), y, sample_weight=weight)
    return dict(imputer=imputer, scaler=scaler, model=model)


def apply_head(head, x):
    return head['model'].predict(head['scaler'].transform(head['imputer'].transform(np.asarray(x,dtype=float))))


def fit(cells, ids, parameters):
    family = cells[ids[0]]['physical_family']
    assert all(cells[k]['physical_family'] == family for k in ids)
    sx, sy, sw, fx, fy, fw, anchors = [], [], [], [], [], [], []
    for key in ids:
        c = cells[key]
        good = np.isfinite(c['soh']) & (c['soh'] > 0)
        idx = np.flatnonzero(good & (c['label_cycle'] >= 10) & (c['label_cycle'] <= 100))
        if not len(idx):
            raise ValueError('missing training state labels')
        for i in idx:
            origin = float(c['label_cycle'][i])
            sx.append(signal_prefix(c, origin)); sy.append(float(c['soh'][i])); sw.append(1/len(idx))
        cell_fx, cell_fy = [], []
        for origin in range(20,101,10):
            exact = np.flatnonzero(good & (c['label_cycle'] == origin))
            if len(exact) != 1:
                continue
            take = np.flatnonzero(good & (c['label_cycle'] > origin))
            if len(take) < 2:
                continue
            if len(take) > 96:
                take = take[np.unique(np.linspace(0,len(take)-1,96).round().astype(int))]
            cell_fx.append(fade_features(signal_prefix(c,origin),origin,c['label_cycle'][take]))
            cell_fy.append(c['soh'][take]-c['soh'][exact[0]])
            anchors.append(dict(cell_id=key, origin=origin, target_count=len(take)))
        if not cell_fx:
            raise ValueError('missing fade supervision')
        for x,y in zip(cell_fx,cell_fy):
            fx.append(x); fy.append(y); fw.extend([1/(len(cell_fx)*len(y))]*len(y))
    sw = np.asarray(sw); sw *= len(sw)/sw.sum()
    fw = np.asarray(fw); fw *= len(fw)/fw.sum()
    return dict(family=family, train_ids=list(ids), parameters=dict(parameters), training_anchors=anchors,
        state=fit_head(np.stack(sx),np.asarray(sy),sw,parameters['state_alpha']),
        fade=fit_head(np.concatenate(fx),np.concatenate(fy),fw,parameters['fade_alpha']))


def predict(bundle, cell, future):
    if cell['physical_family'] != bundle['family']:
        raise ValueError('different physical family')
    prefix = signal_prefix(cell,100)
    baseline = float(apply_head(bundle['state'],prefix[None])[0])
    p = baseline+apply_head(bundle['fade'],fade_features(prefix,100,future))
    if not np.isfinite(p).all():
        raise ValueError('nonfinite forecast')
    return p
