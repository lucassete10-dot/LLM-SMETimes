"""Frozen signal-only origin, horizon-normalized fade head; NCA only."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from battery_smetimes.tail_boost import inputs


def fit(cells, ids, tail_weight, seed, origin):
    ids = list(ids)
    families = {cells[k]['physical_family'] for k in ids}
    if len(families) != 1 or origin['family'] not in families:
        raise ValueError('different physical family')
    if origin['train_ids'] != ids or origin['seed'] != seed:
        raise ValueError('origin training membership/seed mismatch')
    xs, ys, ws = [], [], []
    for k in ids:
        cell = cells[k]
        c, y = np.asarray(cell['cycle']), np.asarray(cell['truth'])
        if c[0] != 101 or c.shape != y.shape or not np.isfinite(y).all():
            raise ValueError('cycle-101 training labels required')
        ii = np.unique(np.linspace(0, len(y)-1, min(96, len(y))).round().astype(int))
        xs.append(inputs(cell, c[ii]))
        ys.append((y[0]-y[ii])/(1+(c[ii]-101)/100))
        w = np.linspace(1, tail_weight, len(ii))
        ws.append(w/w.sum())
    imputer = SimpleImputer(strategy='median', keep_empty_features=True)
    z = imputer.fit_transform(np.concatenate(xs))
    weights = np.concatenate(ws)
    weights *= len(weights)/weights.sum()
    model = HistGradientBoostingRegressor(max_iter=120, learning_rate=.05,
        max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1,
        max_bins=64, early_stopping=False, random_state=seed)
    model.fit(z, np.concatenate(ys), sample_weight=weights)
    return dict(level=origin['level'], level_imputer=origin['level_imputer'],
        rate=model, rate_imputer=imputer, family=origin['family'],
        train_ids=ids, tail_weight=tail_weight, seed=seed)


def predict(b, cell, future):
    if cell['physical_family'] != b['family']:
        raise ValueError('different physical family')
    c = np.asarray(future, dtype=float)
    if c.ndim != 1 or not len(c) or not np.isfinite(c).all() or np.any(c <= 100):
        raise ValueError('invalid future coordinates')
    origin = inputs(cell, np.array([101.]))
    level = b['level'].predict(b['level_imputer'].transform(origin))[0]
    rate = b['rate'].predict(b['rate_imputer'].transform(inputs(cell, c)))
    r0 = b['rate'].predict(b['rate_imputer'].transform(origin))[0]
    p = level-(1+(c-101)/100)*rate+r0
    if not np.isfinite(p).all():
        raise ValueError('invalid prediction')
    return p
